"""Instrument unmodified training scripts (Runtime Monitor Agent tool).

``RuntimeHooks`` needs a model and an optimizer. A training *script* never hands
them to us, so :class:`AutoInstrumenter` finds them the way package101's tracer
does (``trainmedic/detect/tracer.py``), with *global* hooks that are removed
afterwards:

* an ``Optimizer.__init__`` wrapper records every optimizer and the line of the
  script that built it;
* global module forward pre/post hooks track call depth; the module called at
  depth 0 is a *root* model. On its first forward, every optimizer whose
  parameters overlap it is bound to it (``on_bind(model, optimizer)``), before the
  forward runs, so even the first step is observed;
* a global optimizer step post-hook enforces the step and wall-clock budget by
  raising :class:`StopTraining` (a ``BaseException``, so ``except Exception`` in
  user code does not swallow it).

:class:`GradLifecycleSensor` watches ``zero_grad()`` calls and gradient tensors
between optimizer steps to catch a missing ``optimizer.zero_grad()``.

:func:`user_frame` / :class:`SourceAnchors` record where in the user's code the
optimizer was built, the step taken and the model called, so runtime findings
point at a file and line like the static analyzer's do.
"""

from __future__ import annotations

import builtins
import linecache
import os
import sys
import sysconfig
import time
import traceback
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

Location = Tuple[str, int, str]  # (file, line, source text)


class StopTraining(BaseException):
    """Raised from an optimizer hook when the monitoring budget is exhausted (or a critical alert asks to stop)."""


# ------------------------------------------------------------- source anchors
def _internal_prefixes() -> Tuple[str, ...]:
    paths = sysconfig.get_paths()
    out = {os.path.dirname(torch.__file__), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        if paths.get(key):
            out.add(paths[key])
    return tuple(os.path.normcase(os.path.realpath(p)) + os.sep for p in out)


_INTERNAL: Optional[Tuple[str, ...]] = None


def _is_internal(filename: str) -> bool:
    global _INTERNAL
    if _INTERNAL is None:
        _INTERNAL = _internal_prefixes()
    if filename.startswith("<"):
        return True
    return os.path.normcase(os.path.realpath(filename)).startswith(_INTERNAL)


def user_frame(depth: int = 1) -> Optional[Location]:
    """The innermost caller frame that belongs to user code (not torch, this package, stdlib or site-packages)."""
    f = sys._getframe(depth)
    while f is not None:
        fn = f.f_code.co_filename
        if not _is_internal(fn):
            return (fn, f.f_lineno, linecache.getline(fn, f.f_lineno).strip())
        f = f.f_back
    return None


@dataclass
class SourceAnchors:
    """First-seen user-code locations of the training loop's key calls."""

    optimizer: Optional[Location] = None  # where the optimizer was constructed (LR, param groups)
    step: Optional[Location] = None  # first optimizer.step() call
    model_call: Optional[Location] = None  # first training forward of the model
    eval_call: Optional[Location] = None  # first forward under torch.no_grad()
    fallback: Optional[Location] = None  # e.g. where agent.watch() was called

    def get(self, name: str) -> Optional[Location]:
        order = {"optimizer": ("optimizer", "step", "model_call"), "step": ("step", "model_call", "optimizer"),
                 "model_call": ("model_call", "step"), "eval_call": ("eval_call", "model_call"),
                 "layer": ("model_call", "step")}.get(name, ("step", "model_call"))
        for n in order:
            loc = getattr(self, n)
            if loc is not None:
                return loc
        return self.fallback


def layer_location(root: nn.Module, name: str) -> Optional[Location]:
    """Best-effort location of the line that defines layer ``name`` (``"fc1[Linear]"`` or ``"fc1.weight"``).

    Walks the module path and, for each user-defined (non-torch) parent class, looks
    for ``self.<child> = ...`` in its source; the deepest match wins."""
    import inspect
    import re

    root = getattr(root, "_orig_mod", root)  # torch.compile wrapper
    path = name.split("[", 1)[0]
    parts = [p for p in path.split(".") if p]
    if not parts:
        return None
    if path not in dict(root.named_modules()) and len(parts) > 1:
        parts = parts[:-1]  # a parameter name: "fc1.weight" -> module "fc1"
    best: Optional[Location] = None
    module: Optional[nn.Module] = root
    for child in parts:
        if module is None:
            break
        # the class's own __init__ code object knows its file and line, even when the class was defined
        # in a script run as __main__ that is no longer in sys.modules
        init = type(module).__dict__.get("__init__")
        code = getattr(init, "__code__", None)
        if code is not None and not _is_internal(code.co_filename):
            lines = linecache.getlines(code.co_filename)
            if lines:
                try:
                    block = inspect.getblock(lines[code.co_firstlineno - 1:])
                except Exception:
                    block = lines[code.co_firstlineno - 1:code.co_firstlineno + 200]
                pat = re.compile(rf"self\.{re.escape(child)}\s*(:[^=]*)?=")
                for i, line in enumerate(block):
                    if pat.search(line):
                        best = (code.co_filename, code.co_firstlineno + i, line.strip())
                        break
        module = getattr(module, child, None)
    return best


# --------------------------------------------------------- gradient lifecycle
class GradLifecycleSensor:
    """Detects optimizer steps whose gradients were never cleared since the previous step.

    Between two steps of an optimizer, gradients count as cleared when
    ``zero_grad()`` ran on that optimizer or on any module, or when the gradient
    tensors were replaced (``set_to_none``, ``p.grad = None``). Otherwise backward()
    added the new gradient to the old one: the classic missing ``zero_grad()``.
    Clearing gradients in place by hand (``p.grad.zero_()`` without ``zero_grad()``)
    looks the same and is a known false positive (documented)."""

    def __init__(self, patience: int = 3) -> None:
        self.patience = patience
        self.zero_grad_calls = 0
        self._patches: List[Tuple[Any, str, Any, Any]] = []
        self._state: Dict[int, Dict[str, Any]] = {}  # id(optimizer) -> bookkeeping
        self._active = False

    def install(self, optimizers: Sequence[torch.optim.Optimizer] = ()) -> None:
        if not self._patches:
            self._patch(torch.optim.Optimizer, "zero_grad")
            self._patch(nn.Module, "zero_grad")
        for opt in optimizers:
            self.watch(opt)

    def watch(self, opt: torch.optim.Optimizer) -> None:
        cls = type(opt)
        if "zero_grad" in cls.__dict__ and not getattr(cls.__dict__["zero_grad"], "_sba_wrapped", False):
            self._patch(cls, "zero_grad")  # an optimizer that overrides zero_grad()

    def uninstall(self) -> None:
        self._active = False  # a wrapper someone else patched over stays in place, but inert
        for owner, name, orig, wrapper in reversed(self._patches):
            if owner.__dict__.get(name) is wrapper:
                setattr(owner, name, orig)
        self._patches.clear()

    def _patch(self, owner: type, name: str) -> None:
        orig = owner.__dict__[name]
        sensor = self
        self._active = True

        def zero_grad(obj, *a, **kw):  # type: ignore[no-untyped-def]
            if sensor._active:
                sensor.zero_grad_calls += 1
            return orig(obj, *a, **kw)

        zero_grad._sba_wrapped = True  # type: ignore[attr-defined]
        self._patches.append((owner, name, orig, zero_grad))
        setattr(owner, name, zero_grad)

    @staticmethod
    def _grads(opt: torch.optim.Optimizer) -> List[Tuple[int, torch.Tensor]]:
        return [(id(p), p.grad) for g in opt.param_groups for p in g["params"] if p.grad is not None]

    def before_step(self, opt: torch.optim.Optimizer) -> Optional[int]:
        """Call from the step pre-hook; returns the streak of uncleared steps once it reaches ``patience``."""
        st = self._state.setdefault(id(opt), {"refs": None, "calls": self.zero_grad_calls, "streak": 0})
        grads = self._grads(opt)
        refs = st["refs"]
        if refs is not None and grads:
            cleared = self.zero_grad_calls > st["calls"]
            if not cleared:
                # backward() accumulated into the very same gradient tensors (written to since the last step)
                same = 0
                for pid, g in grads:
                    ref = refs.get(pid)
                    if ref is not None and ref[0]() is g and g._version > ref[1]:
                        same += 1
                cleared = same < 0.9 * len(grads)
            st["streak"] = 0 if cleared else st["streak"] + 1
        return st["streak"] if st["streak"] >= self.patience else None

    def after_step(self, opt: torch.optim.Optimizer) -> None:
        st = self._state.setdefault(id(opt), {"refs": None, "calls": self.zero_grad_calls, "streak": 0})
        st["refs"] = {pid: (weakref.ref(g), g._version) for pid, g in self._grads(opt)}
        st["calls"] = self.zero_grad_calls


# ------------------------------------------------------------ auto-binding
@dataclass
class BudgetStatus:
    steps: int = 0
    stopped_early: bool = False
    stop_reason: str = ""
    elapsed: float = 0.0


class AutoInstrumenter:
    """Global hooks that find (model, optimizer) pairs in an unmodified script and bind them.

    ``on_bind(model, optimizer, optimizer_location)`` is called once per optimizer,
    from inside the first forward of the root model that owns its parameters."""

    def __init__(self, on_bind: Callable[[nn.Module, torch.optim.Optimizer, Optional[Location]], None], *,
                 max_steps: Optional[int] = None, max_seconds: Optional[float] = None,
                 should_stop: Optional[Callable[[], Optional[str]]] = None) -> None:
        self.on_bind = on_bind
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.should_stop = should_stop
        self.optimizers: Dict[int, torch.optim.Optimizer] = {}
        self.opt_locations: Dict[int, Optional[Location]] = {}
        self.bound: Dict[int, int] = {}  # id(optimizer) -> id(model)
        self.roots: Dict[int, nn.Module] = {}
        self.steps: Dict[int, int] = {}
        self.status = BudgetStatus()
        self._depth = 0
        self._handles: List[Any] = []
        self._orig_init: Any = None
        self._init_wrapper: Any = None
        self._t0 = time.time()
        self._binding = False

    # lifecycle
    def install(self) -> "AutoInstrumenter":
        warnings.filterwarnings("ignore", message=r"Using `torch.compile\(module\)` when there are global hooks")
        self._handles.append(register_module_forward_pre_hook(self._fwd_pre))
        self._handles.append(register_module_forward_hook(self._fwd_post))
        self._handles.append(register_optimizer_step_pre_hook(self._opt_pre))
        self._handles.append(register_optimizer_step_post_hook(self._opt_post))
        orig = torch.optim.Optimizer.__dict__["__init__"]
        inst = self

        def __init__(opt, *a, **kw):  # type: ignore[no-untyped-def]
            orig(opt, *a, **kw)
            if inst._orig_init is not None and id(opt) not in inst.optimizers:
                inst.optimizers[id(opt)] = opt
                inst.opt_locations[id(opt)] = user_frame(2)

        self._orig_init, self._init_wrapper = orig, __init__
        torch.optim.Optimizer.__init__ = __init__  # type: ignore[method-assign]
        self._t0 = time.time()
        return self

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._orig_init is not None:
            if torch.optim.Optimizer.__dict__.get("__init__") is self._init_wrapper:
                torch.optim.Optimizer.__init__ = self._orig_init  # type: ignore[method-assign]
            self._orig_init = None
        self.status.elapsed = time.time() - self._t0

    # forward: find roots and bind
    def _fwd_pre(self, module: nn.Module, args: Any) -> None:
        if isinstance(module, nn.modules.loss._Loss):
            return
        self._depth += 1
        if self._depth == 1 and not self._binding:
            self._maybe_bind(module)

    def _fwd_post(self, module: nn.Module, args: Any, output: Any) -> None:
        if isinstance(module, nn.modules.loss._Loss):
            return
        self._depth = max(0, self._depth - 1)

    def _maybe_bind(self, module: nn.Module) -> None:
        unbound = [o for k, o in self.optimizers.items() if k not in self.bound]
        if not unbound:
            return
        params = {id(p) for p in module.parameters()}
        if not params:
            return
        for opt in unbound:
            ids = {id(p) for g in opt.param_groups for p in g["params"]}
            if ids & params:
                self.bound[id(opt)] = id(module)
                self.roots[id(module)] = module
                self._binding = True
                try:
                    self.on_bind(module, opt, self.opt_locations.get(id(opt)))
                finally:
                    self._binding = False

    # steps: budget
    def _opt_pre(self, opt: Any, args: Any, kwargs: Any) -> None:
        self._depth = 0  # a forward that raised never ran its post hook

    def _opt_post(self, opt: Any, args: Any, kwargs: Any) -> None:
        self.steps[id(opt)] = self.steps.get(id(opt), 0) + 1
        self.status.steps = max(self.steps.values())
        reason = None
        if self.max_steps is not None and self.status.steps >= self.max_steps:
            reason = f"reached the step budget (max_steps={self.max_steps})"
        elif self.max_seconds is not None and time.time() - self._t0 > self.max_seconds:
            reason = f"reached the time budget (max_seconds={self.max_seconds:g})"
        elif self.should_stop is not None:
            reason = self.should_stop()
        if reason:
            self.status.stopped_early = True
            self.status.stop_reason = reason
            raise StopTraining(reason)


# --------------------------------------------------------------- run a script
@dataclass
class ScriptResult:
    status: str = "ok"  # "ok" | "error" | "stopped"
    error: Optional[str] = None
    duration: float = 0.0
    exit_code: Optional[int] = None


def exec_script(path: str, argv: Sequence[str] = (), cwd: Optional[str] = None,
                display_path: Optional[str] = None) -> ScriptResult:
    """Run ``path`` as ``__main__`` in this process (like ``python path argv...``) and restore argv/path/cwd.

    ``display_path`` makes a patched copy behave as if it were the original file, so
    its relative paths and sibling imports still work."""
    import types

    source_file = os.path.abspath(path)
    script = os.path.abspath(display_path) if display_path else source_file
    old_argv, old_path, old_cwd = list(sys.argv), list(sys.path), os.getcwd()
    main_mod = sys.modules.get("__main__")
    res = ScriptResult()
    t0 = time.time()
    try:
        with open(source_file, encoding="utf-8") as f:
            source = f.read()
        if cwd:
            os.chdir(cwd)
        sys.argv = [script] + list(argv)
        sys.path.insert(0, os.path.dirname(script))
        # a real __main__ module, so pickling and spawned DataLoader workers can import the script
        mod = types.ModuleType("__main__")
        mod.__dict__.update({"__file__": script, "__builtins__": builtins, "__spec__": None, "__package__": None})
        sys.modules["__main__"] = mod
        code = compile(source, script, "exec")
        exec(code, mod.__dict__)
    except StopTraining:
        res.status = "stopped"
    except SystemExit as exc:
        res.exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if exc.code not in (None, 0):
            res.status, res.error = "error", f"SystemExit({exc.code})"
    except BaseException as exc:  # noqa: BLE001 - every failure of the user script is reported, not raised
        res.status = "error"
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        res.error = "".join(tb[-4:]).strip()
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
        os.chdir(old_cwd)
        if main_mod is not None:
            sys.modules["__main__"] = main_mod
        res.duration = time.time() - t0
    return res
