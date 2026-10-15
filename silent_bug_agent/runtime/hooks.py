"""Module 1: non-intrusive PyTorch hooks that feed the Runtime Monitor Agent.

``RuntimeHooks`` attaches to an existing model and optimizer -- the user's
model source is never modified -- and turns training into a stream of
:class:`~silent_bug_agent.invariants.base.TrainingState` snapshots, the same
framework-agnostic contract the Module 0 invariants already check.

Sensors (all removable with :meth:`RuntimeHooks.detach`):

* **forward hooks** on every leaf layer (plus ``nn.MultiheadAttention`` and RNN
  modules as units) -> ``activation_stats``: the contract keys ``mean``, ``std``,
  ``nan_count``, ``inf_count``, ``zero_frac`` plus ``min``, ``max``, ``abs_mean``,
  ``saturated_frac`` (tanh/sigmoid), ``norm``, ``numel``, ``calls``;
* **backward (tensor) hooks** on each layer's output -> gradient flowing back
  through the layer: ``grad_norm``, ``grad_abs_mean``, ``grad_abs_max``,
  ``grad_nan_count``, ``grad_inf_count`` inside the same ``activation_stats`` entry;
* **optimizer step pre/post hooks** -> ``grad_norms`` and ``weight_deltas`` per
  parameter (``||W_after - W_before||`` of that step), ``requires_grad``;
* a **backward() wrapper** -> ``loss_history`` plus loss dynamics (EMA, z-score,
  spike flag) in ``extra["loss"]``;
* a **root forward hook** -> automatic ``is_eval_phase=True`` snapshots when the
  model is run under ``torch.no_grad()`` (validation), with ``module_modes``.

Labels follow ``examples/torch_adapter.py``: layers and modules are keyed
``"<name>[<ClassName>]"`` so the mode invariant can match "batchnorm"/"dropout";
parameters are keyed by their ``named_parameters()`` name.

Low-overhead streaming (adapted from package101 ``trainmedic/detect/tracer.py``):

* statistics are computed on the tensor's own device (GPU or CPU) from *views*;
  activations are never copied or moved. Tensors above ``max_elems`` (default
  65,536) are sampled with a strided view, and one extra reduction over the
  whole tensor still catches any NaN/Inf outside the sample;
* hooks never synchronise with the GPU. At the end of a monitored step every
  statistic is packed into one small device vector, copied to the host once and
  asynchronously into a reused pinned buffer, and turned into a TrainingState
  when the copy has landed (at most ``max_pending`` steps later);
* the only extra device copy is a strided sample of each weight tensor, taken
  just before ``optimizer.step()`` into a buffer allocated once, so the weight
  delta can be measured;
* snapshots go to a :class:`SnapshotStream` (bounded, thread-safe, never blocks
  training) for the Runtime Monitor Agent to consume.

Validation across architectures: ``python -m silent_bug_agent.runtime.validate``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..invariants.base import Invariant, InvariantResult, TrainingState
from ..invariants.registry import run_all
from .numerics import EmaZScore

ACT_FIELDS = ("mean", "std", "min", "max", "abs_mean", "zero_frac", "saturated_frac", "norm", "nan_count", "inf_count")
GRAD_FIELDS = ("grad_norm", "grad_abs_mean", "grad_abs_max", "grad_nan_count", "grad_inf_count")
PARAM_FIELDS = ("grad_norm", "grad_abs_max", "weight_norm", "delta_norm")

_SKIP_TYPES: Tuple[type, ...] = (nn.Identity, nn.Flatten, nn.Unflatten, nn.modules.loss._Loss)
_UNIT_TYPES: Tuple[type, ...] = (nn.MultiheadAttention, nn.RNNBase)


def _compiling() -> bool:
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if fn is None:
        try:
            from torch._dynamo import is_compiling as fn  # torch 2.0
        except Exception:
            return False
    try:
        return bool(fn())
    except Exception:
        return False


def module_label(name: str, module: nn.Module) -> str:
    """``"<name>[<ClassName>]"`` -- the key convention of examples/torch_adapter.py."""
    return f"{name}[{type(module).__name__}]"


def _saturation_kind(module: nn.Module) -> str:
    if isinstance(module, (nn.Tanh, nn.Hardtanh)):
        return "tanh"
    if isinstance(module, nn.Sigmoid):
        return "sigmoid"
    return ""


def _first_float_tensor(obj: Any) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj if obj.is_floating_point() and obj.numel() > 0 else None
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        for o in obj:
            t = _first_float_tensor(o)
            if t is not None:
                return t
    return None


# =============================================================== statistics
class TensorStatsExtractor:
    """Tensor statistics computed on the tensor's own device.

    Every method returns a small device tensor and never calls ``.item()``,
    ``float()`` or ``.tolist()``, so a hook never forces a GPU synchronisation."""

    def __init__(self, max_elems: int = 1 << 16, histogram_bins: int = 0) -> None:
        self.max_elems = max_elems
        self.histogram_bins = histogram_bins

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """A flat strided *view* of at most ``max_elems`` elements, and its stride."""
        flat = x.detach()
        flat = flat.reshape(-1) if flat.is_contiguous() else flat.flatten()
        stride = max(1, math.ceil(flat.numel() / self.max_elems))
        return (flat[::stride] if stride > 1 else flat), stride

    @staticmethod
    def _f32(t: torch.Tensor) -> torch.Tensor:
        return t if t.dtype == torch.float32 else t.float()

    @staticmethod
    def _nan_inf(x: torch.Tensor, f: torch.Tensor, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nan = torch.isnan(f).sum().to(torch.float32) * stride
        inf = torch.isinf(f).sum().to(torch.float32) * stride
        if stride > 1:
            # One extra reduction over the whole tensor: NaN/Inf anywhere propagate into the sum, so a
            # bad value outside the sample still gives a count >= 1. (If NaN *and* Inf both occur only
            # outside the sample, the sum is NaN and only nan_count is raised.)
            total = x.detach().sum(dtype=torch.float32)
            one = torch.ones((), dtype=torch.float32, device=f.device)
            nan = torch.where(torch.isnan(total) & (nan == 0), one, nan)
            inf = torch.where(torch.isinf(total) & (inf == 0), one, inf)
        return nan, inf

    def activation(self, x: torch.Tensor, saturation: str = "") -> torch.Tensor:
        f, stride = self.sample(x)
        f = self._f32(f)
        finite = torch.isfinite(f)
        n = finite.sum().to(f.dtype)
        nf = n.clamp_min(1.0)
        zero = torch.zeros((), dtype=f.dtype, device=f.device)
        z = torch.where(finite, f, zero)
        s2 = (z * z).sum()
        mean = z.sum() / nf
        std = (s2 / nf - mean * mean).clamp_min(0).sqrt()
        inf_t = torch.full((), float("inf"), dtype=f.dtype, device=f.device)
        mn = torch.where(finite, f, inf_t).amin()
        mx = torch.where(finite, f, -inf_t).amax()
        abs_mean = z.abs().sum() / nf
        zero_frac = ((f == 0) & finite).sum().to(f.dtype) / nf
        if saturation == "tanh":
            sat = (z.abs() >= 0.99).sum().to(f.dtype) / nf
        elif saturation == "sigmoid":
            sat = (((z <= 0.01) | (z >= 0.99)) & finite).sum().to(f.dtype) / nf
        else:
            sat = zero
        nan, inf = self._nan_inf(x, f, stride)
        return torch.stack([mean, std, mn, mx, abs_mean, zero_frac, sat, s2.sqrt() * math.sqrt(stride), nan, inf])

    def gradient(self, g: torch.Tensor) -> torch.Tensor:
        f, stride = self.sample(g)
        f = self._f32(f)
        finite = torch.isfinite(f)
        n = finite.sum().to(f.dtype)
        z = torch.where(finite, f, torch.zeros((), dtype=f.dtype, device=f.device)).abs()
        norm = (z * z).sum().sqrt() * math.sqrt(stride)
        nan, inf = self._nan_inf(g, f, stride)
        return torch.stack([norm, z.sum() / n.clamp_min(1.0), z.amax(), nan, inf])

    def histogram(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised histogram over the tensor's own [min, max] (edges = the min/max stats)."""
        f, _ = self.sample(x)
        f = self._f32(f)
        f = torch.where(torch.isfinite(f), f, torch.zeros((), dtype=f.dtype, device=f.device))
        h = torch.histc(f, bins=self.histogram_bins, min=0, max=0)
        return h / h.sum().clamp_min(1.0)


# ================================================================ streaming
class SnapshotStream:
    """Bounded, thread-safe ring buffer of TrainingState snapshots.

    The producer (training) never blocks: when the consumer falls behind, the
    oldest snapshots are dropped and counted in ``dropped``."""

    def __init__(self, capacity: int = 1024) -> None:
        self._q: Deque[TrainingState] = deque(maxlen=capacity)
        self._cv = threading.Condition()
        self._subs: List[Callable[[TrainingState], None]] = []
        self._closed = False
        self.latest: Optional[TrainingState] = None
        self.published = 0
        self.dropped = 0
        self.subscriber_errors = 0

    def publish(self, state: TrainingState) -> None:
        with self._cv:
            if self._q.maxlen is not None and len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append(state)
            self.latest = state
            self.published += 1
            subs = list(self._subs)
            self._cv.notify_all()
        for cb in subs:
            try:
                cb(state)
            except Exception:  # a broken consumer must never break training
                self.subscriber_errors += 1

    def subscribe(self, callback: Callable[[TrainingState], None]) -> Callable[[], None]:
        """Call ``callback(state)`` for every new snapshot; returns an unsubscribe function."""
        with self._cv:
            self._subs.append(callback)

        def unsubscribe() -> None:
            with self._cv:
                if callback in self._subs:
                    self._subs.remove(callback)

        return unsubscribe

    def get(self, timeout: Optional[float] = None) -> Optional[TrainingState]:
        """Next unread snapshot (blocks up to ``timeout``); None on timeout or once closed and empty."""
        with self._cv:
            self._cv.wait_for(lambda: bool(self._q) or self._closed, timeout)
            return self._q.popleft() if self._q else None

    def drain(self) -> List[TrainingState]:
        with self._cv:
            out = list(self._q)
            self._q.clear()
            return out

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> Iterator[TrainingState]:
        while True:
            s = self.get()
            if s is None:
                return
            yield s

    def __len__(self) -> int:
        return len(self._q)


# =============================================================== collectors
class ActivationCollector:
    """Forward hooks (activations) and output-tensor hooks (gradients) on the selected layers."""

    def __init__(self, layers: List[Tuple[str, nn.Module]], extractor: TensorStatsExtractor, *,
                 track_grads: bool = True, capture_eval: bool = True) -> None:
        self.layers = layers
        self.labels = [module_label(n, m) for n, m in layers]
        self.saturation = [_saturation_kind(m) for _, m in layers]
        self.extractor = extractor
        self.track_grads = track_grads
        self.capture_eval = capture_eval
        self.active = True
        self.handles: List[Any] = []
        self.reset("train")
        self.reset("eval")

    def reset(self, phase: str) -> None:
        tables = {"act": {}, "grad": {}, "hist": {}, "calls": {}, "numel": {}}
        if phase == "train":
            self.train = tables
        else:
            self.eval = tables

    def attach(self) -> None:
        for i, (_name, module) in enumerate(self.layers):
            self.handles.append(module.register_forward_hook(partial(self._on_forward, i)))

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _on_forward(self, i: int, module: nn.Module, inputs: Any, output: Any) -> None:
        if _compiling():
            return
        training_pass = torch.is_grad_enabled()
        if training_pass and not self.active:
            return
        if not training_pass and not self.capture_eval:
            return
        t = _first_float_tensor(output)
        if t is None:
            return
        tab = self.train if training_pass else self.eval
        with torch.no_grad():
            tab["act"][i] = self.extractor.activation(t, self.saturation[i])
            if self.extractor.histogram_bins:
                tab["hist"][i] = self.extractor.histogram(t)
        tab["calls"][i] = tab["calls"].get(i, 0) + 1
        tab["numel"][i] = t.numel()
        if training_pass and self.track_grads and t.requires_grad:
            t.register_hook(partial(self._on_grad, i))

    def _on_grad(self, i: int, g: torch.Tensor) -> None:
        if self.active and g is not None:
            with torch.no_grad():
                self.train["grad"][i] = self.extractor.gradient(g)
        return None  # never modifies the gradient


class ParameterCollector:
    """Optimizer-step hooks: per-parameter gradient norms, weight norms and weight deltas.

    Tracks every named parameter of the model (not only those given to the
    optimizer), so a trainable layer left out of the optimizer shows a zero delta."""

    def __init__(self, named_params: List[Tuple[str, nn.Parameter]], max_elems: int = 1 << 20) -> None:
        self.names = [n for n, _ in named_params]
        self.params = [p for _, p in named_params]
        self.max_elems = max_elems
        self._shadow: List[Optional[torch.Tensor]] = [None] * len(self.params)
        self._strides = [1] * len(self.params)
        self._pre: Optional[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]] = None
        self.last_has_grad: List[bool] = [False] * len(self.params)
        self._captured = False
        self.device: Optional[torch.device] = self.params[0].device if self.params else None

    def _to_dev(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.device is None or t.device == self.device else t.to(self.device, non_blocking=True)

    def _sample(self, i: int, t: torch.Tensor) -> torch.Tensor:
        flat = t.detach()
        flat = flat.reshape(-1) if flat.is_contiguous() else flat.flatten()
        s = self._strides[i]
        return flat[::s] if s > 1 else flat

    @torch.no_grad()
    def norms(self) -> None:
        """Per-parameter gradient norm, gradient |max| (on the sample) and weight norm."""
        nan = torch.full((), float("nan"), device=self.device)
        grads = [p.grad for p in self.params if p.grad is not None]
        self.last_has_grad = [p.grad is not None for p in self.params]
        g_norms = iter(torch._foreach_norm(grads)) if grads else iter(())
        w_norms = torch._foreach_norm([p.detach() for p in self.params]) if self.params else []
        gnorm: List[torch.Tensor] = []
        gmax: List[torch.Tensor] = []
        wnorm: List[torch.Tensor] = []
        for i, p in enumerate(self.params):
            self._ensure_shadow(i, p)
            if p.grad is not None:
                gnorm.append(self._to_dev(next(g_norms).float()))  # exact (fused foreach norm)
                gmax.append(self._to_dev(self._sample(i, p.grad).abs().amax().float()))
            else:
                gnorm.append(nan)
                gmax.append(nan)
            wnorm.append(self._to_dev(w_norms[i].float()))
        self._pre = (gnorm, gmax, wnorm)

    def _ensure_shadow(self, i: int, p: torch.Tensor) -> None:
        if self._shadow[i] is None:
            self._strides[i] = max(1, math.ceil(p.numel() / self.max_elems))
            self._shadow[i] = torch.empty_like(self._sample(i, p))

    @torch.no_grad()
    def capture(self) -> None:
        """Remember the current weights (a strided sample, into buffers allocated once)."""
        for i, p in enumerate(self.params):
            self._ensure_shadow(i, p)
            self._shadow[i].copy_(self._sample(i, p))  # the only weight copy
        self._captured = True

    def before_step(self) -> None:
        self.norms()
        self.capture()

    @torch.no_grad()
    def after_step(self) -> torch.Tensor:
        """Rows of PARAM_FIELDS; the delta is measured against the last capture()."""
        if self._pre is None:
            self.norms()
        gnorm, gmax, wnorm = self._pre  # type: ignore[misc]
        if self._captured:
            deltas = []
            for i, p in enumerate(self.params):
                buf = self._shadow[i]
                buf.sub_(self._sample(i, p))  # in place: old - new
                deltas.append(buf)
            d_norms = torch._foreach_norm(deltas) if deltas else []
            dnorm = [self._to_dev(d.float()) * math.sqrt(self._strides[i]) for i, d in enumerate(d_norms)]
        else:
            nan = torch.full((), float("nan"), device=self.device)
            dnorm = [nan for _ in self.params]
        self._pre = None
        self._captured = False
        if not self.params:
            return torch.empty(0, len(PARAM_FIELDS))
        return torch.stack([torch.stack(gnorm), torch.stack(gmax), torch.stack(wnorm), torch.stack(dnorm)], dim=1)


class LossCollector:
    """Captures every scalar passed to ``backward()`` as a device tensor (no sync)."""

    def __init__(self) -> None:
        self.values: List[torch.Tensor] = []
        self._orig: Any = None
        self._in = False

    def install(self) -> None:
        if self._orig is not None:
            return
        self._orig = torch.Tensor.backward
        orig, coll = self._orig, self

        def backward(t, *a, **kw):  # type: ignore[no-untyped-def]
            if not coll._in and not _compiling() and t.numel() == 1 and t.is_floating_point():
                coll.values.append(t.detach().reshape(()).float())
            coll._in = True
            try:
                return orig(t, *a, **kw)
            finally:
                coll._in = False

        torch.Tensor.backward = backward  # type: ignore[assignment]

    def uninstall(self) -> None:
        if self._orig is not None:
            torch.Tensor.backward = self._orig  # type: ignore[assignment]
            self._orig = None

    def observe(self, loss: Union[torch.Tensor, float]) -> None:
        if isinstance(loss, torch.Tensor):
            self.values.append(loss.detach().reshape(()).float())
        else:
            self.values.append(torch.tensor(float(loss)))


# ============================================================= orchestrator
@dataclass
class HookConfig:
    every_n_steps: int = 1
    layers: Union[str, Callable[[str, nn.Module], bool]] = "auto"  # "auto" or a predicate(name, module)
    max_layers: int = 256
    max_elems: int = 1 << 16  # per activation/gradient; larger tensors are sampled (strided view)
    max_param_elems: int = 1 << 20  # per weight tensor, for the weight-delta sample
    histogram_bins: int = 0  # > 0 adds normalised histograms to extra["histograms"]
    track_activation_grads: bool = True
    capture_backward_loss: bool = True
    eval_snapshots: str = "first"  # "first" per validation phase | "every" no-grad forward | "off"
    max_loss_history: Optional[int] = 10_000  # loss values kept in loss_history (None = all)
    stream_capacity: int = 1024
    max_pending: int = 4  # device-to-host copies allowed in flight before waiting
    loss_z: float = 6.0
    loss_warmup: int = 10


def select_layers(model: nn.Module, rule: Union[str, Callable[[str, nn.Module], bool]] = "auto",
                  max_layers: int = 256) -> List[Tuple[str, nn.Module]]:
    """Leaf modules (plus attention / RNN modules as single units), in registration order."""
    out: List[Tuple[str, nn.Module]] = []
    units: List[str] = []
    for name, m in model.named_modules():
        name = name.replace("_orig_mod.", "").replace("_orig_mod", "")
        if not name:
            continue
        if callable(rule):
            keep = bool(rule(name, m))
        else:
            if isinstance(m, _SKIP_TYPES) or any(name.startswith(u + ".") for u in units):
                continue
            unit = isinstance(m, _UNIT_TYPES)
            keep = unit or next(m.children(), None) is None
            if unit:
                units.append(name)
        if keep:
            out.append((name, m))
        if len(out) >= max_layers:
            break
    return out


class RuntimeHooks:
    """Attach non-intrusive hooks to ``model`` (and ``optimizer``) and stream TrainingState snapshots.

    Usage::

        hooks = RuntimeHooks(model, optimizer, invariants=get_default_registry()).attach()
        for x, y in loader:                     # the training loop is unchanged
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()                    # a TrainingState is produced here
        hooks.detach()
        print(hooks.failures)                   # [(step, InvariantResult), ...]
        states = hooks.stream.drain()           # or hooks.stream.get() from an agent thread

    Without an optimizer (manual weight updates) call ``hooks.end_step()`` after each step.
    Validation passes (forward under ``torch.no_grad()``) produce ``is_eval_phase=True`` snapshots."""

    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                 config: Optional[HookConfig] = None, *, invariants: Optional[List[Invariant]] = None,
                 stream: Optional[SnapshotStream] = None,
                 on_failure: Optional[Callable[[TrainingState, InvariantResult], None]] = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or HookConfig()
        self.stream = stream or SnapshotStream(self.config.stream_capacity)
        self.invariants = invariants
        self.on_failure = on_failure
        self.failures: List[Tuple[int, InvariantResult]] = []
        self.extractor = TensorStatsExtractor(self.config.max_elems, self.config.histogram_bins)
        self.layers = select_layers(model, self.config.layers, self.config.max_layers)
        self.acts = ActivationCollector(self.layers, self.extractor, track_grads=self.config.track_activation_grads,
                                        capture_eval=self.config.eval_snapshots != "off")
        named = [(n.replace("_orig_mod.", ""), p) for n, p in model.named_parameters()]
        seen = {id(p) for _, p in named}
        if optimizer is not None:
            extra = [p for g in optimizer.param_groups for p in g["params"] if id(p) not in seen]
            named += [(f"optimizer_param{i}", p) for i, p in enumerate(extra)]
        self.param_coll = ParameterCollector(named, self.config.max_param_elems)
        self.losses = LossCollector()
        self.loss_history: List[float] = []
        self._loss_det = EmaZScore(alpha=0.05, warmup=self.config.loss_warmup, z=self.config.loss_z)
        self._handles: List[Any] = []
        self._steps_done = 0
        self._attached = False
        self._eval_taken = False
        self._pending: Deque[Tuple[Any, torch.Tensor, Dict[str, Any]]] = deque()
        self._free: List[torch.Tensor] = []
        self._nan_rows: Dict[Tuple[int, str], torch.Tensor] = {}

    # ---------------------------------------------------------------- lifecycle
    @property
    def step(self) -> int:
        """Number of optimizer steps completed so far."""
        return self._steps_done

    def _monitoring(self) -> bool:
        return self._steps_done % max(1, self.config.every_n_steps) == 0

    def attach(self) -> "RuntimeHooks":
        if self._attached:
            return self
        self.acts.attach()
        self.acts.active = self._monitoring()
        self._handles.append(self.model.register_forward_hook(self._on_root_forward))
        if self.config.capture_backward_loss:
            self.losses.install()
        if self.optimizer is not None:
            self._handles.append(self.optimizer.register_step_pre_hook(self._on_opt_pre))
            self._handles.append(self.optimizer.register_step_post_hook(self._on_opt_post))
        elif self._monitoring():
            self.param_coll.capture()  # manual updates: deltas are measured from here
        self._attached = True
        return self

    def detach(self) -> None:
        if not self._attached:
            return
        self.flush()
        self.acts.detach()
        self.losses.uninstall()
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._attached = False

    def __enter__(self) -> "RuntimeHooks":
        return self.attach()

    def __exit__(self, *exc: Any) -> None:
        self.detach()

    def observe_loss(self, loss: Union[torch.Tensor, float]) -> None:
        """Report a loss explicitly (if backward() is not used or capture_backward_loss is off)."""
        self.losses.observe(loss)

    # -------------------------------------------------------------- training
    def _on_opt_pre(self, opt: Any, args: Any, kwargs: Any) -> None:
        if self._monitoring():
            self.param_coll.before_step()

    def _on_opt_post(self, opt: Any, args: Any, kwargs: Any) -> None:
        self._complete_step(self.param_coll.after_step() if self._monitoring() else None)

    def end_step(self) -> None:
        """Mark the end of a training step when no optimizer hook is available (manual updates).

        Weight deltas are measured against the weights at the end of the previous step."""
        rows = None
        if self._monitoring():
            self.param_coll.norms()
            rows = self.param_coll.after_step()
        self._complete_step(rows)
        if self.optimizer is None and self._monitoring():
            self.param_coll.capture()

    def _complete_step(self, param_rows: Optional[torch.Tensor]) -> None:
        self._steps_done += 1
        if param_rows is not None:
            self._finalize("train", param_rows)
        self.acts.reset("train")
        self.losses.values.clear()
        self.acts.active = self._monitoring()
        self._eval_taken = False
        self._poll(block=False)

    # ---------------------------------------------------------------- eval
    def _on_root_forward(self, module: nn.Module, inputs: Any, output: Any) -> None:
        if _compiling() or torch.is_grad_enabled() or self.config.eval_snapshots == "off":
            return
        if self.config.eval_snapshots == "first" and self._eval_taken:
            self.acts.reset("eval")
            return
        self._eval_taken = True
        self._finalize("eval", None)
        self.acts.reset("eval")

    # -------------------------------------------------------------- packing
    def _device(self) -> torch.device:
        if self.param_coll.device is not None:
            return self.param_coll.device
        for tab in (self.acts.train, self.acts.eval):
            for t in tab["act"].values():
                return t.device
        return torch.device("cpu")

    def _nan_row(self, n: int, dev: torch.device) -> torch.Tensor:
        key = (n, str(dev))
        if key not in self._nan_rows:
            self._nan_rows[key] = torch.full((n,), float("nan"), device=dev)
        return self._nan_rows[key]

    def _rows(self, table: Dict[int, torch.Tensor], width: int, dev: torch.device) -> torch.Tensor:
        L = len(self.layers)
        if L == 0 or width == 0:
            return torch.empty(0, device=dev)
        rows = [table[i] if i in table else self._nan_row(width, dev) for i in range(L)]
        rows = [r if r.device == dev else r.to(dev, non_blocking=True) for r in rows]
        return torch.stack(rows).reshape(-1)

    def _finalize(self, phase: str, param_rows: Optional[torch.Tensor]) -> None:
        dev = self._device()
        tab = self.acts.train if phase == "train" else self.acts.eval
        bins = self.extractor.histogram_bins
        with torch.no_grad():
            losses = [v if v.device == dev else v.to(dev, non_blocking=True) for v in self.losses.values]
            loss = torch.stack(losses).mean().reshape(1) if (losses and phase == "train") else self._nan_row(1, dev)
            parts = [
                loss,
                self._rows(tab["act"], len(ACT_FIELDS), dev),
                self._rows(tab["grad"], len(GRAD_FIELDS), dev),
                self._rows(tab["hist"], bins, dev) if bins else torch.empty(0, device=dev),
                param_rows.to(dev).reshape(-1).float() if param_rows is not None else torch.empty(0, device=dev),
            ]
            packed = torch.cat(parts)
        host = self._host_buffer(packed)
        host.copy_(packed, non_blocking=packed.is_cuda)
        event = None
        if packed.is_cuda:
            event = torch.cuda.Event()
            event.record()
        meta = {
            "phase": phase,
            "step": self._steps_done,
            "wall_time": time.time(),
            "n_backward": len(self.losses.values) if phase == "train" else 0,
            "calls": dict(tab["calls"]),
            "numel": dict(tab["numel"]),
            "has_act": set(tab["act"]),
            "has_grad": set(tab["grad"]),
            "has_hist": set(tab["hist"]),
            "has_params": param_rows is not None,
            "has_param_grad": list(self.param_coll.last_has_grad),
            "lrs": [float(g.get("lr", float("nan"))) for g in self.optimizer.param_groups] if self.optimizer else [],
            "module_modes": {module_label(n.replace("_orig_mod.", "").replace("_orig_mod", ""), m):
                             ("train" if m.training else "eval") for n, m in self.model.named_modules()},
            "requires_grad": {n: bool(p.requires_grad) for n, p in zip(self.param_coll.names, self.param_coll.params)},
        }
        self._pending.append((event, host, meta))
        while len(self._pending) > self.config.max_pending:
            self._publish(*self._pending.popleft())

    def _host_buffer(self, packed: torch.Tensor) -> torch.Tensor:
        n = packed.numel()
        for i, b in enumerate(self._free):
            if b.numel() >= n:
                return self._free.pop(i)[:n]
        pin = packed.is_cuda and torch.cuda.is_available()
        return torch.empty(max(n, 64), dtype=torch.float32, pin_memory=pin)[:n]

    def _poll(self, block: bool) -> None:
        while self._pending:
            event, host, meta = self._pending[0]
            if not block and event is not None and not event.query():
                return
            self._pending.popleft()
            self._publish(event, host, meta)

    def flush(self) -> None:
        """Publish every pending snapshot (waits for in-flight device-to-host copies)."""
        self._poll(block=True)

    @property
    def latest(self) -> Optional[TrainingState]:
        self._poll(block=False)
        return self.stream.latest

    def _publish(self, event: Any, host: torch.Tensor, meta: Dict[str, Any]) -> None:
        if event is not None:
            event.synchronize()
        vals = host.tolist()
        self._free.append(host)
        state = self._materialize(vals, meta)
        self.stream.publish(state)
        if self.invariants:
            for r in run_all(self.invariants, state):
                if not r.passed:
                    self.failures.append((state.step, r))
                    if self.on_failure is not None:
                        self.on_failure(state, r)

    # ------------------------------------------------------- materialisation
    def _materialize(self, vals: List[float], meta: Dict[str, Any]) -> TrainingState:
        L, bins = len(self.layers), self.extractor.histogram_bins
        pos = 0
        loss = vals[pos]
        pos += 1
        act = vals[pos: pos + L * len(ACT_FIELDS)]
        pos += L * len(ACT_FIELDS)
        grd = vals[pos: pos + L * len(GRAD_FIELDS)]
        pos += L * len(GRAD_FIELDS)
        hist = vals[pos: pos + (L * bins if bins else 0)]
        pos += L * bins if bins else 0
        prm = vals[pos:]

        activation_stats: Dict[str, Dict[str, float]] = {}
        histograms: Dict[str, Dict[str, Any]] = {}
        for i, label in enumerate(self.acts.labels):
            if i not in meta["has_act"] and i not in meta["has_grad"]:
                continue
            st: Dict[str, float] = {"calls": float(meta["calls"].get(i, 0)), "numel": float(meta["numel"].get(i, 0))}
            if i in meta["has_act"]:
                st.update(zip(ACT_FIELDS, act[i * len(ACT_FIELDS): (i + 1) * len(ACT_FIELDS)]))
            if i in meta["has_grad"]:
                st.update(zip(GRAD_FIELDS, grd[i * len(GRAD_FIELDS): (i + 1) * len(GRAD_FIELDS)]))
            activation_stats[label] = st
            if bins and i in meta["has_hist"]:
                histograms[label] = {"min": st.get("min"), "max": st.get("max"), "density": hist[i * bins: (i + 1) * bins]}

        grad_norms: Dict[str, float] = {}
        weight_deltas: Dict[str, float] = {}
        weight_norms: Dict[str, float] = {}
        grad_abs_max: Dict[str, float] = {}
        if meta["has_params"]:
            for j, name in enumerate(self.param_coll.names):
                gn, gm, wn, dn = prm[j * 4: j * 4 + 4]
                if meta["has_param_grad"][j]:
                    grad_norms[name] = gn  # NaN/Inf kept as-is: the gradient invariant flags them
                    grad_abs_max[name] = gm
                weight_deltas[name] = dn
                weight_norms[name] = wn

        extra: Dict[str, Any] = {"source": "silent_bug_agent.runtime.hooks", "wall_time": meta["wall_time"],
                                 "lrs": meta["lrs"]}
        if meta["phase"] == "train":
            loss_v: Optional[float] = loss if meta["n_backward"] > 0 else None
            if loss_v is not None:
                self.loss_history.append(loss_v)
                cap = self.config.max_loss_history
                if cap is not None and len(self.loss_history) > cap:
                    del self.loss_history[: len(self.loss_history) - cap]
            z, spike = (self._loss_det.update(loss_v) if loss_v is not None else (None, False))
            extra["loss"] = {
                "value": loss_v,
                "ema": self._loss_det.mean if self._loss_det.n else None,
                "z": z if z is not None and math.isfinite(z) else None,
                "spike": bool(spike),
                "n_backward": meta["n_backward"],
            }
            finite_g = [v for v in grad_norms.values() if math.isfinite(v)]
            extra["global_grad_norm"] = math.sqrt(sum(v * v for v in finite_g)) if finite_g else None
            wsq = sum(v * v for v in weight_norms.values() if math.isfinite(v))
            dsq = sum(v * v for v in weight_deltas.values() if math.isfinite(v))
            extra["global_weight_norm"] = math.sqrt(wsq)
            extra["global_delta_norm"] = math.sqrt(dsq)
            extra["update_ratio"] = math.sqrt(dsq) / math.sqrt(wsq) if wsq > 0 else None
            extra["weight_norms"] = weight_norms
            extra["grad_abs_max"] = grad_abs_max
        if histograms:
            extra["histograms"] = histograms

        return TrainingState(
            step=meta["step"],
            loss_history=list(self.loss_history),
            grad_norms=grad_norms,
            weight_deltas=weight_deltas,
            requires_grad=meta["requires_grad"],
            activation_stats=activation_stats,
            module_modes=meta["module_modes"],
            is_eval_phase=meta["phase"] == "eval",
            extra=extra,
        )
