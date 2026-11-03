"""A live monitoring session: the Runtime Monitor Agent's observe -> check -> alert loop.

One :class:`MonitorSession` watches one or more *channels* (a model and its
optimizer, e.g. two for a GAN). For each channel it

* attaches :class:`~silent_bug_agent.runtime.hooks.RuntimeHooks` and subscribes
  to its snapshot stream (the telemetry events);
* optionally attaches an :class:`~silent_bug_agent.runtime.profiling.OverheadGovernor`;
* records source anchors (where the optimizer was built, the step taken, the
  model called) and watches the gradient lifecycle (missing ``zero_grad()``).

A consumer thread takes each snapshot off the session's bounded inbox, runs the
:class:`~silent_bug_agent.runtime.anomaly.InvariantEvaluator` (Module 0 suite +
runtime checks + trends), and feeds the signals to the
:class:`~silent_bug_agent.runtime.alerts.AlertDispatcher`, which calls the
early-alert callbacks. Training never waits for the agent: the inbox drops the
oldest snapshot when the consumer falls behind (counted in the summary).

``close()`` flushes and detaches everything and :meth:`report` returns the
findings as a :class:`~silent_bug_agent.findings.Report`.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..findings import Finding, Report
from ..invariants.base import Severity, TrainingState
from .alerts import Alert, AlertDispatcher
from .anomaly import InvariantEvaluator
from .hooks import HookConfig, RuntimeHooks
from .instrument import GradLifecycleSensor, Location, SourceAnchors, StopTraining, layer_location, user_frame
from .profiling import OverheadGovernor
from .trends import Signal, TrendThresholds

# HookConfig defaults for the agent: a shorter stream (the agent drains it at once) and a capped loss
# history, so a snapshot never carries 10,000 floats.
AGENT_HOOK_DEFAULTS: Dict[str, Any] = {"stream_capacity": 256, "max_loss_history": 2048}


@dataclass
class SessionOptions:
    overhead_budget: Optional[float] = 0.05  # None: monitor every step
    min_severity: Severity = Severity.WARNING
    confirm_after: int = 2
    stop_on_critical: bool = False
    grad_lifecycle: bool = True
    inbox_capacity: int = 1024
    hook_overrides: Dict[str, Any] = field(default_factory=dict)
    trends: Optional[TrendThresholds] = None


@dataclass
class _Channel:
    name: str
    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    hooks: RuntimeHooks
    evaluator: InvariantEvaluator
    governor: Optional[OverheadGovernor]
    anchors: SourceAnchors
    handles: List[Any] = field(default_factory=list)
    snapshots: int = 0
    eval_snapshots: int = 0
    first_step_seen: bool = False


class MonitorSession:
    def __init__(self, options: Optional[SessionOptions] = None, *, file: Optional[str] = None,
                 on_alert: Optional[Callable[[Finding, str], None]] = None) -> None:
        self.options = options or SessionOptions()
        self.file = file
        self.dispatcher = AlertDispatcher(confirm_after=self.options.confirm_after,
                                          min_severity=self.options.min_severity)
        self.channels: Dict[str, _Channel] = {}
        self.lifecycle = GradLifecycleSensor() if self.options.grad_lifecycle else None
        self.stop_reason: Optional[str] = None
        self.optimizer_registry: Optional[Callable[[], Sequence[torch.optim.Optimizer]]] = None
        self.dropped = 0
        self.processed = 0
        self.errors: List[str] = []
        self._inbox: Deque[Tuple[str, str, Any]] = deque()
        self._cv = threading.Condition()
        self._closing = False
        self._thread: Optional[threading.Thread] = None
        self._started = time.time()
        self._closed = False
        if on_alert is not None:
            self.dispatcher.subscribe(lambda a, kind: on_alert(self._finding(a), kind))

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "MonitorSession":
        if self._thread is None:
            if self.lifecycle is not None:
                self.lifecycle.install()
            self._thread = threading.Thread(target=self._run, name="RuntimeMonitorAgent", daemon=True)
            self._thread.start()
        return self

    def __enter__(self) -> "MonitorSession":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return exc_type is StopTraining  # a requested stop ends the with-block quietly

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for ch in self.channels.values():
            if ch.governor is not None:
                ch.governor.detach()
            for h in ch.handles:
                h.remove()
            ch.hooks.detach()  # flushes: the last snapshots reach the inbox
        if self.lifecycle is not None:
            self.lifecycle.uninstall()
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join()

    @property
    def elapsed(self) -> float:
        return time.time() - self._started

    # ------------------------------------------------------------- channels
    def add(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, *,
            optimizer_location: Optional[Location] = None, fallback: Optional[Location] = None) -> _Channel:
        """Start monitoring ``model`` (trained by ``optimizer``)."""
        name = type(getattr(model, "_orig_mod", model)).__name__
        base, i = name, 2
        while name in self.channels:
            name, i = f"{base}{i}", i + 1
        cfg = HookConfig(**{**AGENT_HOOK_DEFAULTS, **self.options.hook_overrides})
        hooks = RuntimeHooks(model, optimizer, cfg)
        ch = _Channel(name, model, optimizer, hooks,
                      InvariantEvaluator(trends=self.options.trends), None,
                      SourceAnchors(optimizer=optimizer_location, fallback=fallback))
        hooks.stream.subscribe(lambda state, c=name: self._put("state", c, state))
        hooks.attach()
        if optimizer is not None and self.options.overhead_budget is not None:
            ch.governor = OverheadGovernor(hooks, self.options.overhead_budget).attach(optimizer)
        self._install_channel_hooks(ch)
        self.channels[name] = ch
        self._startup_checks(ch)
        return ch

    def _install_channel_hooks(self, ch: _Channel) -> None:
        anchors = ch.anchors

        def model_pre(module: nn.Module, args: Any) -> None:
            if torch.is_grad_enabled():
                if anchors.model_call is None:
                    anchors.model_call = user_frame()
            elif anchors.eval_call is None:
                anchors.eval_call = user_frame()

        ch.handles.append(ch.model.register_forward_pre_hook(model_pre))
        opt = ch.optimizer
        if opt is None:
            return
        if self.lifecycle is not None:
            self.lifecycle.watch(opt)

        def step_pre(o: Any, args: Any, kwargs: Any) -> None:
            if anchors.step is None:
                anchors.step = user_frame()
            if not ch.first_step_seen:
                ch.first_step_seen = True
                self._coverage_check(ch)
            if self.lifecycle is not None:
                streak = self.lifecycle.before_step(o)
                if streak is not None:
                    self._put("signals", ch.name, [Signal(
                        "runtime:missing_zero_grad", "missing_zero_grad", Severity.CRITICAL, 0.85,
                        f"gradients were not cleared before the last {streak} optimizer steps: no zero_grad() call, "
                        "and backward() accumulated into the same gradient tensors, so every update applies the "
                        "sum of all previous gradients",
                        ch.hooks.step + 1, value=float(streak), anchor="step")])

        def step_post(o: Any, args: Any, kwargs: Any) -> None:
            if self.lifecycle is not None:
                self.lifecycle.after_step(o)
            if self.options.stop_on_critical and self.stop_reason:
                raise StopTraining(self.stop_reason)

        ch.handles.append(opt.register_step_pre_hook(step_pre))
        ch.handles.append(opt.register_step_post_hook(step_post))

    def _startup_checks(self, ch: _Channel) -> None:
        params = list(ch.model.parameters())
        if torch.cuda.is_available() and params and all(p.device.type == "cpu" for p in params):
            self._put("signals", ch.name, [Signal(
                "runtime:device_placement", "silent_device_fallback", Severity.WARNING, 0.5,
                f"{ch.name} trains on the CPU although a CUDA GPU is available", 0, anchor="model_call")])

    def _coverage_check(self, ch: _Channel) -> None:
        """Trainable parameters that no optimizer updates (package101 OptimizerCoversTrainableParams)."""
        opts: List[torch.optim.Optimizer] = [c.optimizer for c in self.channels.values() if c.optimizer is not None]
        if ch.optimizer is not None and all(o is not ch.optimizer for o in opts):
            opts.append(ch.optimizer)
        if self.optimizer_registry is not None:
            opts += [o for o in self.optimizer_registry() if all(o is not x for x in opts)]
        covered = {id(p) for o in opts for g in o.param_groups for p in g["params"]}
        missing = [n for n, p in ch.model.named_parameters() if p.requires_grad and id(p) not in covered]
        if missing:
            names = [n.replace("_orig_mod.", "") for n in missing]
            self._put("signals", ch.name, [Signal(
                "runtime:optimizer_coverage", "frozen_layer_no_grad", Severity.WARNING, 0.9,
                f"{len(names)} trainable parameter tensor(s) of {ch.name} are not in any optimizer, so they never "
                f"update: {', '.join(names[:5])}{' ...' if len(names) > 5 else ''}",
                1, value=float(len(names)), offenders={n: 0.0 for n in names[:10]}, anchor="optimizer")])

    # ------------------------------------------------------- consumer thread
    def _put(self, kind: str, channel: str, item: Any) -> None:
        with self._cv:
            if len(self._inbox) >= self.options.inbox_capacity:
                self._inbox.popleft()
                self.dropped += 1
            self._inbox.append((kind, channel, item))
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._inbox and not self._closing:
                    self._cv.wait()
                if not self._inbox and self._closing:
                    return
                kind, channel, item = self._inbox.popleft()
            try:
                self._process(kind, channel, item)
            except Exception as exc:  # the agent must never take training down
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def _process(self, kind: str, channel: str, item: Any) -> None:
        ch = self.channels.get(channel)
        if ch is None:
            return
        if kind == "signals":
            fired = self.dispatcher.ingest(channel, item, lambda s: {"step": s.step, "phase": s.phase})
        else:
            state: TrainingState = item
            self.processed += 1
            if state.is_eval_phase:
                ch.eval_snapshots += 1
            else:
                ch.snapshots += 1
            signals = ch.evaluator.evaluate(state)
            self.dispatcher.tick(channel, "eval" if state.is_eval_phase else "train")
            fired = self.dispatcher.ingest(channel, signals, lambda s, st=state, c=ch: self._context(c, st, s))
        for a in fired:
            if a.severity == Severity.CRITICAL and self.stop_reason is None:
                self.stop_reason = f"critical runtime alert: {a.bug_id} at step {a.last_step}"

    @staticmethod
    def _context(ch: _Channel, state: TrainingState, s: Signal) -> Dict[str, Any]:
        """A small reproducible trace of the moment an alert first fired."""
        extra = state.extra or {}
        loss = extra.get("loss") or {}
        layers = [k for k in ([s.layer] if s.layer else []) + list(s.offenders) if k in state.activation_stats][:5]
        keep = ("mean", "std", "min", "max", "zero_frac", "dead_unit_frac", "saturated_frac", "nan_count", "inf_count",
                "grad_norm", "grad_abs_max")
        return {
            "step": state.step,
            "phase": "eval" if state.is_eval_phase else "train",
            "loss": loss.get("value"),
            "loss_ema": loss.get("ema"),
            "loss_tail": state.loss_history[-5:],
            "global_grad_norm": extra.get("global_grad_norm"),
            "update_ratio": extra.get("update_ratio"),
            "lrs": extra.get("lrs"),
            "every_n_steps": ch.hooks.config.every_n_steps,
            "layers": {k: {f: v for f, v in state.activation_stats[k].items() if f in keep} for k in layers},
            "trend": ch.evaluator.trends.summary(),
        }

    # --------------------------------------------------------------- output
    @property
    def alerts(self) -> List[Alert]:
        return self.dispatcher.confirmed()

    def _relpath(self, path: str) -> str:
        if self.file and os.path.realpath(path) == os.path.realpath(self.file):
            return self.file
        try:
            rel = os.path.relpath(path)
            return path if rel.startswith("..") else rel
        except ValueError:
            return path

    def _locate(self, alert: Alert) -> Tuple[str, int, str]:
        ch = self.channels.get(alert.channel)
        loc: Optional[Location] = None
        if ch is not None:
            if alert.anchor == "layer" and alert.layer:
                loc = layer_location(ch.model, alert.layer)
            loc = loc or ch.anchors.get(alert.anchor)
        if loc is None:
            return (self.file or "<runtime>", 0, "")
        return (self._relpath(loc[0]), loc[1], loc[2])

    def _finding(self, alert: Alert) -> Finding:
        return alert.to_finding(self._locate, self.file or "<runtime>", show_channel=len(self.channels) > 1)

    def findings(self) -> List[Finding]:
        return self.dispatcher.findings(self._locate, self.file or "<runtime>", show_channel=len(self.channels) > 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "channels": {
                name: {
                    "model": type(getattr(ch.model, "_orig_mod", ch.model)).__name__,
                    "optimizer": type(ch.optimizer).__name__ if ch.optimizer is not None else None,
                    "steps": ch.hooks.step,
                    "snapshots": ch.snapshots,
                    "eval_snapshots": ch.eval_snapshots,
                    "layers_hooked": len(ch.hooks.layers),
                    "every_n_steps": ch.hooks.config.every_n_steps,
                    "governor": ch.governor.stats.to_dict() if ch.governor is not None else None,
                    "stream_dropped": ch.hooks.stream.dropped,
                }
                for name, ch in self.channels.items()
            },
            "alerts": len(self.dispatcher.confirmed()),
            "alert_events": [list(e) for e in self.dispatcher.events],
            "suppressed_signals": self.dispatcher.suppressed,
            "inbox_dropped": self.dropped,
            "callback_errors": self.dispatcher.callback_errors,
            "agent_errors": self.errors[:10],
            "stop_reason": self.stop_reason,
            "elapsed_s": round(self.elapsed, 3),
        }

    def diagnostics(self) -> List[str]:
        out = []
        for name, c in self.summary()["channels"].items():
            gov = c["governor"]
            rate = f"every {c['every_n_steps']} step(s)"
            if gov and gov.get("estimated_overhead") is not None:
                rate += f", estimated monitoring overhead {gov['estimated_overhead']:.1%}"
            out.append(f"runtime: monitored {name} ({c['optimizer'] or 'no optimizer'}) for {c['steps']} step(s): "
                       f"{c['snapshots']} training + {c['eval_snapshots']} validation snapshot(s), "
                       f"{c['layers_hooked']} layer(s) hooked, {rate}")
        for name, ch in self.channels.items():
            for error in ch.hooks.errors[:2]:
                out.append(f"runtime: monitoring of {name} switched itself off after an internal error, "
                           f"and training continued: {error}")
        if self.dispatcher.suppressed:
            out.append(f"runtime: {self.dispatcher.suppressed} transient signal(s) did not persist and were not reported")
        if self.dropped:
            out.append(f"runtime: the agent fell behind and skipped {self.dropped} snapshot(s)")
        for e in self.errors[:3]:
            out.append(f"runtime: internal error in the monitor: {e}")
        return out

    def report(self) -> Report:
        return Report(findings=self.findings(), files_scanned=[self.file] if self.file else [],
                      diagnostics=self.diagnostics())
