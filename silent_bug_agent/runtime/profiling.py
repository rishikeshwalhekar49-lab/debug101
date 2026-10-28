"""Telemetry overhead profiling (Runtime Monitor Agent tools).

* :class:`OverheadGovernor` -- keeps the *amortised* cost of monitoring under a
  budget (default 5% of step time) during a live run. It times every optimizer
  step, calibrates with a few monitored steps, probes a few unmonitored ones, and
  sets ``every_n_steps`` to the smallest N whose estimated overhead fits the
  budget. Heavy models keep N = 1 (every step); tiny ones are sampled.
* :class:`MemoryLeakDetector` -- samples device memory (CUDA / MPS, exact
  allocated bytes) and host RSS, and checks for growth across steps and for
  memory left behind after ``detach()``.
* :func:`profile_steps` / :func:`profile_diff` -- PyTorch profiler integration:
  which operators monitoring adds, and how much time they take.
* :func:`time_steps` / :func:`synchronize` -- step timing with device sync, used
  by ``examples/benchmark_hooks.py`` (the synthetic benchmark runner).

Benchmark methodology and results: ``docs/overhead_benchmark.md``.
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch

from .hooks import RuntimeHooks


# ------------------------------------------------------------------ timing
def synchronize(device: Any = None) -> None:
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
    elif dev is not None and dev.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def time_steps(step_fn: Callable[[], Any], n: int, device: Any = None) -> List[float]:
    """Wall-clock seconds of ``n`` calls to ``step_fn``, each bracketed by a device sync."""
    out = []
    for _ in range(n):
        synchronize(device)
        t0 = time.perf_counter()
        step_fn()
        synchronize(device)
        out.append(time.perf_counter() - t0)
    return out


# ---------------------------------------------------------------- governor
@dataclass
class GovernorStats:
    every_n_steps: int = 1
    estimated_overhead: Optional[float] = None  # amortised fraction of step time at the current N
    cost_per_monitored_step: Optional[float] = None  # seconds added by one monitored step
    baseline_step: Optional[float] = None  # seconds per unmonitored step (median of the last probe)
    probes: int = 0
    adjustments: List[Tuple[int, int]] = field(default_factory=list)  # (step, new N)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "every_n_steps": self.every_n_steps,
            "estimated_overhead": self.estimated_overhead,
            "cost_per_monitored_step_ms": None if self.cost_per_monitored_step is None else self.cost_per_monitored_step * 1e3,
            "baseline_step_ms": None if self.baseline_step is None else self.baseline_step * 1e3,
            "probes": self.probes,
            "adjustments": self.adjustments[-20:],
        }


class OverheadGovernor:
    """Adapts ``hooks.every_n_steps`` so monitoring costs at most ``budget`` of the step time.

    Phases: *calibrate* (every step monitored for ``calibrate_steps``), *probe*
    (monitoring paused for ``probe_steps``), then *run* at the chosen N, re-probing
    every ``reprobe_every`` steps. Step durations are measured between optimizer
    post-hooks, so the cost includes snapshot packing and publishing; with
    asynchronous (CUDA) copies part of it lands on the following step, which the
    estimator counts too:

        cost = (median monitored step - t_u) + max(0, median step after a monitored one - t_u)
        N    = ceil(cost / (margin * budget * t_u))      t_u = median unmonitored step (probe)

    ``margin`` (0.6) aims well below the budget, so noise in the step timings does
    not push the real overhead over it. Medians make it robust to the occasional slow step
    (validation, data loading)."""

    def __init__(self, hooks: RuntimeHooks, budget: float = 0.05, *, calibrate_steps: int = 8, probe_steps: int = 6,
                 reprobe_every: int = 200, max_every_n: int = 100, margin: float = 0.6) -> None:
        self.hooks = hooks
        self.budget = budget
        self.margin = margin
        self.calibrate_steps = calibrate_steps
        self.probe_steps = probe_steps
        self.reprobe_every = reprobe_every
        self.max_every_n = max_every_n
        self.stats = GovernorStats(every_n_steps=hooks.config.every_n_steps)
        self._phase = "calibrate"
        self._phase_steps = 0
        self._last_end: Optional[float] = None
        self._cur_monitored = False
        self._prev_monitored = False
        self._mon: Deque[float] = deque(maxlen=64)  # durations of monitored steps
        self._after: Deque[float] = deque(maxlen=64)  # durations of unmonitored steps right after a monitored one
        self._probe: List[float] = []
        self._handles: List[Any] = []

    def attach(self, optimizer: Optional[torch.optim.Optimizer]) -> "OverheadGovernor":
        if optimizer is not None:
            # registered after RuntimeHooks.attach(), so these run after the hooks' own step hooks
            self._handles.append(optimizer.register_step_pre_hook(lambda *_: self.before_step()))
            self._handles.append(optimizer.register_step_post_hook(lambda *_: self.after_step()))
        self.hooks.set_every_n_steps(1)
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self.hooks.paused:
            self.hooks.resume()

    def before_step(self) -> None:
        self._cur_monitored = self.hooks._monitoring()

    def after_step(self) -> None:
        now = time.perf_counter()
        last, self._last_end = self._last_end, now
        self._phase_steps += 1
        if last is None:
            self._prev_monitored = self._cur_monitored
            return
        dur = now - last
        if self._phase == "probe":
            if self._phase_steps > 1:  # the first probe step still carries deferred work
                self._probe.append(dur)
            if self._phase_steps >= self.probe_steps:
                self._finish_probe()
        else:
            warm = self._phase_steps > 2  # skip the first steps after a phase change (warm-up, transitions)
            if warm and self._cur_monitored:
                self._mon.append(dur)
            elif warm and self._prev_monitored:
                self._after.append(dur)
            limit = self.calibrate_steps if self._phase == "calibrate" else self.reprobe_every
            if self._phase_steps >= limit:
                self._start_probe()
        self._prev_monitored = self._cur_monitored

    def _start_probe(self) -> None:
        self._phase, self._phase_steps, self._probe = "probe", 0, []
        self.hooks.pause()

    def _finish_probe(self) -> None:
        self.stats.probes += 1
        self.hooks.resume()
        self._phase, self._phase_steps = "run", 0
        if not self._probe or not self._mon:
            return
        t_u = statistics.median(self._probe)
        if t_u <= 0:
            return
        cost = max(0.0, statistics.median(self._mon) - t_u)
        if self._after:
            cost += max(0.0, statistics.median(self._after) - t_u)
        n = max(1, min(self.max_every_n, math.ceil(cost / (self.margin * self.budget * t_u)) if cost > 0 else 1))
        self.stats.baseline_step, self.stats.cost_per_monitored_step = t_u, cost
        self.stats.estimated_overhead = cost / (n * t_u)
        if n != self.hooks.config.every_n_steps:
            self.stats.adjustments.append((self.hooks.step, n))
        self.hooks.set_every_n_steps(n)
        self.stats.every_n_steps = n
        self._mon.clear()
        self._after.clear()


# ------------------------------------------------------------ memory leaks
def device_allocated(device: Any) -> Optional[int]:
    """Exact bytes currently allocated by tensors on ``device`` (None on CPU)."""
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated(dev))
    if dev.type == "mps" and torch.backends.mps.is_available():
        return int(torch.mps.current_allocated_memory())
    return None


def host_rss() -> Optional[int]:
    try:
        import psutil  # optional
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


@dataclass
class LeakReport:
    device: str
    samples: int
    device_growth_bytes: Optional[int]  # allocated after the run minus after warm-up
    device_slope_bytes_per_step: Optional[float]
    host_growth_bytes: Optional[int]
    detach_residual_bytes: Optional[int] = None  # allocated after detach() minus before attach()
    tolerance_bytes: int = 0
    leak: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class MemoryLeakDetector:
    """Samples memory at step boundaries and decides whether monitoring leaks.

    Device memory is exact (``torch.cuda.memory_allocated`` / ``torch.mps.current_allocated_memory``),
    so a healthy monitored run shows *zero* growth once warmed up. Host RSS is
    reported for information only (allocator and OS noise make it inexact)."""

    def __init__(self, device: Any, *, warmup: int = 3, tolerance_bytes: int = 0) -> None:
        self.device = str(device)
        self.warmup = warmup
        self.tolerance_bytes = tolerance_bytes
        self.device_samples: List[int] = []
        self.host_samples: List[int] = []
        self._before_attach: Optional[int] = None
        self._after_detach: Optional[int] = None

    def sample(self) -> None:
        synchronize(self.device)
        d = device_allocated(self.device)
        if d is not None:
            self.device_samples.append(d)
        h = host_rss()
        if h is not None:
            self.host_samples.append(h)

    def mark_before_attach(self) -> None:
        gc.collect()
        synchronize(self.device)
        self._before_attach = device_allocated(self.device)

    def mark_after_detach(self) -> None:
        gc.collect()
        synchronize(self.device)
        self._after_detach = device_allocated(self.device)

    @staticmethod
    def _slope(ys: Sequence[int]) -> Optional[float]:
        n = len(ys)
        if n < 3:
            return None
        mx, my = (n - 1) / 2, sum(ys) / n
        den = sum((i - mx) ** 2 for i in range(n))
        return sum((i - mx) * (y - my) for i, y in enumerate(ys)) / den

    def report(self) -> LeakReport:
        dev = self.device_samples[self.warmup:]
        host = self.host_samples[self.warmup:]
        growth = (dev[-1] - dev[0]) if len(dev) >= 2 else None
        residual = (self._after_detach - self._before_attach) \
            if self._after_detach is not None and self._before_attach is not None else None
        leak = (growth is not None and growth > self.tolerance_bytes) or \
               (residual is not None and residual > self.tolerance_bytes)
        return LeakReport(self.device, len(self.device_samples), growth, self._slope(dev),
                          (host[-1] - host[0]) if len(host) >= 2 else None, residual, self.tolerance_bytes, leak)


# --------------------------------------------------------------- profiler
def profile_steps(step_fn: Callable[[], Any], steps: int = 3, device: Any = None) -> List[Dict[str, Any]]:
    """Run ``step_fn`` ``steps`` times under ``torch.profiler``; per-operator totals (microseconds)."""
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if device is not None and torch.device(device).type == "cuda" and torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    synchronize(device)
    with profile(activities=acts) as prof:
        for _ in range(steps):
            step_fn()
        synchronize(device)
    rows = []
    for e in prof.key_averages():
        dev_us = getattr(e, "self_device_time_total", None)
        if dev_us is None:
            dev_us = getattr(e, "self_cuda_time_total", 0.0)
        rows.append({"op": e.key, "count": int(e.count) // max(1, steps), "self_cpu_us": e.self_cpu_time_total / steps,
                     "self_device_us": float(dev_us or 0.0) / steps})
    return rows


def profile_diff(baseline: List[Dict[str, Any]], monitored: List[Dict[str, Any]], top: int = 10) -> List[Dict[str, Any]]:
    """Operators whose per-step self time grew most when monitoring is on (what the hooks add)."""
    base = {r["op"]: r for r in baseline}
    out = []
    for r in monitored:
        b = base.get(r["op"], {"count": 0, "self_cpu_us": 0.0, "self_device_us": 0.0})
        out.append({"op": r["op"], "added_calls": r["count"] - b["count"],
                    "added_cpu_us": r["self_cpu_us"] - b["self_cpu_us"],
                    "added_device_us": r["self_device_us"] - b["self_device_us"]})
    out.sort(key=lambda r: r["added_cpu_us"] + r["added_device_us"], reverse=True)
    return out[:top]
