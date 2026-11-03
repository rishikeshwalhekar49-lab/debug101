"""Rolling-window trend analysis over TrainingState snapshots (Runtime Monitor Agent tool).

The Module 0 invariants answer "is this snapshot healthy?" with fixed
thresholds and their own patience counters. Some silent bugs are visible
earlier as a *trend* across snapshots, before any threshold is crossed:

* gradient norms growing exponentially (exploding gradients, a few steps before
  ``gradient_norm_bounds`` fires or the loss turns NaN);
* the loss doubling over a short window (divergence, long before the 90-step
  ``loss_trend`` window is full);
* ReLU units dying: a layer whose ``dead_unit_frac`` stays high, even though the
  layer's overall ``zero_frac`` is below the 90% ``activation_health`` threshold;
* the relative update ``||dW||/||W||`` staying far too large or far too small.

Pure Python, no torch: it runs in the agent's consumer thread on the plain
values of each snapshot. Thresholds follow package101
(``trainmedic/detect/invariants.py``: ``UpdateRatioSane``, ``NoLossSpikes``).
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

from ..invariants.base import Severity, TrainingState


class RollingWindow:
    """The last ``size`` (step, value) pairs of one signal, with simple statistics."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._data: Deque[Tuple[int, float]] = deque(maxlen=size)

    def add(self, step: int, value: Optional[float]) -> None:
        if value is not None and math.isfinite(value):
            self._data.append((step, float(value)))

    def __len__(self) -> int:
        return len(self._data)

    @property
    def full(self) -> bool:
        return len(self._data) == self.size

    @property
    def values(self) -> List[float]:
        return [v for _, v in self._data]

    @property
    def last(self) -> Optional[float]:
        return self._data[-1][1] if self._data else None

    def median(self) -> Optional[float]:
        return statistics.median(self.values) if self._data else None

    def thirds(self) -> Tuple[float, float]:
        """Mean of the first and of the last third of the window."""
        head, tail = self.third_values()
        return statistics.fmean(head), statistics.fmean(tail)

    def third_values(self) -> Tuple[List[float], List[float]]:
        vals = self.values
        k = max(1, len(vals) // 3)
        return vals[:k], vals[-k:]

    def log_slope(self) -> Optional[float]:
        """Least-squares slope of log(value) per step (positive values only)."""
        pts = [(s, math.log(v)) for s, v in self._data if v > 0]
        if len(pts) < 3:
            return None
        n = len(pts)
        ms = sum(s for s, _ in pts) / n
        mv = sum(v for _, v in pts) / n
        den = sum((s - ms) ** 2 for s, _ in pts)
        if den == 0:
            return None
        return sum((s - ms) * (v - mv) for s, v in pts) / den

    def reset(self) -> None:
        self._data.clear()


@dataclass
class Signal:
    """One anomaly observed in one snapshot, before aggregation into an alert."""

    rule: str  # "invariant:<id>", "runtime:<check>" or "trend:<check>"
    bug_id: str  # Module 0 taxonomy id
    severity: Severity
    confidence: float  # the check's base confidence that this is a real bug
    message: str
    step: int
    value: Optional[float] = None  # headline measurement, used for the peak
    offenders: Dict[str, float] = field(default_factory=dict)
    anchor: str = "step"  # where in the user's code to point: optimizer | step | model_call | eval_call | layer
    layer: Optional[str] = None  # layer label, for anchor == "layer"
    phase: str = "train"
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class TrendThresholds:
    window: int = 8  # monitored snapshots per window
    grad_growth: float = 100.0  # global grad norm grew this much across the window ...
    grad_growth_last: float = 10.0  # ... and the last value is this many times the window's first third
    loss_window: int = 20
    loss_divergence: float = 2.0  # tail-third mean / head-third mean
    loss_divergence_critical: float = 10.0
    dead_units: float = 0.5  # dead_unit_frac at or above this ...
    dead_patience: int = 5  # ... for this many consecutive snapshots
    update_window: int = 10
    update_ratio_max: float = 1.0  # a single step changing the weights by more than their norm
    update_ratio_hi: float = 0.2  # rolling median above this: LR far too high
    update_ratio_lo: float = 1e-8  # rolling median below this (with non-zero grads and LR): LR far too low
    saturation: float = 0.9  # share of tanh/sigmoid outputs pinned at their limits ...
    saturation_patience: int = 5  # ... for this many consecutive snapshots


class TrendAnalyzer:
    """Feeds every training snapshot into rolling windows and emits trend signals."""

    def __init__(self, thresholds: Optional[TrendThresholds] = None) -> None:
        self.t = thresholds or TrendThresholds()
        self.grad = RollingWindow(self.t.window)
        self.loss = RollingWindow(self.t.loss_window)
        self.update = RollingWindow(self.t.update_window)
        self._dead_streak: Dict[str, int] = {}
        self._saturated_streak: Dict[str, int] = {}

    def reset(self) -> None:
        self.grad.reset()
        self.loss.reset()
        self.update.reset()
        self._dead_streak.clear()
        self._saturated_streak.clear()

    def summary(self) -> Dict[str, Optional[float]]:
        """Current trend statistics, attached to findings as context."""
        return {
            "grad_norm_median": self.grad.median(),
            "grad_norm_log_slope": self.grad.log_slope(),
            "loss_median": self.loss.median(),
            "update_ratio_median": self.update.median(),
        }

    def observe(self, state: TrainingState) -> List[Signal]:
        if state.is_eval_phase:
            return []
        extra = state.extra or {}
        step = state.step
        out: List[Signal] = []
        loss = (extra.get("loss") or {}).get("value")
        self.grad.add(step, extra.get("global_grad_norm"))
        self.loss.add(step, loss)
        self.update.add(step, extra.get("update_ratio"))
        out += self._grad_growth(step)
        out += self._loss_divergence(step)
        out += self._update_ratio(step, extra)
        out += self._dead_units(step, state.activation_stats.items())
        out += self._saturation(step, state.activation_stats.items())
        return out

    # ------------------------------------------------------------------ checks
    def _grad_growth(self, step: int) -> List[Signal]:
        if not self.grad.full:
            return []
        vals = self.grad.values
        head, _ = self.grad.thirds()
        slope = self.grad.log_slope()
        steps = self.grad._data[-1][0] - self.grad._data[0][0]
        if slope is None or head <= 0 or steps <= 0:
            return []
        growth = math.exp(slope * steps)
        if growth >= self.t.grad_growth and vals[-1] >= self.t.grad_growth_last * head:
            return [Signal(
                "trend:grad_norm_growth", "exploding_gradients", Severity.WARNING, 0.7,
                f"global gradient norm is growing exponentially: x{growth:.3g} over the last {steps} steps "
                f"(now {vals[-1]:.3g}, was {head:.3g})",
                step, value=vals[-1], anchor="step", details={"growth": growth, "window_steps": steps},
            )]
        return []

    def _loss_divergence(self, step: int) -> List[Signal]:
        if not self.loss.full:
            return []
        head_vals, tail_vals = self.loss.third_values()
        head, tail = statistics.median(head_vals), statistics.median(tail_vals)
        if head <= 0 or tail <= 0:
            return []
        ratio = tail / head
        # a shift, not noise: the median doubled and even the lowest recent loss is above the typical earlier one
        if ratio < self.t.loss_divergence or min(tail_vals) <= head:
            return []
        critical = ratio >= self.t.loss_divergence_critical
        return [Signal(
            "trend:loss_divergence", "learning_rate_misconfigured",
            Severity.CRITICAL if critical else Severity.WARNING, 0.75 if critical else 0.6,
            f"training loss is diverging: the median of the last {len(tail_vals)} monitored steps is "
            f"{ratio:.3g}x the median of the first {len(head_vals)} in a {len(self.loss)}-step window "
            f"({head:.4g} -> {tail:.4g})",
            step, value=ratio, anchor="optimizer", details={"head_median": head, "tail_median": tail},
        )]

    def _update_ratio(self, step: int, extra: Dict[str, object]) -> List[Signal]:
        out: List[Signal] = []
        current = extra.get("update_ratio")
        if isinstance(current, (int, float)) and math.isfinite(current) and current > self.t.update_ratio_max:
            out.append(Signal(
                "trend:update_ratio_spike", "learning_rate_misconfigured", Severity.WARNING, 0.65,
                f"one optimizer step changed the weights by {current:.3g}x their norm (limit {self.t.update_ratio_max:g}); "
                "the learning rate is far too high",
                step, value=current, anchor="optimizer",
            ))
        if not self.update.full:
            return out
        med = self.update.median()
        if med is None:
            return out
        if med > self.t.update_ratio_hi:
            out.append(Signal(
                "trend:update_ratio_high", "learning_rate_misconfigured", Severity.WARNING, 0.45,
                f"median relative update ||dW||/||W|| = {med:.3g} over {len(self.update)} steps exceeds "
                f"{self.t.update_ratio_hi:g}; the learning rate is likely too high",
                step, value=med, anchor="optimizer",
            ))
        elif 0 < med < self.t.update_ratio_lo:
            lrs = [lr for lr in (extra.get("lrs") or []) if isinstance(lr, (int, float)) and lr > 0]
            g = extra.get("global_grad_norm")
            if lrs and isinstance(g, (int, float)) and g > 0:
                out.append(Signal(
                    "trend:update_ratio_low", "learning_rate_misconfigured", Severity.WARNING, 0.45,
                    f"median relative update ||dW||/||W|| = {med:.3g} is below {self.t.update_ratio_lo:g} although "
                    "gradients are non-zero; the model barely changes (learning rate far too low)",
                    step, value=med, anchor="optimizer",
                ))
        return out

    def _saturation(self, step: int, layers: Iterable[Tuple[str, Dict[str, float]]]) -> List[Signal]:
        out: List[Signal] = []
        for label, st in layers:
            if not any(kind in label for kind in ("[Tanh]", "[Sigmoid]", "[Hardtanh]")):
                continue
            frac = st.get("saturated_frac")
            if frac is None or frac != frac:
                continue
            self._saturated_streak[label] = self._saturated_streak.get(label, 0) + 1 if frac >= self.t.saturation else 0
            if self._saturated_streak[label] >= self.t.saturation_patience:
                out.append(Signal(
                    "trend:saturation", "non_normalized_inputs", Severity.WARNING, 0.6,
                    f"{frac:.0%} of the outputs of {label} sit at the activation's limits, for "
                    f"{self._saturated_streak[label]} consecutive monitored steps, so its gradients vanish; the inputs "
                    "are probably not normalised",
                    step, value=frac, offenders={label: frac}, anchor="model_call", layer=label,
                ))
        return out

    def _dead_units(self, step: int, layers: Iterable[Tuple[str, Dict[str, float]]]) -> List[Signal]:
        out: List[Signal] = []
        for label, st in layers:
            frac = st.get("dead_unit_frac")
            if frac is None or frac != frac:
                continue
            if frac >= self.t.dead_units:
                self._dead_streak[label] = self._dead_streak.get(label, 0) + 1
            else:
                self._dead_streak[label] = 0
            if self._dead_streak[label] >= self.t.dead_patience:
                out.append(Signal(
                    "trend:dead_units", "dead_relu", Severity.WARNING, 0.7,
                    f"{frac:.0%} of the units of {label} output zero for every sample, for "
                    f"{self._dead_streak[label]} consecutive monitored steps (dying ReLU)",
                    step, value=frac, offenders={label: frac}, anchor="layer", layer=label,
                ))
        return out
