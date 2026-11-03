"""Live invariant evaluation and runtime anomaly detection (Runtime Monitor Agent tool).

:class:`InvariantEvaluator` checks every :class:`TrainingState` snapshot with

1. the **Module 0 invariant suite** (``get_default_registry()``: gradient norm
   bounds, frozen weights, train/eval mode, activation health, loss trend), and
2. **runtime checks** ported from package101 (``trainmedic/detect/invariants.py``):
   non-finite loss (``FiniteLoss``) and loss spikes (``NoLossSpikes``, using the
   EMA z-score the hooks already compute), plus
3. the rolling-window :class:`~silent_bug_agent.runtime.trends.TrendAnalyzer`
   (exploding-gradient trend, loss divergence, dead units, update ratio).

Each failure becomes a :class:`~silent_bug_agent.runtime.trends.Signal` that
names a Module 0 taxonomy ``bug_id``, so the findings the agent builds from them
line up with the static and semantic detectors. Pure Python, no torch.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from ..invariants.base import Invariant, InvariantResult, Severity, TrainingState
from ..invariants.registry import get_default_registry, run_all
from .trends import Signal, TrendAnalyzer, TrendThresholds

# Other taxonomy entries that produce the same runtime symptom. The diagnosis
# agent ranks them; the runtime monitor only reports what it measured.
ALTERNATIVE_BUG_IDS: Dict[str, List[str]] = {
    "exploding_gradients": ["learning_rate_misconfigured", "mixed_precision_overflow", "non_normalized_inputs",
                            "missing_zero_grad"],
    "vanishing_gradients": ["dead_relu", "learning_rate_misconfigured", "frozen_layer_no_grad"],
    "frozen_layer_no_grad": ["learning_rate_misconfigured", "vanishing_gradients"],
    "dead_relu": ["learning_rate_misconfigured", "non_normalized_inputs"],
    "non_normalized_inputs": ["learning_rate_misconfigured", "vanishing_gradients"],
    "learning_rate_misconfigured": ["wrong_loss_reduction", "lr_scheduler_step_order_bug", "label_data_misalignment",
                                    "missing_zero_grad"],
    "train_eval_mode_mismatch": [],
    "missing_zero_grad": ["gradient_accumulation_miscount"],
    "silent_device_fallback": [],
}

# Base confidence of each Module 0 invariant when it fails (package101's base_confidence for the same checks).
_INVARIANT_CONFIDENCE = {
    "gradient_norm_bounds": 0.85,
    "frozen_weights": 0.85,
    "train_eval_mode_mismatch": 0.9,
    "activation_health": 0.8,
    "loss_trend": 0.55,
}


def _top(offenders: Dict[str, float], n: int = 10, key=abs) -> Dict[str, float]:
    def k(item):
        v = item[1]
        return float("inf") if v != v else key(v)

    return dict(sorted(offenders.items(), key=k, reverse=True)[:n])


def _bias_before_norm(param: str, state: TrainingState) -> bool:
    """A bias right before BatchNorm/LayerNorm is cancelled by the normalisation, so its gradient is ~0 by design."""
    if not param.endswith(".bias"):
        return False
    module = param[: -len(".bias")]
    layers = list(state.activation_stats)
    names = [label.split("[", 1)[0] for label in layers]
    if module not in names:
        return False
    following = layers[names.index(module) + 1: names.index(module) + 2]
    return any("Norm" in label for label in following)


def _fmt(v: float) -> str:
    return "nan" if v != v else f"{v:.3g}"


class InvariantEvaluator:
    """Stateful per run: one instance per monitored model (invariants keep rolling windows)."""

    def __init__(self, invariants: Optional[Sequence[Invariant]] = None,
                 trends: Optional[TrendThresholds] = None, *, loss_spikes: bool = True) -> None:
        self.invariants: List[Invariant] = list(invariants) if invariants is not None else get_default_registry()
        self.trends = TrendAnalyzer(trends)
        self.loss_spikes = loss_spikes
        self.snapshots = 0
        self.last_results: List[InvariantResult] = []

    def reset(self) -> None:
        for inv in self.invariants:
            inv.reset()
        self.trends.reset()
        self.snapshots = 0

    def evaluate(self, state: TrainingState) -> List[Signal]:
        self.snapshots += 1
        results = run_all(self.invariants, state)
        self.last_results = results
        out: List[Signal] = []
        for r in results:
            if not r.passed:
                out += self._from_invariant(r, state)
        out += self._runtime_checks(state)
        out += self.trends.observe(state)
        return out

    # ------------------------------------------------------- Module 0 suite
    def _from_invariant(self, r: InvariantResult, state: TrainingState) -> List[Signal]:
        step, phase = state.step, ("eval" if state.is_eval_phase else "train")
        conf = _INVARIANT_CONFIDENCE.get(r.invariant_id, 0.6)
        rule = f"invariant:{r.invariant_id}"
        d = r.details or {}
        if r.invariant_id == "gradient_norm_bounds":
            off = d.get("offenders", {})
            inv = next((i for i in self.invariants if i.id == "gradient_norm_bounds"), None)
            hi = getattr(inv, "max_norm", 1e3)
            big = {n: v for n, v in off.items() if v != v or v > hi}
            small = {n: v for n, v in off.items() if n not in big and not _bias_before_norm(n, state)}
            out = []
            if big:
                worst = next(iter(_top(big, 1)))
                out.append(Signal(rule, "exploding_gradients", Severity.CRITICAL, conf,
                                  f"{len(big)} parameter(s) with gradient norm above {hi:g} or non-finite "
                                  f"(worst: {worst} = {_fmt(big[worst])})",
                                  step, value=max((v for v in big.values() if v == v), default=float("inf")),
                                  offenders=_top(big), anchor="step", phase=phase))
            if small:
                worst = next(iter(_top(small, 1, key=lambda v: -abs(v))))
                out.append(Signal(rule, "vanishing_gradients", Severity.WARNING, 0.5,
                                  f"{len(small)} parameter(s) with gradient norm below "
                                  f"{getattr(inv, 'min_norm', 1e-7):g} (e.g. {worst} = {_fmt(small[worst])})",
                                  step, value=min(small.values()), offenders=_top(small, key=lambda v: -abs(v)),
                                  anchor="layer", layer=worst, phase=phase))
            return out
        if r.invariant_id == "frozen_weights":
            off = {k: float(v) for k, v in d.get("offenders", {}).items()}
            return [Signal(rule, "frozen_layer_no_grad", r.severity, conf,
                           f"{r.message}: {', '.join(list(off)[:5])}{' ...' if len(off) > 5 else ''}",
                           step, value=float(len(off)), offenders=_top(off), anchor="optimizer", phase=phase)]
        if r.invariant_id == "train_eval_mode_mismatch":
            off = d.get("offenders", {})
            return [Signal(rule, "train_eval_mode_mismatch", r.severity, conf,
                           f"{r.message}: {', '.join(list(off)[:5])}{' ...' if len(off) > 5 else ''}",
                           step, value=float(len(off)), offenders={k: 1.0 for k in list(off)[:10]},
                           anchor="eval_call" if state.is_eval_phase else "model_call", phase=phase,
                           details={"modes": dict(list(off.items())[:10])})]
        if r.invariant_id == "activation_health":
            out = []
            bad = d.get("nan_inf_offenders", {})
            if bad:
                order = list(state.activation_stats)
                first = min(bad, key=lambda k: order.index(k) if k in order else len(order))
                origin = " (the first monitored layer, so the input batch itself is likely non-finite)" \
                    if order and first == order[0] else ""
                out.append(Signal(rule, "exploding_gradients", Severity.CRITICAL, conf,
                                  f"{len(bad)} layer(s) producing NaN/Inf activations; the first in forward order is "
                                  f"{first}{origin}",
                                  step, value=float(len(bad)),
                                  offenders={k: float(v.get("nan_count", 0) + v.get("inf_count", 0)) for k, v in bad.items()},
                                  anchor="step", layer=first, phase=phase, details={"first_nonfinite_layer": first}))
            dead = d.get("dead_offenders", {})
            if dead:
                worst = next(iter(_top(dead, 1)))
                out.append(Signal(rule, "dead_relu", Severity.WARNING, 0.75,
                                  f"{len(dead)} layer(s) with >= 90% zero activations (worst: {worst}, "
                                  f"{dead[worst]:.0%} zeros)",
                                  step, value=dead[worst], offenders=_top(dead), anchor="layer", layer=worst,
                                  phase=phase))
            return out
        if r.invariant_id == "loss_trend":
            if r.severity != Severity.CRITICAL and self._converged(state.loss_history):
                return []  # a plateau after the loss already fell a lot is convergence, not a bug
            return [Signal(rule, "learning_rate_misconfigured", r.severity, conf if r.severity != Severity.CRITICAL else 0.7,
                           r.message, step, value=d.get("mean_tail"), anchor="optimizer", phase=phase,
                           details={k: d[k] for k in ("mean_head", "mean_tail") if k in d})]
        return [Signal(rule, "unknown", r.severity, 0.5, r.message, step, phase=phase)]

    @staticmethod
    def _converged(history: Sequence[float], k: int = 5, drop: float = 0.5) -> bool:
        """The loss is at most ``drop`` x its initial level (mean of the first and last ``k`` values)."""
        vals = [v for v in history if v == v and math.isfinite(v)]
        if len(vals) < 2 * k:
            return False
        first, last = sum(vals[:k]) / k, sum(vals[-k:]) / k
        return first > 0 and last <= drop * first

    # -------------------------------------------------- package101 checks
    def _runtime_checks(self, state: TrainingState) -> List[Signal]:
        if state.is_eval_phase:
            return []
        loss = (state.extra or {}).get("loss") or {}
        value = loss.get("value")
        if value is None:
            return []
        if not math.isfinite(value):
            return [Signal("runtime:nonfinite_loss", "exploding_gradients", Severity.CRITICAL, 0.95,
                           f"training loss became non-finite ({value}) at step {state.step}",
                           state.step, value=value, anchor="step")]
        if self.loss_spikes and loss.get("spike"):
            z, ema = loss.get("z"), loss.get("ema")
            z_txt = f"{z:.3g} standard deviations" if isinstance(z, (int, float)) else "far"
            ema_txt = f" {ema:.4g}" if isinstance(ema, (int, float)) else ""
            return [Signal("runtime:loss_spike", "exploding_gradients", Severity.WARNING, 0.4,
                           f"loss spike at step {state.step}: {value:.4g} is {z_txt} above its moving average{ema_txt}",
                           state.step, value=z if isinstance(z, (int, float)) else None, anchor="step")]
        return []
