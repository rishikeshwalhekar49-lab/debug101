"""The classifier's input: a fixed vector of numbers computed from one traced run.

Every feature is scale-free (a ratio, a fraction or a log), so a model trained on
small generated scripts can say something about other scripts too. A value that
the run could not measure is 0, and a matching `has_*` feature says whether it
was measured.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

FEATURES: List[str] = [
    "loss_log_ratio",  # log(final / initial loss)
    "loss_log_slope",  # slope of log(loss) over the run, steps scaled to 0..1
    "loss_up_fraction",  # share of steps where the loss went up
    "loss_nonfinite",  # 1 if any loss was NaN or infinite
    "log_initial_loss",
    "log_grad_norm",  # mean gradient norm (log10)
    "grad_growth",  # log(mean of the last 5 gradient norms / mean of the first 5)
    "grad_cv",  # spread of the gradient norm (std / mean)
    "no_grad_fraction",  # mean share of parameter tensors that had no gradient
    "log_update_ratio",  # mean |step| / |weights| (log10)
    "frozen_fraction",  # share of the optimizer's tensors that never moved
    "log_lr",
    "zero_grad_per_step",  # zero_grad() calls per optimizer step, capped at 3
    "unit_outputs",  # share of model outputs inside [0, 1]
    "rowsum_one_outputs",  # share of model outputs whose rows sum to 1 (softmax)
    "ce_on_probabilities",  # share of cross_entropy calls given probabilities instead of logits
    "relu_zero_fraction",
    "has_relu",
    "log_input_scale",  # mean |input| (log10)
    "log_input_std",
    "no_grad_forwards",  # share of forward passes under no_grad (validation)
    "no_grad_training_mode",  # share of those that ran with the model still in training mode
    "bias_mean",  # mean bias value, squashed to -1..1
    "has_updates",
]


def _log10(value: Optional[float], floor: float = 1e-12) -> float:
    return math.log10(max(float(value), floor)) if value is not None and math.isfinite(value) else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _slope(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = [i / (n - 1) for i in range(n)]
    mx, my = _mean(xs), _mean(values)
    var = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / var if var else 0.0


def vectorize(report: Dict[str, Any]) -> List[float]:
    """Feature vector for one `script_runner --trace` report (its `losses` plus its `trace`)."""
    trace = report.get("trace") or {}
    losses = [float(v) for v in report.get("losses") or []]
    finite = [v for v in losses if math.isfinite(v)]
    positive = [v for v in finite if v > 0]
    head, tail = _mean(positive[:3]), _mean(positive[-3:])
    grads = [g for g in trace.get("grad_norms") or [] if math.isfinite(g)]
    first, last = _mean(grads[:5]), _mean(grads[-5:])
    grad_mean = _mean(grads)
    grad_std = math.sqrt(_mean([(g - grad_mean) ** 2 for g in grads])) if grads else 0.0
    updates = [u for u in trace.get("update_ratios") or [] if math.isfinite(u)]
    steps = max(1, len(trace.get("grad_norms") or []))  # every step, including ones whose gradient was NaN
    measured = trace.get("measured_outputs") or 0
    forwards = trace.get("forwards") or 0
    no_grad = trace.get("no_grad_forwards") or 0
    relu = trace.get("relu_zero_fraction")
    bias = trace.get("bias_mean")
    values = {
        "loss_log_ratio": max(-10.0, min(10.0, math.log(tail / head))) if head and tail else 0.0,
        "loss_log_slope": max(-20.0, min(20.0, _slope([math.log(v) for v in positive]))),
        "loss_up_fraction": _mean([1.0 if b > a else 0.0 for a, b in zip(finite, finite[1:])]),
        "loss_nonfinite": 1.0 if len(finite) < len(losses) else 0.0,
        "log_initial_loss": _log10(head) if head else 0.0,
        "log_grad_norm": _log10(grad_mean) if grads else 0.0,
        "grad_growth": max(-10.0, min(10.0, math.log((last + 1e-12) / (first + 1e-12)))) if grads else 0.0,
        "grad_cv": min(10.0, grad_std / grad_mean) if grad_mean else 0.0,
        "no_grad_fraction": _mean(trace.get("no_grad_fraction") or []),
        "log_update_ratio": _log10(_mean(updates)) if updates else 0.0,
        "frozen_fraction": float(trace.get("frozen_fraction") or 0.0),
        "log_lr": _log10(trace.get("lr")) if trace.get("lr") else 0.0,
        "zero_grad_per_step": min(3.0, (trace.get("zero_grad_calls") or 0) / steps),
        "unit_outputs": (trace.get("unit_outputs") or 0) / measured if measured else 0.0,
        "rowsum_one_outputs": (trace.get("rowsum_one_outputs") or 0) / measured if measured else 0.0,
        "ce_on_probabilities": ((trace.get("ce_probability_inputs") or 0) / trace["ce_calls"]
                                if trace.get("ce_calls") else 0.0),
        "relu_zero_fraction": float(relu) if relu is not None else 0.0,
        "has_relu": 1.0 if relu is not None else 0.0,
        "log_input_scale": _log10(trace.get("input_abs_mean")) if trace.get("input_abs_mean") else 0.0,
        "log_input_std": _log10(trace.get("input_std")) if trace.get("input_std") else 0.0,
        "no_grad_forwards": no_grad / forwards if forwards else 0.0,
        "no_grad_training_mode": (trace.get("no_grad_training_forwards") or 0) / no_grad if no_grad else 0.0,
        "bias_mean": math.tanh(float(bias)) if bias is not None else 0.0,
        "has_updates": 1.0 if updates else 0.0,
    }
    # a diverged run can leave NaN weights (so a NaN mean bias): `loss_nonfinite` already says it diverged
    return [float(values[name]) if math.isfinite(values[name]) else 0.0 for name in FEATURES]
