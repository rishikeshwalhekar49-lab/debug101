"""The classifier's input: a fixed vector of numbers computed from one traced run.

Every feature is scale-free (a ratio, a fraction or a log), so a model trained on
small generated scripts can say something about other scripts too. A value that
the run could not measure is 0, and a matching `has_*` feature says whether it
was measured.

Version 4 adds the backward-pass measurements of `trace.py` (gradients left over
from before an update, the step size actually taken, first-to-last layer gradients,
saturation, the optimizer's kind), a few more points of the loss curve, and the
facts read from the code (`code.py`). Scripts with no optimizer get their gradient
and step statistics from the backward passes instead.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .code import CODE_FEATURES, code_features

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
    "eval_missing_where_it_matters",  # share of those in training mode, counted only if the model has dropout/BN
    "has_mode_layers",  # the model has dropout or batch norm, so train() and eval() behave differently
    "outside_optimizer",  # share of the model's trainable parameters that no optimizer updates
    "bias_mean",  # mean bias value, squashed to -1..1
    "has_updates",
    # from the loss-function calls; they separate the bugs whose loss "barely falls"
    "sum_reduction",  # share of training loss calls with reduction="sum"
    "start_vs_chance",  # log10(first training losses / the loss of guessing: ln(classes))
    "end_vs_chance",  # log10(last training losses / that same chance level)
    "has_chance",  # 1 for classification losses, where a chance level exists
    "val_slope",  # slope of log(validation loss) over the run
    "val_vs_train_end",  # log10(last validation loss / last training loss)
    "val_end_vs_chance",  # log10(last validation loss / chance level)
    "has_validation_loss",
]
V3_FEATURES = list(FEATURES)
TRACE_V4: List[str] = [
    "stale_after_update",  # share of updates after which the next backward pass still found old gradients
    "stale_any",  # share of backward passes that found old gradients (also legitimate gradient accumulation)
    "has_backward_checks",
    "log_step_over_grad",  # log10(median |weight change| / |gradient|): the step size actually used
    "first_last_grad",  # median log10(first weight's gradient norm / last weight's)
    "saturated_fraction",  # share of tanh/sigmoid outputs within 1% of their limits
    "has_saturating",
    "relu_zero_all",  # share of zeros among F.relu/torch.relu/nn.ReLU outputs
    "has_relu_all",
    "adaptive_optimizer",  # Adam, RMSprop and the like
    "manual_updates",  # weights changed with no optimizer: updated by hand
    "loss_q25",  # log(loss a quarter of the way in / initial loss)
    "loss_q50",
    "loss_q75",
    "loss_min_ratio",  # log(lowest loss / initial loss)
]
FEATURES = V3_FEATURES + TRACE_V4 + CODE_FEATURES
CHANCE = {"binary_cross_entropy": 0.6931, "binary_cross_entropy_with_logits": 0.6931}


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


def _loss_features(calls: List[Dict[str, Any]]) -> Dict[str, float]:
    finite = [c for c in calls if c.get("value") is not None and math.isfinite(c["value"]) and c["value"] > 0]
    train = [c for c in finite if c["train"]]
    val = [c for c in finite if not c["train"]]
    chance_of = [math.log(c["classes"]) if c.get("classes") and c["classes"] >= 2 else CHANCE.get(c["fn"])
                 for c in train]
    chances = [x for x in chance_of if x]
    chance = _mean(chances) if chances else None
    start, end = _mean([c["value"] for c in train[:3]]), _mean([c["value"] for c in train[-3:]])
    val_end = _mean([c["value"] for c in val[-3:]])
    all_train = [c for c in calls if c["train"]]
    return {
        "sum_reduction": _mean([1.0 if c["reduction"] == "sum" else 0.0 for c in all_train]),
        "start_vs_chance": _log10(start / chance) if chance and start else 0.0,
        "end_vs_chance": _log10(end / chance) if chance and end else 0.0,
        "has_chance": 1.0 if chance else 0.0,
        "val_slope": max(-20.0, min(20.0, _slope([math.log(c["value"]) for c in val]))) if len(val) > 1 else 0.0,
        "val_vs_train_end": _log10(val_end / end) if val and end else 0.0,
        "val_end_vs_chance": _log10(val_end / chance) if val and chance else 0.0,
        "has_validation_loss": 1.0 if val else 0.0,
    }


def _median(values: Sequence[float]) -> Optional[float]:
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    return values[len(values) // 2] if values else None


def _v4_features(trace: Dict[str, Any], positive: List[float]) -> Dict[str, float]:
    head = _mean(positive[:3])

    def at(fraction: float) -> float:
        if not positive or not head:
            return 0.0
        i = min(len(positive) - 1, int(fraction * (len(positive) - 1)))
        window = positive[max(0, i - 1):i + 2]
        return max(-10.0, min(10.0, math.log(_mean(window) / head)))

    step = _median(trace.get("bw_step_over_grad") or [])
    ratio = _median(trace.get("layer_grad_ratios") or [])
    saturated, relu = trace.get("saturated_fraction"), trace.get("relu_zero_all")
    return {
        "stale_after_update": ((trace.get("stale_after_update") or 0) / trace["updates_seen"]
                               if trace.get("updates_seen") else 0.0),
        "stale_any": (trace.get("stale_any") or 0) / trace["stale_checks"] if trace.get("stale_checks") else 0.0,
        "has_backward_checks": 1.0 if trace.get("stale_checks") else 0.0,
        "log_step_over_grad": _log10(step) if step else 0.0,
        "first_last_grad": max(-6.0, min(6.0, ratio)) if ratio is not None else 0.0,
        "saturated_fraction": float(saturated) if saturated is not None else 0.0,
        "has_saturating": 1.0 if saturated is not None else 0.0,
        "relu_zero_all": float(relu) if relu is not None else 0.0,
        "has_relu_all": 1.0 if relu is not None else 0.0,
        "adaptive_optimizer": 1.0 if trace.get("adaptive_optimizer") else 0.0,
        "manual_updates": 1.0 if not trace.get("optimizer_kinds") and trace.get("updates_seen") else 0.0,
        "loss_q25": at(0.25), "loss_q50": at(0.5), "loss_q75": at(0.75),
        "loss_min_ratio": max(-10.0, min(10.0, math.log(min(positive) / head))) if positive and head else 0.0,
    }


def vectorize(report: Dict[str, Any], source: Optional[str] = None) -> List[float]:
    """Feature vector for one `script_runner --trace` report (its `losses` plus its `trace`) and the script's code."""
    trace = report.get("trace") or {}
    losses = [float(v) for v in report.get("losses") or []]
    finite = [v for v in losses if math.isfinite(v)]
    positive = [v for v in finite if v > 0]
    head, tail = _mean(positive[:3]), _mean(positive[-3:])
    # with no optimizer, the gradient and step statistics come from the backward passes
    raw_grads = trace.get("grad_norms") or trace.get("bw_grad_norms") or []
    grads = [g for g in raw_grads if math.isfinite(g)]
    first, last = _mean(grads[:5]), _mean(grads[-5:])
    grad_mean = _mean(grads)
    grad_std = math.sqrt(_mean([(g - grad_mean) ** 2 for g in grads])) if grads else 0.0
    updates = [u for u in trace.get("update_ratios") or trace.get("bw_update_ratios") or [] if math.isfinite(u)]
    steps = max(1, len(raw_grads))  # every step, including ones whose gradient was NaN
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
        "eval_missing_where_it_matters": ((trace.get("no_grad_training_forwards") or 0) / no_grad
                                          if no_grad and trace.get("mode_layers") else 0.0),
        "has_mode_layers": 1.0 if trace.get("mode_layers") else 0.0,
        "outside_optimizer": float(trace.get("outside_optimizer_fraction") or 0.0),
        "bias_mean": math.tanh(float(bias)) if bias is not None else 0.0,
        "has_updates": 1.0 if updates else 0.0,
        **_loss_features(trace.get("loss_calls") or []),
        **_v4_features(trace, positive),
        **(code_features(source) if source is not None else {name: 0.0 for name in CODE_FEATURES}),
    }
    # a diverged run can leave NaN weights (so a NaN mean bias): `loss_nonfinite` already says it diverged
    return [float(values[name]) if math.isfinite(values[name]) else 0.0 for name in FEATURES]
