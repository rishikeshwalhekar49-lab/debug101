from .base import Invariant, InvariantResult, Severity, TrainingState


class GradientNormBoundsInvariant(Invariant):
    """Flags per-parameter gradient norms outside a healthy range.

    Catches both vanishing gradients (norm collapses toward zero, the
    parameter effectively stops learning) and exploding gradients
    (norm spikes, often preceding a NaN loss). Thresholds are
    intentionally configurable per project — "healthy" depends on
    architecture, batch size, and loss scale.
    """

    id = "gradient_norm_bounds"
    description = "Gradient norm per parameter must stay within [min_norm, max_norm]."

    def __init__(self, min_norm: float = 1e-7, max_norm: float = 1e3):
        self.min_norm = min_norm
        self.max_norm = max_norm

    def check(self, state: TrainingState) -> InvariantResult:
        if not state.grad_norms:
            return InvariantResult(self.id, True, Severity.INFO, "No gradients recorded this step")

        offenders = {
            name: norm
            for name, norm in state.grad_norms.items()
            if norm < self.min_norm or norm > self.max_norm or norm != norm  # last check catches NaN
        }
        if not offenders:
            return InvariantResult(self.id, True, Severity.INFO, "Gradient norms within bounds")

        exploding = any(v != v or v > self.max_norm for v in offenders.values())
        severity = Severity.CRITICAL if exploding else Severity.WARNING
        return InvariantResult(
            self.id,
            False,
            severity,
            f"{len(offenders)} parameter(s) with gradient norm outside "
            f"[{self.min_norm}, {self.max_norm}]",
            {"offenders": offenders, "step": state.step},
        )
