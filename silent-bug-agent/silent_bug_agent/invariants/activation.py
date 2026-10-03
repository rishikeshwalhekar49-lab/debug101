from .base import Invariant, InvariantResult, Severity, TrainingState


class ActivationHealthInvariant(Invariant):
    """Flags NaN/Inf activations and large fractions of dead units.

    NaN/Inf is always critical — it will silently propagate and corrupt
    every downstream computation. A high zero_frac (e.g. most ReLU
    outputs are zero) is a softer signal of dying units and is reported
    as a warning, since some sparsity is normal.
    """

    id = "activation_health"
    description = "Layer activations must be finite and not mostly dead."

    def __init__(self, dead_frac_threshold: float = 0.9):
        self.dead_frac_threshold = dead_frac_threshold

    def check(self, state: TrainingState) -> InvariantResult:
        nan_inf_offenders = {}
        dead_offenders = {}
        for layer, stats in state.activation_stats.items():
            if stats.get("nan_count", 0) > 0 or stats.get("inf_count", 0) > 0:
                nan_inf_offenders[layer] = {
                    "nan_count": stats.get("nan_count", 0),
                    "inf_count": stats.get("inf_count", 0),
                }
            elif stats.get("zero_frac", 0.0) >= self.dead_frac_threshold:
                dead_offenders[layer] = stats.get("zero_frac", 0.0)

        if not nan_inf_offenders and not dead_offenders:
            return InvariantResult(self.id, True, Severity.INFO, "Activations are finite and active")

        if nan_inf_offenders:
            return InvariantResult(
                self.id,
                False,
                Severity.CRITICAL,
                f"{len(nan_inf_offenders)} layer(s) producing NaN/Inf activations",
                {"nan_inf_offenders": nan_inf_offenders, "dead_offenders": dead_offenders, "step": state.step},
            )

        return InvariantResult(
            self.id,
            False,
            Severity.WARNING,
            f"{len(dead_offenders)} layer(s) with >= {self.dead_frac_threshold:.0%} dead activations",
            {"dead_offenders": dead_offenders, "step": state.step},
        )
