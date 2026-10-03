import statistics

from .base import Invariant, InvariantResult, Severity, TrainingState


class LossTrendInvariant(Invariant):
    """Flags loss that isn't trending down, without requiring monotonicity.

    Minibatch noise means loss almost never decreases step-over-step
    even in healthy training, so this deliberately does NOT check
    "did loss go down this step". Instead it waits for a minimum
    window of steps, then compares the mean loss of the first third of
    the window against the last third. If the last third isn't lower
    by at least `tolerance` (relative), it flags a stall — a much
    weaker and more honest claim than a formal monotonic-decrease spec.
    """

    id = "loss_trend"
    description = "Mean loss over the tail of a window must be lower than the head, within tolerance."

    def __init__(self, window: int = 90, tolerance: float = 0.01):
        if window % 3 != 0:
            window += 3 - (window % 3)  # keep window evenly splittable into thirds
        self.window = window
        self.tolerance = tolerance

    def check(self, state: TrainingState) -> InvariantResult:
        history = state.loss_history[-self.window :]
        if len(history) < self.window:
            return InvariantResult(
                self.id, True, Severity.INFO, f"Warming up ({len(history)}/{self.window} steps)"
            )

        third = self.window // 3
        head = history[:third]
        tail = history[-third:]
        mean_head = statistics.fmean(head)
        mean_tail = statistics.fmean(tail)

        if mean_head == 0:
            return InvariantResult(self.id, True, Severity.INFO, "Head loss is zero, skipping trend check")

        improvement = (mean_head - mean_tail) / abs(mean_head)
        if improvement >= self.tolerance:
            return InvariantResult(
                self.id, True, Severity.INFO, f"Loss improved {improvement:.1%} over the last {self.window} steps"
            )

        severity = Severity.CRITICAL if mean_tail > mean_head * 1.5 else Severity.WARNING
        return InvariantResult(
            self.id,
            False,
            severity,
            f"Loss has not meaningfully improved over the last {self.window} steps "
            f"(head mean {mean_head:.4g}, tail mean {mean_tail:.4g})",
            {"mean_head": mean_head, "mean_tail": mean_tail, "step": state.step},
        )
