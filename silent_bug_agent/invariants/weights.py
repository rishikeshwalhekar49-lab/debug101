from collections import defaultdict
from typing import Dict

from .base import Invariant, InvariantResult, Severity, TrainingState


class FrozenWeightsInvariant(Invariant):
    """Flags a parameter that should be training but hasn't moved.

    Catches the "accidentally frozen layer" class of bug: requires_grad
    is True, but a misplaced no_grad() block, an optimizer that was
    never given this parameter, or a zeroed learning rate means the
    weight never actually updates. A single near-zero delta is normal
    noise; `patience` consecutive near-zero deltas is the signal.
    """

    id = "frozen_weights"
    description = "A trainable parameter's weights must change over a window of steps."

    def __init__(self, delta_threshold: float = 1e-9, patience: int = 20):
        self.delta_threshold = delta_threshold
        self.patience = patience
        self._stale_counts: Dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self._stale_counts.clear()

    def check(self, state: TrainingState) -> InvariantResult:
        offenders = {}
        for name, delta in state.weight_deltas.items():
            trainable = state.requires_grad.get(name, True)
            if not trainable:
                self._stale_counts[name] = 0
                continue
            if delta <= self.delta_threshold:
                self._stale_counts[name] += 1
            else:
                self._stale_counts[name] = 0
            if self._stale_counts[name] >= self.patience:
                offenders[name] = self._stale_counts[name]

        if not offenders:
            return InvariantResult(self.id, True, Severity.INFO, "All trainable weights are updating")

        return InvariantResult(
            self.id,
            False,
            Severity.WARNING,
            f"{len(offenders)} trainable parameter(s) unchanged for >= {self.patience} steps",
            {"offenders": offenders, "step": state.step},
        )
