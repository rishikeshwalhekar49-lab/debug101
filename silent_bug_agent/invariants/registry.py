from typing import List

from .activation import ActivationHealthInvariant
from .base import Invariant, InvariantResult, TrainingState
from .gradient import GradientNormBoundsInvariant
from .loss import LossTrendInvariant
from .mode import TrainEvalModeInvariant
from .weights import FrozenWeightsInvariant


def get_default_registry() -> List[Invariant]:
    """Return one instance of each built-in invariant, default thresholds.

    Callers that need custom thresholds (a different loss scale, a
    stricter frozen-weight patience, etc.) should construct the
    relevant Invariant subclasses directly instead of using this.
    """
    return [
        GradientNormBoundsInvariant(),
        FrozenWeightsInvariant(),
        TrainEvalModeInvariant(),
        ActivationHealthInvariant(),
        LossTrendInvariant(),
    ]


def run_all(invariants: List[Invariant], state: TrainingState) -> List[InvariantResult]:
    """Run every invariant against one TrainingState snapshot."""
    return [inv.check(state) for inv in invariants]
