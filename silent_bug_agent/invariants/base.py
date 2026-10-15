"""Core types shared by every invariant.

TrainingState is the contract between whatever produces training data
(a live PyTorch run in Module 1, a replayed log, or a synthetic test
fixture here in Module 0) and the invariants that check it. Nothing in
this module or in the concrete invariants imports torch — that keeps
the library testable without a GPU or even torch installed, and keeps
Module 0 usable standalone before Module 1's hooks exist.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class InvariantResult:
    invariant_id: str
    passed: bool
    severity: Severity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL[{self.severity.value}]"
        return f"<{status} {self.invariant_id}: {self.message}>"


@dataclass
class TrainingState:
    """A framework-agnostic snapshot of training at a single step.

    Fields are intentionally plain Python (dicts, lists, floats) so the
    same invariants can run against a live PyTorch run (Module 1), a
    replayed training log, or a synthetic fixture in a test.

    Attributes:
        step: Global training step this snapshot corresponds to.
        loss_history: Loss values up to and including this step, oldest
            first. Only the most recent window is needed by most
            invariants, but keep the full history for the demo/tests.
        grad_norms: {parameter_name: L2 norm of its gradient} for this step.
        weight_deltas: {parameter_name: L2 norm of (weight_now - weight_prev)}
            since the previous optimizer step.
        requires_grad: {parameter_name: bool} — whether the parameter is
            currently trainable.
        activation_stats: {layer_name: {"mean", "std", "nan_count",
            "inf_count", "zero_frac"}} computed over that layer's output
            for the current batch.
        module_modes: {module_name: "train" | "eval"} — the .training
            flag of named submodules (BatchNorm, Dropout, etc.) at this step.
        is_eval_phase: True if this snapshot was taken during an
            evaluation/validation pass rather than a training step.
        extra: Optional producer-specific data that the invariants do not
            need, e.g. loss dynamics (EMA, z-score, spike flag), global
            gradient/weight norms, learning rates and histograms written by
            the Module 1 runtime hooks. Plain Python values only.
    """

    step: int
    loss_history: List[float] = field(default_factory=list)
    grad_norms: Dict[str, float] = field(default_factory=dict)
    weight_deltas: Dict[str, float] = field(default_factory=dict)
    requires_grad: Dict[str, bool] = field(default_factory=dict)
    activation_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    module_modes: Dict[str, str] = field(default_factory=dict)
    is_eval_phase: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class Invariant:
    """Base class for a single checkable training invariant.

    Subclasses may keep internal state (rolling windows, per-parameter
    counters) across successive check() calls, since a single
    TrainingState only carries one step's snapshot but many invariants
    need trend information across steps.
    """

    id: str = "base_invariant"
    description: str = ""

    def check(self, state: TrainingState) -> InvariantResult:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any internal rolling state. Call between independent runs."""
        return None
