"""silent_bug_agent — Phase 0: bug taxonomy and invariant library.

This package has no training-loop integration yet. It defines:
  - a framework-agnostic TrainingState snapshot shape
  - a library of Invariant checks that operate on that shape
  - a taxonomy of known silent DL bugs, cross-referenced to invariants

Phase 1 wires PyTorch hooks into TrainingState; this package deliberately
has no torch import so it can be tested and reused independent of that.
"""

from .invariants.base import Invariant, InvariantResult, Severity, TrainingState
from .invariants.registry import get_default_registry, run_all

__all__ = [
    "Invariant",
    "InvariantResult",
    "Severity",
    "TrainingState",
    "get_default_registry",
    "run_all",
]
