"""Weight-Space Programming (WSP) Module.

Bridges code-space inspection with direct weight-space perception and surgery.
Natively reads, analyzes, diagnoses, and edits raw neural network weights
alongside source code AST patterns.
"""

from .schemas import (
    DualSolution,
    WeightAnomalyType,
    WeightFinding,
    WeightPatch,
    WeightReport,
    WeightSurgeryType,
)
from .inspector import WeightInspector
from .editor import WeightEditor

__all__ = [
    "WeightAnomalyType",
    "WeightFinding",
    "WeightReport",
    "WeightSurgeryType",
    "WeightPatch",
    "DualSolution",
    "WeightInspector",
    "WeightEditor",
]

