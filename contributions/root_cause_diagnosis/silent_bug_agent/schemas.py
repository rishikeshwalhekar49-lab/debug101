"""
silent_bug_agent/schemas.py — Public schema re-exports.

This module re-exports all public input and output data models so that
callers can import from a single ``schemas`` namespace:

    from silent_bug_agent.schemas import ASTFinding, SemanticFinding
    from silent_bug_agent.schemas import RuntimeAlert, Hypothesis, DiagnosisResult

The canonical implementations live in:
    * :mod:`silent_bug_agent.models.findings`   — input models
    * :mod:`silent_bug_agent.models.hypothesis` — output models

This file exists for API convenience and naming compatibility.
"""

from silent_bug_agent.models.findings import (
    ASTFinding,
    NormalizedFinding,
    RuntimeAlert,
    SemanticFinding,
    Severity,
    SourceCategory,
)
from silent_bug_agent.models.hypothesis import (
    DiagnosisResult,
    DiagnosisStatus,
    EvidenceRef,
    Hypothesis,
)

__all__ = [
    # Input models
    "ASTFinding",
    "SemanticFinding",
    "RuntimeAlert",
    "NormalizedFinding",
    "Severity",
    "SourceCategory",
    # Output models
    "Hypothesis",
    "EvidenceRef",
    "DiagnosisResult",
    "DiagnosisStatus",
]
