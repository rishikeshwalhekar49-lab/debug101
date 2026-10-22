"""
Input data models for the Silent Bug Agent.

Three categories of findings are accepted:

* :class:`ASTFinding`   — findings from static AST analysis.
* :class:`SemanticFinding` — findings from semantic code review.
* :class:`RuntimeAlert` — runtime invariant violation alerts.

All models are Pydantic v2 ``BaseModel`` subclasses so they can be
constructed from plain dicts (``Model(**d)``) or validated from JSON.

The :class:`NormalizedFinding` is an internal representation used by the
causal inference engine; it is not part of the public API.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Finding severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Source category enum (used by NormalizedFinding)
# ---------------------------------------------------------------------------


class SourceCategory(str, Enum):
    """Which analysis phase produced a finding."""

    AST = "ast"
    SEMANTIC = "semantic"
    RUNTIME = "runtime"


# ---------------------------------------------------------------------------
# A. AST Static Finding
# ---------------------------------------------------------------------------


class ASTFinding(BaseModel):
    """A finding produced by static AST analysis.

    Attributes:
        id: Unique identifier for the finding (e.g. ``"AST-001"``).
        type: Finding type / category hint (may be a raw string or a taxonomy ID).
        message: Human-readable description of the finding.
        file: Source file path (optional).
        line: Line number in *file* where the issue was detected (optional).
        severity: Severity level (default: ``"medium"``).
        code_snippet: Short code excerpt (optional).
        metadata: Any additional key-value data the analyser wants to attach.
    """

    id: str = Field(..., description="Unique finding identifier.")
    type: str = Field(..., description="Finding type / category hint.")
    message: str = Field(..., description="Human-readable description.")
    file: Optional[str] = Field(None, description="Source file path.")
    line: Optional[int] = Field(None, ge=1, description="Line number (1-indexed).")
    severity: Severity = Field(Severity.MEDIUM, description="Severity level.")
    code_snippet: Optional[str] = Field(None, description="Code excerpt.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata."
    )

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# B. Semantic Review Finding
# ---------------------------------------------------------------------------


class SemanticFinding(BaseModel):
    """A finding produced by semantic code review.

    Attributes:
        id: Unique identifier (e.g. ``"SEM-001"``).
        type: Semantic issue category.
        message: Human-readable description.
        file: Source file path (optional).
        line: Line number (optional).
        severity: Severity level (default: ``"medium"``).
        context: Additional semantic context (e.g. surrounding logic description).
        metadata: Any additional key-value data.
    """

    id: str = Field(..., description="Unique finding identifier.")
    type: str = Field(..., description="Semantic issue category.")
    message: str = Field(..., description="Human-readable description.")
    file: Optional[str] = Field(None, description="Source file path.")
    line: Optional[int] = Field(None, ge=1, description="Line number (1-indexed).")
    severity: Severity = Field(Severity.MEDIUM, description="Severity level.")
    context: Optional[str] = Field(None, description="Semantic context description.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata."
    )

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# C. Runtime Invariant Alert
# ---------------------------------------------------------------------------


class RuntimeAlert(BaseModel):
    """A runtime invariant violation detected during training or evaluation.

    Attributes:
        id: Unique identifier (e.g. ``"RUN-001"``).
        type: Alert type (e.g. ``"gradient_instability"``, ``"loss_nan"``).
        message: Human-readable description.
        step: Training step at which the alert fired (optional).
        epoch: Training epoch (optional).
        severity: Severity level (default: ``"high"`` for runtime alerts).
        metadata: Numeric measurements, stack traces, or other context.
    """

    id: str = Field(..., description="Unique alert identifier.")
    type: str = Field(..., description="Alert type / category hint.")
    message: str = Field(..., description="Human-readable description.")
    step: Optional[int] = Field(None, ge=0, description="Training step.")
    epoch: Optional[int] = Field(None, ge=0, description="Training epoch.")
    severity: Severity = Field(Severity.HIGH, description="Severity level.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Numeric measurements or context."
    )

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Internal normalised representation
# ---------------------------------------------------------------------------


class NormalizedFinding(BaseModel):
    """Internal representation of any finding type, used by the causal engine.

    Produced by :class:`~silent_bug_agent.inference.normalizer.FindingNormalizer`
    and consumed by :class:`~silent_bug_agent.inference.causal_engine.CausalEngine`.

    Attributes:
        original_id: ID from the source finding.
        source: Which analysis phase produced this finding.
        raw_type: The original type string from the source.
        message_lower: Lower-cased message text (used for keyword matching).
        severity: Normalised severity.
        file: Source file path (if available).
        line: Line number (if available).
        metadata: Preserved metadata dict.
        candidate_categories: Taxonomy category IDs matched by the normaliser
            (may be empty if no keyword matched).
    """

    original_id: str
    source: SourceCategory
    raw_type: str
    message_lower: str
    severity: Severity
    file: Optional[str] = None
    line: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    candidate_categories: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
