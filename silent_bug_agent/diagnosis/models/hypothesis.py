"""
Structured output models for the Silent Bug Agent diagnosis pipeline.

:class:`Hypothesis` represents a single ranked root-cause hypothesis.
:class:`DiagnosisResult` is the complete response returned by the agent,
containing ranked hypotheses, diagnosis status, evidence summary, and a
JSON-serialisation helper.
"""

from __future__ import annotations

import json
import logging
import math
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DiagnosisStatus(str, Enum):
    """Overall status of a :class:`DiagnosisResult`.

    * ``diagnosed``              — at least one hypothesis with confidence ≥ 0.40.
    * ``ambiguous``              — multiple competing hypotheses or confidence < 0.40.
    * ``insufficient_evidence``  — no meaningful hypothesis could be formed.
    """

    DIAGNOSED = "diagnosed"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ---------------------------------------------------------------------------
# Evidence reference
# ---------------------------------------------------------------------------


class EvidenceRef(BaseModel):
    """Reference to a specific source finding that supports or contradicts a hypothesis.

    Attributes:
        finding_id: The ``id`` field of the source finding.
        source: ``"ast"``, ``"semantic"``, or ``"runtime"``.
        message: Short description from the original finding.
        file: Source file (if available).
        line: Line number (if available).
    """

    finding_id: str
    source: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


class Hypothesis(BaseModel):
    """A single ranked root-cause hypothesis.

    Attributes:
        rank: Integer rank, starting at 1 (lower = more likely).
        root_cause_id: Stable taxonomy category identifier.
        root_cause: Human-readable root-cause name.
        confidence: Evidence-based confidence score in [0.0, 1.0].
            This is **not** a statistically calibrated probability; it reflects
            the strength and consistency of the supporting evidence relative to
            the maximum possible score for the matched pattern.
        explanation: Narrative explanation of why this root cause is suspected.
        supporting_evidence: Findings that strengthen this hypothesis.
        contradicting_evidence: Findings that weaken this hypothesis.
        causal_chain: Ordered list of steps from likely cause to observed symptom.
        affected_files: ``{file_path: [line_numbers]}`` where the issue manifests.
        verification_checks: Actionable steps to confirm or reject this hypothesis.
        alternative_hypotheses: Related plausible root-cause IDs worth considering.
    """

    rank: int = Field(..., ge=1, description="Rank (1 = highest confidence).")
    root_cause_id: str = Field(..., description="Taxonomy category ID.")
    root_cause: str = Field(..., description="Human-readable root-cause name.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Evidence-based confidence score in [0.0, 1.0].",
    )
    explanation: str = Field(..., description="Narrative evidence-based explanation.")
    supporting_evidence: List[EvidenceRef] = Field(
        default_factory=list, description="Findings supporting this hypothesis."
    )
    contradicting_evidence: List[EvidenceRef] = Field(
        default_factory=list, description="Findings contradicting this hypothesis."
    )
    causal_chain: List[str] = Field(
        default_factory=list,
        description="Steps from likely cause to observed symptom.",
    )
    affected_files: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Affected file paths mapped to relevant line numbers.",
    )
    verification_checks: List[str] = Field(
        default_factory=list, description="Actionable verification steps."
    )
    alternative_hypotheses: List[str] = Field(
        default_factory=list, description="Alternative taxonomy IDs worth considering."
    )

    @model_validator(mode="after")
    def _validate_confidence_finite(self) -> "Hypothesis":
        if not math.isfinite(self.confidence):
            raise ValueError(
                f"Hypothesis confidence must be a finite float; got {self.confidence!r}"
            )
        return self

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Diagnosis result
# ---------------------------------------------------------------------------


class DiagnosisResult(BaseModel):
    """Complete response from the :class:`~silent_bug_agent.agents.diagnosis_agent.DiagnosisAgent`.

    Attributes:
        status: Overall diagnosis status.
        primary_hypothesis: The top-ranked hypothesis, or ``None`` if no diagnosis.
        hypotheses: All ranked hypotheses in descending confidence order.
        evidence_summary: Free-text summary of the evidence considered.
        evidence_counts: ``{"ast": n, "semantic": m, "runtime": k}`` input counts.
        warnings: Any warnings about incomplete, ambiguous, or contradictory evidence.
    """

    status: DiagnosisStatus
    primary_hypothesis: Optional[Hypothesis] = None
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    evidence_summary: str = Field(
        default="", description="Summary of evidence considered during diagnosis."
    )
    evidence_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of findings per source category.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about evidence quality or completeness.",
    )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict representation of the diagnosis.

        All nested Pydantic models are converted to dicts.
        """
        return self.model_dump(mode="python")

    def to_json(self, indent: int = 2) -> str:
        """Return a JSON string representation of the diagnosis."""
        return self.model_dump_json(indent=indent)

    # ------------------------------------------------------------------
    # Validation invariants
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_invariants(self) -> "DiagnosisResult":
        # Ranks must be consecutive starting at 1
        if self.hypotheses:
            for i, h in enumerate(self.hypotheses, start=1):
                if h.rank != i:
                    raise ValueError(
                        f"Hypothesis ranks must be consecutive starting at 1; "
                        f"expected rank {i}, got {h.rank} for {h.root_cause_id!r}."
                    )
        # primary_hypothesis must be present when diagnosed
        if self.status == DiagnosisStatus.DIAGNOSED and self.primary_hypothesis is None:
            raise ValueError(
                "DiagnosisResult.status is 'diagnosed' but primary_hypothesis is None."
            )
        return self

    model_config = {"extra": "ignore"}
