"""
Deterministic Causal Inference Engine for ML/DL bug diagnosis.

This module implements the core scoring, correlation, and ranking algorithm
that converts a list of :class:`~silent_bug_agent.diagnosis.models.findings.NormalizedFinding`
objects into ranked :class:`~silent_bug_agent.diagnosis.models.hypothesis.Hypothesis` objects.

Algorithm overview
------------------
1.  **Category scoring** — For each taxonomy category, accumulate an evidence
    score from findings that matched that category.
2.  **Source-reliability weighting** — AST, semantic, and runtime evidence each
    carry a different base weight (documented below).
3.  **Severity multiplier** — Higher-severity findings contribute more evidence.
4.  **Duplicate evidence penalty** — If the same category is supported by
    multiple findings from the *same source*, additional findings contribute
    diminishing returns to avoid inflating confidence.
5.  **Contradicting evidence penalty** — Findings that explicitly contradict a
    category reduce its score.
6.  **Causal graph dampening** — Categories that are downstream symptoms of a
    higher-scoring upstream cause have their score dampened, preventing
    downstream effects from outranking their root cause.
7.  **Single-source penalty** — If only one evidence source contributed,
    confidence is reduced.
8.  **Normalisation** — Scores are clamped to [0.0, 1.0].
9.  **Hypothesis construction** — Top-scoring categories become hypotheses with
    supporting/contradicting evidence references, causal chains, and
    verification checks.

Weight constants
----------------
All weights are module-level constants so they are independently testable and
easily documented.

``W_AST``      = 0.30  — AST findings: structural, syntactic evidence.
``W_SEMANTIC`` = 0.40  — Semantic findings: higher-level logical evidence.
``W_RUNTIME``  = 0.50  — Runtime alerts: direct observed symptoms.
``W_CAUSAL``   = 0.20  — Bonus for causal graph match (upstream cause found).
``P_DUPLICATE``= 0.10  — Penalty per duplicate finding from same source.
``P_CONTRADICT``= 0.25 — Penalty per explicitly contradicting finding.
``P_SINGLE_SOURCE`` = 0.10 — Penalty when only one source type contributed.
``SEVERITY_MULTIPLIERS`` — Low:0.6, Medium:1.0, High:1.3, Critical:1.5.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from silent_bug_agent.diagnosis.knowledge.taxonomy import TAXONOMY_REGISTRY, TaxonomyEntry
from silent_bug_agent.diagnosis.models.findings import NormalizedFinding, Severity, SourceCategory
from silent_bug_agent.diagnosis.models.hypothesis import (
    DiagnosisResult,
    DiagnosisStatus,
    EvidenceRef,
    Hypothesis,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight constants — all scoring parameters live here for testability
# ---------------------------------------------------------------------------

# Base weights per evidence source
W_AST: float = 0.30
W_SEMANTIC: float = 0.40
W_RUNTIME: float = 0.50

# Causal graph bonus when evidence implies an upstream root cause
W_CAUSAL: float = 0.20

# Penalties
P_DUPLICATE: float = 0.10   # per extra finding from the same source for the same category
P_CONTRADICT: float = 0.25  # per contradicting finding
P_SINGLE_SOURCE: float = 0.10  # when only one source type contributed evidence

# Severity multipliers
SEVERITY_MULTIPLIERS: Dict[str, float] = {
    Severity.LOW.value: 0.60,
    Severity.MEDIUM.value: 1.00,
    Severity.HIGH.value: 1.30,
    Severity.CRITICAL.value: 1.50,
}

# Minimum confidence threshold to consider a hypothesis "diagnosed"
DIAGNOSIS_CONFIDENCE_THRESHOLD: float = 0.40

# Minimum confidence to include a hypothesis in results at all
MIN_CONFIDENCE_THRESHOLD: float = 0.05

# Maximum number of hypotheses to return
MAX_HYPOTHESES: int = 5

# Causal dampening factor applied to downstream categories
DOWNSTREAM_DAMPENING: float = 0.60

# Source base weight lookup
_SOURCE_WEIGHTS: Dict[SourceCategory, float] = {
    SourceCategory.AST: W_AST,
    SourceCategory.SEMANTIC: W_SEMANTIC,
    SourceCategory.RUNTIME: W_RUNTIME,
}


# ---------------------------------------------------------------------------
# Internal data structure: per-category accumulated evidence
# ---------------------------------------------------------------------------


class _CategoryEvidence:
    """Mutable accumulator for evidence supporting/contradicting a category."""

    __slots__ = (
        "category_id",
        "raw_score",
        "supporting",
        "contradicting",
        "source_counts",
    )

    def __init__(self, category_id: str) -> None:
        self.category_id: str = category_id
        self.raw_score: float = 0.0
        self.supporting: List[NormalizedFinding] = []
        self.contradicting: List[NormalizedFinding] = []
        # track how many findings per source contributed
        self.source_counts: Dict[SourceCategory, int] = defaultdict(int)


# ---------------------------------------------------------------------------
# Public causal engine
# ---------------------------------------------------------------------------


class CausalEngine:
    """Deterministic causal inference engine.

    Usage::

        engine = CausalEngine()
        hypotheses = engine.infer(ast_findings, semantic_findings, runtime_alerts)

    All arguments are lists of :class:`NormalizedFinding`.

    The engine is stateless between calls and is safe to reuse.
    """

    def __init__(self) -> None:
        self._taxonomy = TAXONOMY_REGISTRY
        self._valid_ids: FrozenSet[str] = self._taxonomy.category_ids()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def infer(
        self,
        ast_findings: List[NormalizedFinding],
        semantic_findings: List[NormalizedFinding],
        runtime_alerts: List[NormalizedFinding],
    ) -> DiagnosisResult:
        """Run causal inference and return a :class:`DiagnosisResult`.

        Args:
            ast_findings: Normalised AST findings.
            semantic_findings: Normalised semantic findings.
            runtime_alerts: Normalised runtime alerts.

        Returns:
            A fully populated :class:`DiagnosisResult`.
        """
        all_findings = ast_findings + semantic_findings + runtime_alerts

        evidence_counts = {
            "ast": len(ast_findings),
            "semantic": len(semantic_findings),
            "runtime": len(runtime_alerts),
        }

        # --- Step 1: Accumulate evidence per category ---
        category_evidence = self._accumulate_evidence(all_findings)

        # --- Step 2: Apply causal graph dampening ---
        self._apply_causal_dampening(category_evidence)

        # --- Step 3: Compute normalised confidence ---
        scored = self._compute_confidence(category_evidence, all_findings)

        # --- Step 4: Filter and rank ---
        ranked = self._rank(scored)

        # --- Step 5: Build hypothesis objects ---
        hypotheses = self._build_hypotheses(ranked, category_evidence)

        # --- Step 6: Determine status and assemble result ---
        warnings = self._generate_warnings(all_findings, category_evidence, hypotheses)
        status, primary = self._determine_status(hypotheses)
        evidence_summary = self._build_summary(evidence_counts, hypotheses, warnings)

        return DiagnosisResult(
            status=status,
            primary_hypothesis=primary,
            hypotheses=hypotheses,
            evidence_summary=evidence_summary,
            evidence_counts=evidence_counts,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Step 1: Accumulate evidence
    # ------------------------------------------------------------------

    def _accumulate_evidence(
        self, findings: List[NormalizedFinding]
    ) -> Dict[str, _CategoryEvidence]:
        """Score each taxonomy category based on matching findings."""
        evidence: Dict[str, _CategoryEvidence] = {}

        for finding in findings:
            if not finding.candidate_categories:
                # Finding didn't match any category — contributes nothing
                continue

            source_weight = _SOURCE_WEIGHTS.get(finding.source, W_AST)
            severity_mult = SEVERITY_MULTIPLIERS.get(finding.severity.value, 1.0)

            for cat_id in finding.candidate_categories:
                if cat_id not in self._valid_ids:
                    logger.warning(
                        "Normalizer produced unknown category %r for finding %s; skipping.",
                        cat_id,
                        finding.original_id,
                    )
                    continue

                if cat_id not in evidence:
                    evidence[cat_id] = _CategoryEvidence(cat_id)

                ev = evidence[cat_id]

                # Duplicate penalty: first finding from this source scores fully,
                # each subsequent one is penalised.
                prior_count = ev.source_counts[finding.source]
                effective_weight = max(
                    source_weight * severity_mult - P_DUPLICATE * prior_count,
                    0.0,
                )

                ev.raw_score += effective_weight
                ev.supporting.append(finding)
                ev.source_counts[finding.source] += 1

        return evidence

    # ------------------------------------------------------------------
    # Step 2: Causal graph dampening
    # ------------------------------------------------------------------

    def _apply_causal_dampening(
        self, evidence: Dict[str, _CategoryEvidence]
    ) -> None:
        """Dampen downstream categories when a likely upstream cause is present.

        If category A causes category B (per the causal graph) and A has a
        higher raw score than B, then B's score is reduced so that B does not
        outrank A as a root cause.
        """
        for source_id, source_ev in list(evidence.items()):
            downstream = self._taxonomy.causes(source_id)
            for target_id, edge_weight in downstream.items():
                if target_id in evidence:
                    target_ev = evidence[target_id]
                    # Only dampen if source is a plausible cause (non-trivial score)
                    if source_ev.raw_score > 0.1:
                        # Scale dampening by how strongly the edge predicts causation
                        dampen_by = DOWNSTREAM_DAMPENING * edge_weight
                        target_ev.raw_score = max(
                            target_ev.raw_score * (1.0 - dampen_by),
                            0.0,
                        )

    # ------------------------------------------------------------------
    # Step 3: Compute normalised confidence
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        evidence: Dict[str, _CategoryEvidence],
        all_findings: List[NormalizedFinding],
    ) -> Dict[str, float]:
        """Compute a normalised confidence score in [0.0, 1.0] for each category.

        The maximum achievable score is estimated as the sum of maximum possible
        weights across all findings if they all pointed to one category.
        """
        if not evidence:
            return {}

        # Maximum possible score: all findings at max source weight * max severity
        max_possible = sum(
            _SOURCE_WEIGHTS.get(f.source, W_AST)
            * SEVERITY_MULTIPLIERS.get(f.severity.value, 1.0)
            for f in all_findings
        ) or 1.0  # avoid div/zero

        scored: Dict[str, float] = {}
        for cat_id, ev in evidence.items():
            raw = ev.raw_score

            # Single-source penalty
            active_sources = sum(1 for v in ev.source_counts.values() if v > 0)
            if active_sources == 1:
                raw = max(raw - P_SINGLE_SOURCE, 0.0)

            # Contradicting evidence penalty (applied after normalisation)
            contra_penalty = P_CONTRADICT * len(ev.contradicting)

            # Normalise
            conf = raw / max_possible
            conf = max(conf - contra_penalty, 0.0)
            conf = min(conf, 1.0)

            scored[cat_id] = conf

        return scored

    # ------------------------------------------------------------------
    # Step 4: Rank categories
    # ------------------------------------------------------------------

    def _rank(self, scored: Dict[str, float]) -> List[Tuple[str, float]]:
        """Return category IDs sorted by confidence descending, filtered by threshold."""
        ranked = [
            (cat_id, conf)
            for cat_id, conf in scored.items()
            if conf >= MIN_CONFIDENCE_THRESHOLD
        ]
        ranked.sort(key=lambda x: (-x[1], x[0]))  # stable: ties sorted by ID
        return ranked[:MAX_HYPOTHESES]

    # ------------------------------------------------------------------
    # Step 5: Build hypothesis objects
    # ------------------------------------------------------------------

    def _build_hypotheses(
        self,
        ranked: List[Tuple[str, float]],
        evidence: Dict[str, _CategoryEvidence],
    ) -> List[Hypothesis]:
        if not ranked:
            fallback = self._build_fallback_hypothesis(rank=1)
            return [fallback]

        hypotheses: List[Hypothesis] = []
        all_ranked_ids = [cat_id for cat_id, _ in ranked]

        for rank_idx, (cat_id, confidence) in enumerate(ranked, start=1):
            entry = self._taxonomy.lookup(cat_id)
            ev = evidence[cat_id]

            supporting_refs = self._build_evidence_refs(ev.supporting)
            contradicting_refs = self._build_evidence_refs(ev.contradicting)
            causal_chain = self._build_causal_chain(cat_id, confidence)
            affected_files = self._extract_affected_files(ev.supporting)
            alternatives = [c for c in all_ranked_ids if c != cat_id][:3]
            explanation = self._build_explanation(cat_id, entry, ev, confidence)

            h = Hypothesis(
                rank=rank_idx,
                root_cause_id=cat_id,
                root_cause=entry.name,
                confidence=round(confidence, 4),
                explanation=explanation,
                supporting_evidence=supporting_refs,
                contradicting_evidence=contradicting_refs,
                causal_chain=causal_chain,
                affected_files=affected_files,
                verification_checks=list(entry.verification_checks),
                alternative_hypotheses=alternatives,
            )
            hypotheses.append(h)

        return hypotheses

    def _build_fallback_hypothesis(self, rank: int) -> Hypothesis:
        """Build the fallback 'unknown/insufficient evidence' hypothesis."""
        entry = self._taxonomy.lookup("unknown_or_insufficient_evidence")
        return Hypothesis(
            rank=rank,
            root_cause_id="unknown_or_insufficient_evidence",
            root_cause=entry.name,
            confidence=0.0,
            explanation=(
                "Insufficient or ambiguous evidence to identify a specific root cause. "
                "Collect additional diagnostic information to proceed."
            ),
            supporting_evidence=[],
            contradicting_evidence=[],
            causal_chain=["Diagnosis not possible with current evidence."],
            affected_files={},
            verification_checks=list(entry.verification_checks),
            alternative_hypotheses=[],
        )

    # ------------------------------------------------------------------
    # Helper builders
    # ------------------------------------------------------------------

    def _build_evidence_refs(
        self, findings: List[NormalizedFinding]
    ) -> List[EvidenceRef]:
        """Deduplicate and convert findings to :class:`EvidenceRef` objects."""
        seen_ids: Set[str] = set()
        refs: List[EvidenceRef] = []
        for f in findings:
            if f.original_id in seen_ids:
                continue
            seen_ids.add(f.original_id)
            refs.append(
                EvidenceRef(
                    finding_id=f.original_id,
                    source=f.source.value,
                    message=f.message_lower,
                    file=f.file,
                    line=f.line,
                )
            )
        return refs

    def _build_causal_chain(self, cat_id: str, confidence: float) -> List[str]:
        """Produce a short causal chain narrative for the hypothesis."""
        chain: List[str] = []
        entry = self._taxonomy.lookup(cat_id)

        # Upstream causes (parents)
        upstream = self._taxonomy.caused_by(cat_id)
        if upstream:
            top_parent = max(upstream, key=lambda k: upstream[k])
            parent_entry = self._taxonomy.lookup(top_parent)
            chain.append(f"Potential upstream cause: {parent_entry.name}")

        chain.append(f"Primary issue: {entry.name}")
        chain.append(entry.description.split(".")[0] + ".")

        # Downstream effects (children)
        downstream = self._taxonomy.causes(cat_id)
        if downstream:
            top_child = max(downstream, key=lambda k: downstream[k])
            child_entry = self._taxonomy.lookup(top_child)
            chain.append(f"Likely downstream symptom: {child_entry.name}")

        return chain

    def _extract_affected_files(
        self, findings: List[NormalizedFinding]
    ) -> Dict[str, List[int]]:
        """Collect file paths and line numbers from supporting findings."""
        files: Dict[str, List[int]] = {}
        for f in findings:
            if f.file:
                lines = files.setdefault(f.file, [])
                if f.line and f.line not in lines:
                    lines.append(f.line)
        return files

    def _build_explanation(
        self,
        cat_id: str,
        entry: TaxonomyEntry,
        ev: _CategoryEvidence,
        confidence: float,
    ) -> str:
        """Build a narrative explanation for a hypothesis."""
        n_supporting = len(set(f.original_id for f in ev.supporting))
        sources_used = [
            s.value for s, cnt in ev.source_counts.items() if cnt > 0
        ]
        sources_str = ", ".join(sources_used) if sources_used else "no sources"

        conf_desc = (
            "high" if confidence >= 0.65
            else "moderate" if confidence >= 0.40
            else "low"
        )

        return (
            f"{entry.name} is suspected with {conf_desc} confidence "
            f"(score={confidence:.2f}). "
            f"{n_supporting} finding(s) from {sources_str} source(s) support this hypothesis. "
            f"{entry.description}"
        )

    # ------------------------------------------------------------------
    # Step 6: Status, warnings, summary
    # ------------------------------------------------------------------

    def _determine_status(
        self, hypotheses: List[Hypothesis]
    ) -> Tuple[DiagnosisStatus, Optional[Hypothesis]]:
        """Determine overall diagnosis status and the primary hypothesis."""
        if not hypotheses:
            return DiagnosisStatus.INSUFFICIENT_EVIDENCE, None

        top = hypotheses[0]

        if top.root_cause_id == "unknown_or_insufficient_evidence":
            return DiagnosisStatus.INSUFFICIENT_EVIDENCE, None

        if top.confidence >= DIAGNOSIS_CONFIDENCE_THRESHOLD:
            # Check for ambiguity: second hypothesis close to the top
            if (
                len(hypotheses) > 1
                and hypotheses[1].confidence >= DIAGNOSIS_CONFIDENCE_THRESHOLD
                and (top.confidence - hypotheses[1].confidence) < 0.10
            ):
                return DiagnosisStatus.AMBIGUOUS, top
            return DiagnosisStatus.DIAGNOSED, top

        return DiagnosisStatus.AMBIGUOUS, top

    def _generate_warnings(
        self,
        all_findings: List[NormalizedFinding],
        evidence: Dict[str, _CategoryEvidence],
        hypotheses: List[Hypothesis],
    ) -> List[str]:
        """Generate informational warnings about evidence quality."""
        warnings: List[str] = []

        unmatched = [f for f in all_findings if not f.candidate_categories]
        if unmatched:
            ids = ", ".join(f.original_id for f in unmatched[:5])
            warnings.append(
                f"{len(unmatched)} finding(s) could not be matched to any taxonomy "
                f"category and were excluded from scoring: {ids}."
            )

        if not all_findings:
            warnings.append("No findings were provided; diagnosis is not possible.")
            return warnings

        source_types = {f.source for f in all_findings if f.candidate_categories}
        if len(source_types) == 1:
            warnings.append(
                f"Evidence comes from only one source type ({next(iter(source_types)).value}); "
                "cross-source corroboration is absent — confidence may be understated."
            )

        if hypotheses and hypotheses[0].confidence < DIAGNOSIS_CONFIDENCE_THRESHOLD:
            warnings.append(
                "Top hypothesis confidence is below the diagnosis threshold "
                f"({DIAGNOSIS_CONFIDENCE_THRESHOLD:.0%}); result should be treated as ambiguous."
            )

        return warnings

    def _build_summary(
        self,
        evidence_counts: Dict[str, int],
        hypotheses: List[Hypothesis],
        warnings: List[str],
    ) -> str:
        total = sum(evidence_counts.values())
        if total == 0:
            return "No findings were provided."

        parts = [
            f"Diagnosed using {total} finding(s): "
            f"{evidence_counts.get('ast', 0)} AST, "
            f"{evidence_counts.get('semantic', 0)} semantic, "
            f"{evidence_counts.get('runtime', 0)} runtime."
        ]

        if hypotheses and hypotheses[0].root_cause_id != "unknown_or_insufficient_evidence":
            top = hypotheses[0]
            parts.append(
                f"Top hypothesis: {top.root_cause} (confidence={top.confidence:.2f})."
            )

        if warnings:
            parts.append(f"{len(warnings)} warning(s) generated.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Public scoring utility (for unit tests)
    # ------------------------------------------------------------------

    def score_category(
        self,
        category_id: str,
        findings: List[NormalizedFinding],
    ) -> float:
        """Compute a raw (pre-normalisation) score for a single category.

        This is a convenience method for unit testing the scoring model
        in isolation.
        """
        evidence = self._accumulate_evidence(findings)
        if category_id not in evidence:
            return 0.0
        return evidence[category_id].raw_score
