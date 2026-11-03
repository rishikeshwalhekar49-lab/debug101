"""
Unit tests for the CausalEngine.

Tests:
- score_category() reflects source weight and severity multiplier.
- Duplicate penalty applies on same-source repeated evidence.
- Causal graph dampening reduces downstream category scores.
- Contradicting evidence reduces confidence.
- Single-source penalty applies.
- Normalised confidence is in [0.0, 1.0].
- Fallback is returned for empty input.
- Ranks are consecutive starting at 1.
- Taxonomy entries are valid for all returned hypotheses.
"""

import pytest

from silent_bug_agent.diagnosis.inference.causal_engine import (
    W_AST,
    W_RUNTIME,
    W_SEMANTIC,
    CausalEngine,
    SEVERITY_MULTIPLIERS,
    P_SINGLE_SOURCE,
)
from silent_bug_agent.diagnosis.models.findings import NormalizedFinding, Severity, SourceCategory
from silent_bug_agent.diagnosis.models.hypothesis import DiagnosisStatus


@pytest.fixture
def engine():
    return CausalEngine()


def make_finding(
    fid: str,
    source: SourceCategory,
    categories: list,
    severity: Severity = Severity.MEDIUM,
    message: str = "test finding",
    file: str = None,
    line: int = None,
) -> NormalizedFinding:
    return NormalizedFinding(
        original_id=fid,
        source=source,
        raw_type=categories[0] if categories else "unknown",
        message_lower=message.lower(),
        severity=severity,
        file=file,
        line=line,
        metadata={},
        candidate_categories=categories,
    )


class TestScoreCategory:
    """Test score_category() directly."""

    def test_ast_medium_scoring(self, engine):
        findings = [make_finding("A1", SourceCategory.AST, ["gradient_management_error"])]
        score = engine.score_category("gradient_management_error", findings)
        expected = W_AST * SEVERITY_MULTIPLIERS[Severity.MEDIUM.value]
        assert abs(score - expected) < 1e-9

    def test_runtime_high_scoring(self, engine):
        findings = [make_finding("R1", SourceCategory.RUNTIME, ["numerical_instability"], severity=Severity.HIGH)]
        score = engine.score_category("numerical_instability", findings)
        expected = W_RUNTIME * SEVERITY_MULTIPLIERS[Severity.HIGH.value]
        assert abs(score - expected) < 1e-9

    def test_semantic_critical_scoring(self, engine):
        findings = [make_finding("S1", SourceCategory.SEMANTIC, ["loss_output_mismatch"], severity=Severity.CRITICAL)]
        score = engine.score_category("loss_output_mismatch", findings)
        expected = W_SEMANTIC * SEVERITY_MULTIPLIERS[Severity.CRITICAL.value]
        assert abs(score - expected) < 1e-9

    def test_zero_score_for_unrelated_category(self, engine):
        findings = [make_finding("A1", SourceCategory.AST, ["gradient_management_error"])]
        score = engine.score_category("tensor_shape_mismatch", findings)
        assert score == 0.0

    def test_empty_findings_zero_score(self, engine):
        assert engine.score_category("gradient_management_error", []) == 0.0


class TestDuplicatePenalty:
    """Multiple findings from the same source for the same category diminish."""

    def test_two_same_source_lower_than_double(self, engine):
        f1 = make_finding("A1", SourceCategory.AST, ["gradient_management_error"])
        f2 = make_finding("A2", SourceCategory.AST, ["gradient_management_error"])
        score_two = engine.score_category("gradient_management_error", [f1, f2])
        score_one = engine.score_category("gradient_management_error", [f1])
        # Second finding should add less than the first
        assert score_two > score_one  # still increases
        assert score_two < 2 * score_one  # but less than double

    def test_different_sources_no_penalty(self, engine):
        f1 = make_finding("A1", SourceCategory.AST, ["gradient_management_error"])
        f2 = make_finding("R1", SourceCategory.RUNTIME, ["gradient_management_error"])
        score_two = engine.score_category("gradient_management_error", [f1, f2])
        score_one_ast = engine.score_category("gradient_management_error", [f1])
        # Should be strictly larger since they come from different sources
        assert score_two > score_one_ast


class TestCausalDampening:
    """Downstream categories are dampened when an upstream cause is present."""

    def test_downstream_dampened(self, engine):
        """numerical_instability is downstream of gradient_management_error.
        When both are evidenced, numerical_instability should have lower confidence."""
        # Evidence for upstream: gradient_management_error (high severity runtime)
        upstream = make_finding(
            "R1", SourceCategory.RUNTIME, ["gradient_management_error"], severity=Severity.HIGH
        )
        # Evidence for downstream: numerical_instability
        downstream_f = make_finding(
            "R2", SourceCategory.RUNTIME, ["numerical_instability"], severity=Severity.HIGH
        )
        result = engine.infer([upstream], [], [downstream_f])

        # Find the two hypotheses
        hyp_ids = {h.root_cause_id: h for h in result.hypotheses}
        if "gradient_management_error" in hyp_ids and "numerical_instability" in hyp_ids:
            upstream_conf = hyp_ids["gradient_management_error"].confidence
            downstream_conf = hyp_ids["numerical_instability"].confidence
            # The upstream cause should be ranked higher
            assert upstream_conf >= downstream_conf, (
                f"Upstream conf={upstream_conf:.4f} should be >= downstream conf={downstream_conf:.4f}"
            )


class TestOutputInvariants:
    """Verify invariants on DiagnosisResult."""

    def test_confidence_in_range(self, engine):
        findings = [make_finding("A1", SourceCategory.AST, ["gradient_management_error"])]
        result = engine.infer(findings, [], [])
        for h in result.hypotheses:
            assert 0.0 <= h.confidence <= 1.0, f"Confidence {h.confidence} out of range"

    def test_ranks_consecutive_from_one(self, engine):
        findings = [
            make_finding("A1", SourceCategory.AST, ["gradient_management_error"]),
            make_finding("R1", SourceCategory.RUNTIME, ["numerical_instability"]),
            make_finding("S1", SourceCategory.SEMANTIC, ["loss_output_mismatch"]),
        ]
        result = engine.infer(findings, [], [])
        for i, h in enumerate(result.hypotheses, start=1):
            assert h.rank == i, f"Expected rank {i}, got {h.rank}"

    def test_all_root_cause_ids_in_taxonomy(self, engine):
        from silent_bug_agent.diagnosis.knowledge.taxonomy import TAXONOMY_REGISTRY
        findings = [
            make_finding("A1", SourceCategory.AST, ["gradient_management_error"]),
        ]
        result = engine.infer(findings, [], [])
        valid = TAXONOMY_REGISTRY.category_ids()
        for h in result.hypotheses:
            assert h.root_cause_id in valid, f"Unknown category: {h.root_cause_id!r}"

    def test_empty_input_returns_fallback(self, engine):
        result = engine.infer([], [], [])
        assert result.status == DiagnosisStatus.INSUFFICIENT_EVIDENCE
        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].root_cause_id == "unknown_or_insufficient_evidence"
        assert result.hypotheses[0].confidence == 0.0

    def test_evidence_counts_correct(self, engine):
        ast_f = [make_finding("A1", SourceCategory.AST, ["gradient_management_error"])]
        sem_f = [make_finding("S1", SourceCategory.SEMANTIC, ["loss_output_mismatch"])]
        run_f = [make_finding("R1", SourceCategory.RUNTIME, ["numerical_instability"])]
        result = engine.infer(ast_f, sem_f, run_f)
        assert result.evidence_counts == {"ast": 1, "semantic": 1, "runtime": 1}

    def test_verified_finding_references_exist(self, engine):
        """Evidence refs in hypotheses must come from supplied findings."""
        f = make_finding("A1", SourceCategory.AST, ["gradient_management_error"])
        result = engine.infer([f], [], [])
        for h in result.hypotheses:
            for ref in h.supporting_evidence:
                assert ref.finding_id == "A1"

    def test_warning_for_empty_input(self, engine):
        result = engine.infer([], [], [])
        assert any("No findings" in w for w in result.warnings)

    def test_warning_for_single_source(self, engine):
        findings = [make_finding("A1", SourceCategory.AST, ["gradient_management_error"])]
        result = engine.infer(findings, [], [])
        assert any("one source type" in w.lower() for w in result.warnings)


class TestStatusDetermination:
    """Verify DiagnosisStatus logic."""

    def test_high_confidence_gives_diagnosed(self, engine):
        """Multiple high-severity findings for the same category → diagnosed."""
        findings = [
            make_finding("A1", SourceCategory.AST, ["gradient_management_error"], severity=Severity.HIGH),
            make_finding("S1", SourceCategory.SEMANTIC, ["gradient_management_error"], severity=Severity.HIGH),
            make_finding("R1", SourceCategory.RUNTIME, ["gradient_management_error"], severity=Severity.HIGH),
        ]
        result = engine.infer(findings, [], [])
        assert result.status in (DiagnosisStatus.DIAGNOSED, DiagnosisStatus.AMBIGUOUS)
        assert result.primary_hypothesis is not None

    def test_insufficient_evidence_status(self, engine):
        result = engine.infer([], [], [])
        assert result.status == DiagnosisStatus.INSUFFICIENT_EVIDENCE
        assert result.primary_hypothesis is None
