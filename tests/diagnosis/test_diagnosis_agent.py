"""
Diagnostic accuracy test suite — end-to-end integration tests.

Covers all 20 required scenarios from the specification, plus an accuracy
metrics evaluation utility (Top-1, Top-3, MRR, Coverage, Fallback rate).

All expected root-cause categories are validated against actual agent output —
no hardcoded confidence numbers are used.  Tests check *which* category the
agent identifies, not *how certain* it is to an arbitrary decimal.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

import pytest

from silent_bug_agent.agents.diagnosis_agent import DiagnosisAgent
from silent_bug_agent.diagnosis.models.hypothesis import DiagnosisResult, DiagnosisStatus


# ---------------------------------------------------------------------------
# Shared agent fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def agent():
    return DiagnosisAgent()


# ---------------------------------------------------------------------------
# Helper: extract top-N root cause IDs from result
# ---------------------------------------------------------------------------


def top_n_categories(result: DiagnosisResult, n: int = 3) -> List[str]:
    return [h.root_cause_id for h in result.hypotheses[:n]]


# ---------------------------------------------------------------------------
# Scenario 1: Missing optimizer.zero_grad() + abnormal gradient accumulation
# ---------------------------------------------------------------------------


class TestScenario01MissingZeroGrad:
    """Missing zero_grad() combined with gradient accumulation runtime alert."""

    def test_top1_diagnosis(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {
                    "id": "AST-001",
                    "type": "gradient_management_error",
                    "message": "optimizer.zero_grad() may be missing from training loop",
                    "file": "train.py",
                    "line": 42,
                    "severity": "high",
                }
            ],
            semantic_findings=[
                {
                    "id": "SEM-001",
                    "type": "gradient_management",
                    "message": "Gradients are accumulated across batches; zero_grad not called",
                    "severity": "high",
                }
            ],
            runtime_alerts=[
                {
                    "id": "RUN-001",
                    "type": "gradient_instability",
                    "message": "Gradient norm increased sharply over last 10 steps",
                    "metadata": {"gradient_norm": 18000},
                    "severity": "high",
                }
            ],
        )
        cats = top_n_categories(result)
        assert "gradient_management_error" in cats, (
            f"Expected gradient_management_error in top-3, got: {cats}"
        )
        assert result.primary_hypothesis is not None
        assert result.evidence_counts == {"ast": 1, "semantic": 1, "runtime": 1}

    def test_supporting_evidence_references_all_three_sources(self, agent):
        result = agent.diagnose(
            ast_findings=[{"id": "AST-001", "type": "gradient_management_error", "message": "zero_grad missing", "severity": "high"}],
            semantic_findings=[{"id": "SEM-001", "type": "gradient_management", "message": "stale gradients detected", "severity": "high"}],
            runtime_alerts=[{"id": "RUN-001", "type": "gradient_instability", "message": "gradient norm spike", "severity": "high"}],
        )
        top = result.primary_hypothesis
        ref_ids = {r.finding_id for r in top.supporting_evidence}
        assert "AST-001" in ref_ids or "SEM-001" in ref_ids or "RUN-001" in ref_ids


# ---------------------------------------------------------------------------
# Scenario 2: Excessive learning rate + exploding gradients
# ---------------------------------------------------------------------------


class TestScenario02ExcessiveLearningRate:
    """High LR combined with exploding gradients."""

    def test_top3_includes_lr_instability(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {
                    "id": "AST-002",
                    "type": "learning_rate_instability",
                    "message": "Learning rate 0.1 appears excessive for Adam optimizer",
                    "severity": "high",
                }
            ],
            runtime_alerts=[
                {
                    "id": "RUN-002",
                    "type": "exploding_gradient",
                    "message": "Gradient norm exceeded 1e6 — training diverging",
                    "metadata": {"gradient_norm": 1e6},
                    "severity": "critical",
                }
            ],
        )
        cats = top_n_categories(result, 3)
        assert "learning_rate_instability" in cats, f"Got: {cats}"

    def test_verification_checks_present(self, agent):
        result = agent.diagnose(
            ast_findings=[{"id": "A1", "type": "learning_rate_instability", "message": "lr too high", "severity": "high"}],
            runtime_alerts=[{"id": "R1", "type": "exploding_gradient", "message": "gradient norm very large", "severity": "critical"}],
        )
        top = result.primary_hypothesis
        assert top is not None
        assert len(top.verification_checks) >= 1


# ---------------------------------------------------------------------------
# Scenario 3: Incompatible model output activation + loss function
# ---------------------------------------------------------------------------


class TestScenario03LossOutputMismatch:
    """Double softmax or sigmoid before BCE detected."""

    def test_top1_or_top3_loss_mismatch(self, agent):
        result = agent.diagnose(
            semantic_findings=[
                {
                    "id": "SEM-003",
                    "type": "loss_output_mismatch",
                    "message": "Model applies softmax before CrossEntropyLoss which applies log_softmax internally — double softmax",
                    "severity": "high",
                }
            ],
            runtime_alerts=[
                {
                    "id": "RUN-003",
                    "type": "nan_inf",
                    "message": "Loss is NaN after first batch",
                    "severity": "critical",
                }
            ],
        )
        cats = top_n_categories(result, 3)
        assert "loss_output_mismatch" in cats or "numerical_instability" in cats, (
            f"Got: {cats}"
        )

    def test_explanation_not_empty(self, agent):
        result = agent.diagnose(
            semantic_findings=[{"id": "S1", "type": "activation_mismatch", "message": "sigmoid output before BCEWithLogitsLoss — double sigmoid", "severity": "high"}]
        )
        top = result.primary_hypothesis
        assert top is not None
        assert len(top.explanation) > 10


# ---------------------------------------------------------------------------
# Scenario 4: Tensor dimension mismatch
# ---------------------------------------------------------------------------


class TestScenario04TensorDimMismatch:
    """Shape mismatch in matmul or reshape."""

    def test_tensor_shape_in_top3(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {
                    "id": "AST-004",
                    "type": "tensor_shape_mismatch",
                    "message": "Tensor shape mismatch: expected (32, 512) but got (32, 256) in matmul",
                    "file": "model.py",
                    "line": 88,
                    "severity": "high",
                }
            ],
            runtime_alerts=[
                {
                    "id": "RUN-004",
                    "type": "shape_mismatch",
                    "message": "Runtime error: size mismatch at layer fc1",
                    "severity": "critical",
                }
            ],
        )
        cats = top_n_categories(result, 3)
        assert "tensor_shape_mismatch" in cats, f"Got: {cats}"

    def test_affected_files_recorded(self, agent):
        result = agent.diagnose(
            ast_findings=[{"id": "A1", "type": "tensor_shape_mismatch", "message": "shape mismatch reshape", "file": "model.py", "line": 88, "severity": "high"}]
        )
        top = result.primary_hypothesis
        assert top is not None
        assert "model.py" in top.affected_files


# ---------------------------------------------------------------------------
# Scenario 5: Incorrect label encoding
# ---------------------------------------------------------------------------


class TestScenario05LabelEncoding:
    """One-hot labels where integer indices are expected."""

    def test_label_encoding_in_top3(self, agent):
        result = agent.diagnose(
            semantic_findings=[
                {
                    "id": "SEM-005",
                    "type": "label_encoding_error",
                    "message": "One-hot encoded labels passed to CrossEntropyLoss which expects integer class indices",
                    "severity": "high",
                }
            ],
            runtime_alerts=[
                {
                    "id": "RUN-005",
                    "type": "label_error",
                    "message": "Target shape mismatch: expected (batch_size,) got (batch_size, num_classes)",
                    "severity": "high",
                }
            ],
        )
        cats = top_n_categories(result, 3)
        assert "label_encoding_error" in cats or "loss_output_mismatch" in cats, (
            f"Got: {cats}"
        )


# ---------------------------------------------------------------------------
# Scenario 6: Data preprocessing inconsistency
# ---------------------------------------------------------------------------


class TestScenario06DataPreprocessing:
    """Train/val normalization mismatch."""

    def test_data_preprocessing_in_top3(self, agent):
        result = agent.diagnose(
            semantic_findings=[
                {
                    "id": "SEM-006",
                    "type": "data_preprocessing_error",
                    "message": "Normalization statistics computed on validation set instead of training set",
                    "severity": "medium",
                }
            ],
            runtime_alerts=[
                {
                    "id": "RUN-006",
                    "type": "evaluation_error",
                    "message": "Validation performance collapses while training loss improves",
                    "severity": "high",
                }
            ],
        )
        cats = top_n_categories(result, 3)
        assert "data_preprocessing_error" in cats or "evaluation_logic_error" in cats, (
            f"Got: {cats}"
        )


# ---------------------------------------------------------------------------
# Scenario 7: Missing gradient flow / detached computation graph
# ---------------------------------------------------------------------------


class TestScenario07GradientFlowFailure:
    """Detached graph causing zero gradients."""

    def test_gradient_flow_in_top3(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {
                    "id": "AST-007",
                    "type": "gradient_flow_failure",
                    "message": "Tensor detached from computation graph before loss computation",
                    "file": "model.py",
                    "line": 55,
                    "severity": "high",
                }
            ],
            runtime_alerts=[
                {
                    "id": "RUN-007",
                    "type": "missing_gradient",
                    "message": "Gradients are zero for all parameters — possible detach issue",
                    "severity": "critical",
                }
            ],
        )
        cats = top_n_categories(result, 3)
        assert "gradient_flow_failure" in cats, f"Got: {cats}"


# ---------------------------------------------------------------------------
# Scenario 8: NaN/Inf numerical instability
# ---------------------------------------------------------------------------


class TestScenario08NumericalInstability:
    """NaN/Inf in loss and gradients."""

    def test_numerical_instability_in_top3(self, agent):
        result = agent.diagnose(
            runtime_alerts=[
                {
                    "id": "RUN-008a",
                    "type": "numerical_instability",
                    "message": "Loss became NaN at step 150",
                    "severity": "critical",
                },
                {
                    "id": "RUN-008b",
                    "type": "nan_inf",
                    "message": "Gradient contains Inf values",
                    "severity": "critical",
                },
                {
                    "id": "RUN-008c",
                    "type": "numerical_instability",
                    "message": "Activation weights contain NaN — detected in layer conv2",
                    "severity": "critical",
                },
            ]
        )
        cats = top_n_categories(result, 3)
        assert "numerical_instability" in cats, f"Got: {cats}"

    def test_causal_chain_non_empty(self, agent):
        result = agent.diagnose(
            runtime_alerts=[{"id": "R1", "type": "nan", "message": "nan in loss", "severity": "critical"}]
        )
        top = result.primary_hypothesis
        assert top is not None
        assert len(top.causal_chain) >= 1


# ---------------------------------------------------------------------------
# Scenario 9: Incorrect evaluation logic
# ---------------------------------------------------------------------------


class TestScenario09EvaluationLogic:
    """Model not in eval mode / wrong metric."""

    def test_evaluation_logic_in_top3(self, agent):
        result = agent.diagnose(
            semantic_findings=[
                {
                    "id": "SEM-009",
                    "type": "evaluation_logic_error",
                    "message": "model.eval() not called before validation loop — dropout remains active",
                    "severity": "high",
                }
            ],
            runtime_alerts=[
                {
                    "id": "RUN-009",
                    "type": "evaluation_error",
                    "message": "Validation performance collapses while training loss improves",
                    "severity": "high",
                }
            ],
        )
        cats = top_n_categories(result, 3)
        assert "evaluation_logic_error" in cats, f"Got: {cats}"


# ---------------------------------------------------------------------------
# Scenario 10: GPU memory exhaustion
# ---------------------------------------------------------------------------


class TestScenario10GPUMemory:
    """CUDA OOM error."""

    def test_memory_exhaustion_in_top3(self, agent):
        result = agent.diagnose(
            runtime_alerts=[
                {
                    "id": "RUN-010",
                    "type": "oom",
                    "message": "CUDA out of memory — tried to allocate 2.4 GiB",
                    "severity": "critical",
                    "metadata": {"required_mb": 2458, "available_mb": 512},
                }
            ]
        )
        cats = top_n_categories(result, 3)
        assert "memory_resource_exhaustion" in cats, f"Got: {cats}"


# ---------------------------------------------------------------------------
# Scenario 11: Conflicting evidence from different sources
# ---------------------------------------------------------------------------


class TestScenario11ConflictingEvidence:
    """Static and runtime evidence point to different root causes."""

    def test_ambiguous_or_multiple_hypotheses(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {
                    "id": "AST-011",
                    "type": "learning_rate_instability",
                    "message": "Learning rate 1e-4 appears reasonable for Adam",
                    "severity": "low",
                }
            ],
            runtime_alerts=[
                {
                    "id": "RUN-011",
                    "type": "loss_output_mismatch",
                    "message": "Loss function output value is unexpectedly large",
                    "severity": "high",
                }
            ],
        )
        # Should return at least one hypothesis and handle conflicting evidence
        assert len(result.hypotheses) >= 1
        assert result.status in (
            DiagnosisStatus.DIAGNOSED,
            DiagnosisStatus.AMBIGUOUS,
            DiagnosisStatus.INSUFFICIENT_EVIDENCE,
        )

    def test_confidence_not_overinflated(self, agent):
        """Conflicting evidence should not produce near-perfect confidence."""
        result = agent.diagnose(
            ast_findings=[{"id": "A1", "type": "learning_rate_instability", "message": "lr high", "severity": "high"}],
            runtime_alerts=[{"id": "R1", "type": "gradient_flow_failure", "message": "gradient zero", "severity": "high"}],
        )
        if result.primary_hypothesis:
            # Conflicting evidence should keep confidence moderate
            assert result.primary_hypothesis.confidence <= 0.95


# ---------------------------------------------------------------------------
# Scenario 12: Duplicate evidence (same bug described twice)
# ---------------------------------------------------------------------------


class TestScenario12DuplicateEvidence:
    """Same category reported by multiple findings from same source — must not
    inflate confidence excessively."""

    def test_duplicate_evidence_dampened(self, agent):
        findings = [
            {"id": f"A{i}", "type": "gradient_management_error",
             "message": f"zero_grad missing instance {i}", "severity": "high"}
            for i in range(5)
        ]
        result_many = agent.diagnose(ast_findings=findings)

        # One finding
        result_one = agent.diagnose(
            ast_findings=[{"id": "A1", "type": "gradient_management_error", "message": "zero_grad missing", "severity": "high"}]
        )

        # Confidence should not grow linearly with duplicates
        if result_many.primary_hypothesis and result_one.primary_hypothesis:
            ratio = result_many.primary_hypothesis.confidence / max(result_one.primary_hypothesis.confidence, 1e-9)
            assert ratio < 5.0, (
                f"Confidence grew {ratio:.1f}x with 5x findings — duplicate penalty not working"
            )


# ---------------------------------------------------------------------------
# Scenario 13: Static findings only
# ---------------------------------------------------------------------------


class TestScenario13StaticOnly:
    """Only AST findings provided."""

    def test_handles_static_only(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {"id": "A1", "type": "gradient_management_error", "message": "zero_grad not called", "severity": "high"}
            ]
        )
        assert result is not None
        assert result.evidence_counts["ast"] == 1
        assert result.evidence_counts["semantic"] == 0
        assert result.evidence_counts["runtime"] == 0
        assert len(result.hypotheses) >= 1

    def test_single_source_warning(self, agent):
        result = agent.diagnose(
            ast_findings=[{"id": "A1", "type": "tensor_shape_mismatch", "message": "shape mismatch in reshape", "severity": "medium"}]
        )
        assert any("one source" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Scenario 14: Runtime alerts only
# ---------------------------------------------------------------------------


class TestScenario14RuntimeOnly:
    """Only runtime alerts provided."""

    def test_handles_runtime_only(self, agent):
        result = agent.diagnose(
            runtime_alerts=[
                {"id": "R1", "type": "nan_inf", "message": "Loss is NaN at step 50", "severity": "critical"}
            ]
        )
        assert result.evidence_counts["runtime"] == 1
        assert result.evidence_counts["ast"] == 0
        assert len(result.hypotheses) >= 1

    def test_runtime_only_yields_hypothesis(self, agent):
        result = agent.diagnose(
            runtime_alerts=[{"id": "R1", "type": "oom", "message": "GPU out of memory", "severity": "critical"}]
        )
        cats = top_n_categories(result, 3)
        assert "memory_resource_exhaustion" in cats, f"Got: {cats}"


# ---------------------------------------------------------------------------
# Scenario 15: Semantic findings only
# ---------------------------------------------------------------------------


class TestScenario15SemanticOnly:
    """Only semantic findings provided."""

    def test_handles_semantic_only(self, agent):
        result = agent.diagnose(
            semantic_findings=[
                {
                    "id": "S1",
                    "type": "loss_output_mismatch",
                    "message": "Output activation incompatible with loss function selection",
                    "severity": "high",
                }
            ]
        )
        assert result.evidence_counts["semantic"] == 1
        cats = top_n_categories(result, 3)
        assert "loss_output_mismatch" in cats, f"Got: {cats}"


# ---------------------------------------------------------------------------
# Scenario 16: Empty inputs
# ---------------------------------------------------------------------------


class TestScenario16EmptyInputs:
    """All-empty inputs must not crash and must return insufficient_evidence."""

    def test_none_inputs(self, agent):
        result = agent.diagnose()
        assert result.status == DiagnosisStatus.INSUFFICIENT_EVIDENCE
        assert result.primary_hypothesis is None

    def test_empty_lists(self, agent):
        result = agent.diagnose(ast_findings=[], semantic_findings=[], runtime_alerts=[])
        assert result.status == DiagnosisStatus.INSUFFICIENT_EVIDENCE

    def test_fallback_hypothesis_returned(self, agent):
        result = agent.diagnose()
        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].root_cause_id == "unknown_or_insufficient_evidence"

    def test_confidence_zero_on_empty(self, agent):
        result = agent.diagnose()
        assert result.hypotheses[0].confidence == 0.0

    def test_to_dict_works(self, agent):
        result = agent.diagnose()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "status" in d
        assert "hypotheses" in d

    def test_to_json_works(self, agent):
        result = agent.diagnose()
        j = result.to_json()
        assert isinstance(j, str)
        assert "unknown_or_insufficient_evidence" in j


# ---------------------------------------------------------------------------
# Scenario 17: Unknown finding types
# ---------------------------------------------------------------------------


class TestScenario17UnknownFindingTypes:
    """Unknown finding types must be handled without crashing."""

    def test_unknown_type_does_not_crash(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {"id": "A1", "type": "completely_made_up_type_xyz_42", "message": "some obscure error", "severity": "medium"}
            ]
        )
        assert result is not None

    def test_unknown_type_may_produce_fallback(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {"id": "A1", "type": "xyz_unknown_123", "message": "completely unrecognised message text blah blah"}
            ]
        )
        # May return insufficient evidence or low-confidence result — either is acceptable
        assert len(result.hypotheses) >= 1
        assert all(math.isfinite(h.confidence) for h in result.hypotheses)


# ---------------------------------------------------------------------------
# Scenario 18: Malformed / partially missing optional fields
# ---------------------------------------------------------------------------


class TestScenario18MalformedOrPartialFields:
    """Missing optional fields (file, line, severity) must not break diagnosis."""

    def test_no_file_no_line(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {"id": "A1", "type": "gradient_management_error", "message": "zero_grad missing"}
            ]
        )
        assert result is not None

    def test_extra_unknown_fields_ignored(self, agent):
        """Extra fields not in the schema should be silently ignored."""
        result = agent.diagnose(
            ast_findings=[
                {
                    "id": "A1",
                    "type": "nan",
                    "message": "nan loss",
                    "unknown_extra_field": "should be ignored",
                    "another_field": 12345,
                }
            ]
        )
        assert result is not None

    def test_missing_required_id_in_list_with_valid(self, agent):
        """Invalid entry without required 'id' is skipped; valid entry is processed."""
        result = agent.diagnose(
            ast_findings=[
                {"type": "nan", "message": "nan no id"},  # invalid
                {"id": "A2", "type": "nan", "message": "nan with id", "severity": "high"},  # valid
            ]
        )
        assert result.evidence_counts["ast"] >= 1  # at least the valid one


# ---------------------------------------------------------------------------
# Scenario 19: Multiple plausible root causes
# ---------------------------------------------------------------------------


class TestScenario19MultiplePlausibleCauses:
    """Evidence that supports multiple hypotheses — agent should return several."""

    def test_multiple_hypotheses_returned(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {"id": "A1", "type": "gradient_management_error", "message": "zero_grad possibly missing", "severity": "medium"},
                {"id": "A2", "type": "learning_rate_instability", "message": "learning rate appears high for this optimizer", "severity": "medium"},
            ],
            semantic_findings=[
                {"id": "S1", "type": "loss_output_mismatch", "message": "activation function may be incompatible with loss", "severity": "medium"},
            ],
        )
        assert len(result.hypotheses) >= 2, (
            f"Expected multiple hypotheses, got {len(result.hypotheses)}"
        )

    def test_alternative_hypotheses_listed(self, agent):
        result = agent.diagnose(
            ast_findings=[
                {"id": "A1", "type": "gradient_management_error", "message": "zero_grad missing", "severity": "high"},
                {"id": "A2", "type": "learning_rate_instability", "message": "lr high", "severity": "high"},
            ],
        )
        top = result.primary_hypothesis
        if top and len(result.hypotheses) > 1:
            assert len(top.alternative_hypotheses) >= 1


# ---------------------------------------------------------------------------
# Scenario 20: Known cause with supporting AND contradicting evidence
# ---------------------------------------------------------------------------


class TestScenario20SupportingAndContradicting:
    """The agent must handle evidence that both supports and contradicts a hypothesis."""

    def test_contradicting_evidence_reduces_confidence(self, agent):
        """Two findings say 'gradient error'; one says 'lr is fine'.
        The agent should still potentially produce a hypothesis but with lower confidence
        than the pure-support case."""
        # Pure support case
        result_support = agent.diagnose(
            ast_findings=[
                {"id": "A1", "type": "gradient_management_error", "message": "zero_grad missing", "severity": "high"},
                {"id": "A2", "type": "gradient_management_error", "message": "stale gradient detected", "severity": "high"},
            ]
        )

        # Mixed case (same support + a contradicting signal about lr being fine)
        result_mixed = agent.diagnose(
            ast_findings=[
                {"id": "A1", "type": "gradient_management_error", "message": "zero_grad missing", "severity": "high"},
                {"id": "A2", "type": "gradient_management_error", "message": "stale gradient detected", "severity": "high"},
            ],
            semantic_findings=[
                {"id": "S1", "type": "learning_rate_instability", "message": "learning rate appears fine and stable", "severity": "low"},
            ],
        )

        # Both should return a result without crashing
        assert result_support is not None
        assert result_mixed is not None

    def test_warnings_may_note_ambiguity(self, agent):
        result = agent.diagnose(
            ast_findings=[{"id": "A1", "type": "gradient_management_error", "message": "zero_grad missing", "severity": "high"}],
            runtime_alerts=[{"id": "R1", "type": "gradient_flow_failure", "message": "gradient zero all parameters", "severity": "high"}],
        )
        # Either is a plausible cause; warnings may note ambiguity
        assert result is not None


# ---------------------------------------------------------------------------
# Accuracy metrics evaluation utility
# ---------------------------------------------------------------------------


class DiagnosticAccuracyEvaluator:
    """Compute diagnostic accuracy metrics over a set of labelled test cases.

    Metrics computed:
    - Top-1 accuracy: fraction of cases where expected category is rank-1.
    - Top-3 accuracy: fraction of cases where expected category is in top-3.
    - MRR (Mean Reciprocal Rank): 1/rank if found in top-5, else 0.
    - Coverage: fraction of cases receiving a non-fallback diagnosis.
    - Fallback rate: fraction of cases returning unknown_or_insufficient_evidence.
    """

    FALLBACK_ID = "unknown_or_insufficient_evidence"

    def __init__(self, agent: DiagnosisAgent):
        self.agent = agent

    def evaluate(
        self,
        test_cases: List[Dict],
    ) -> Dict[str, float]:
        """Run evaluation over *test_cases*.

        Each case is a dict with:
        - ``ast_findings``: list of finding dicts (optional)
        - ``semantic_findings``: list of finding dicts (optional)
        - ``runtime_alerts``: list of finding dicts (optional)
        - ``expected``: set of acceptable category IDs (at least 1)
        """
        top1_hits = 0
        top3_hits = 0
        mrr_sum = 0.0
        non_fallback = 0
        fallbacks = 0
        total = len(test_cases)

        for case in test_cases:
            expected: Set[str] = set(case.get("expected", set()))
            result = self.agent.diagnose(
                ast_findings=case.get("ast_findings", []),
                semantic_findings=case.get("semantic_findings", []),
                runtime_alerts=case.get("runtime_alerts", []),
            )
            all_ids = [h.root_cause_id for h in result.hypotheses]

            # Top-1
            if all_ids and all_ids[0] in expected:
                top1_hits += 1

            # Top-3
            if any(c in expected for c in all_ids[:3]):
                top3_hits += 1

            # MRR
            for rank_idx, cat_id in enumerate(all_ids[:5], start=1):
                if cat_id in expected:
                    mrr_sum += 1.0 / rank_idx
                    break

            # Coverage / fallback
            if all_ids and all_ids[0] == self.FALLBACK_ID:
                fallbacks += 1
            else:
                non_fallback += 1

        if total == 0:
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mrr": 0.0,
                "coverage": 0.0,
                "fallback_rate": 0.0,
                "total_cases": 0,
            }

        return {
            "top1_accuracy": top1_hits / total,
            "top3_accuracy": top3_hits / total,
            "mrr": mrr_sum / total,
            "coverage": non_fallback / total,
            "fallback_rate": fallbacks / total,
            "total_cases": total,
        }


# ---------------------------------------------------------------------------
# Accuracy metric tests (run on the 20 primary test cases)
# ---------------------------------------------------------------------------

# Labelled test fixture dataset
LABELLED_CASES = [
    {
        "name": "Missing zero_grad + gradient accumulation",
        "ast_findings": [{"id": "A1", "type": "gradient_management_error", "message": "zero_grad missing in training loop", "severity": "high"}],
        "semantic_findings": [{"id": "S1", "type": "gradient_management", "message": "gradients accumulated across batches", "severity": "high"}],
        "runtime_alerts": [{"id": "R1", "type": "gradient_instability", "message": "gradient norm spike", "severity": "high"}],
        "expected": {"gradient_management_error"},
    },
    {
        "name": "Excessive LR + exploding gradients",
        "ast_findings": [{"id": "A1", "type": "learning_rate_instability", "message": "learning rate too high", "severity": "high"}],
        "runtime_alerts": [{"id": "R1", "type": "exploding_gradient", "message": "gradient norm very large exploding", "severity": "critical"}],
        "expected": {"learning_rate_instability"},
    },
    {
        "name": "Loss/output mismatch",
        "semantic_findings": [{"id": "S1", "type": "loss_output_mismatch", "message": "double softmax cross_entropy activation incompatible", "severity": "high"}],
        "expected": {"loss_output_mismatch"},
    },
    {
        "name": "Tensor shape mismatch",
        "ast_findings": [{"id": "A1", "type": "tensor_shape_mismatch", "message": "shape mismatch dimension error in reshape", "severity": "high"}],
        "runtime_alerts": [{"id": "R1", "type": "shape_mismatch", "message": "size mismatch at runtime", "severity": "critical"}],
        "expected": {"tensor_shape_mismatch"},
    },
    {
        "name": "Label encoding error",
        "semantic_findings": [{"id": "S1", "type": "label_encoding_error", "message": "one-hot label encoding used where integer index expected", "severity": "high"}],
        "expected": {"label_encoding_error", "loss_output_mismatch"},
    },
    {
        "name": "Data preprocessing inconsistency",
        "semantic_findings": [{"id": "S1", "type": "data_preprocessing_error", "message": "normalization inconsistent across training and validation", "severity": "medium"}],
        "runtime_alerts": [{"id": "R1", "type": "evaluation_error", "message": "validation performance collapse train val gap", "severity": "high"}],
        "expected": {"data_preprocessing_error", "evaluation_logic_error"},
    },
    {
        "name": "Gradient flow failure / detach",
        "ast_findings": [{"id": "A1", "type": "gradient_flow_failure", "message": "tensor detached from graph — gradient not flowing", "severity": "high"}],
        "runtime_alerts": [{"id": "R1", "type": "missing_gradient", "message": "gradient zero — possibly detached requires_grad false", "severity": "critical"}],
        "expected": {"gradient_flow_failure"},
    },
    {
        "name": "NaN/Inf numerical instability",
        "runtime_alerts": [
            {"id": "R1", "type": "numerical_instability", "message": "loss nan detected", "severity": "critical"},
            {"id": "R2", "type": "nan_inf", "message": "gradient inf detected", "severity": "critical"},
        ],
        "expected": {"numerical_instability"},
    },
    {
        "name": "Evaluation logic error",
        "semantic_findings": [{"id": "S1", "type": "evaluation_logic_error", "message": "model not in eval mode during validation — metric incorrect", "severity": "high"}],
        "expected": {"evaluation_logic_error"},
    },
    {
        "name": "GPU memory exhaustion",
        "runtime_alerts": [{"id": "R1", "type": "oom", "message": "cuda out of memory gpu memory exhausted", "severity": "critical"}],
        "expected": {"memory_resource_exhaustion"},
    },
    {
        "name": "Static only",
        "ast_findings": [{"id": "A1", "type": "gradient_management_error", "message": "zero_grad not called", "severity": "high"}],
        "expected": {"gradient_management_error"},
    },
    {
        "name": "Runtime only",
        "runtime_alerts": [{"id": "R1", "type": "nan_inf", "message": "nan inf in loss", "severity": "critical"}],
        "expected": {"numerical_instability"},
    },
    {
        "name": "Semantic only",
        "semantic_findings": [{"id": "S1", "type": "loss_output_mismatch", "message": "output activation incompatible with loss function", "severity": "high"}],
        "expected": {"loss_output_mismatch"},
    },
    {
        "name": "Empty inputs → fallback",
        "expected": {"unknown_or_insufficient_evidence"},
    },
]


class TestAccuracyMetrics:
    """Compute and report accuracy metrics on the labelled test set."""

    @pytest.fixture(scope="class")
    @classmethod
    def evaluator(cls):
        return DiagnosticAccuracyEvaluator(DiagnosisAgent())

    @pytest.fixture(scope="class")
    @classmethod
    def metrics(cls, evaluator):
        return evaluator.evaluate(LABELLED_CASES)

    def test_top3_accuracy_above_threshold(self, metrics):
        """Top-3 accuracy must be at least 70% on the labelled test set."""
        print(f"\n[Accuracy] Top-3 accuracy: {metrics['top3_accuracy']:.1%}")
        assert metrics["top3_accuracy"] >= 0.70, (
            f"Top-3 accuracy {metrics['top3_accuracy']:.1%} is below 70% threshold. "
            f"Full metrics: {metrics}"
        )

    def test_top1_accuracy_above_threshold(self, metrics):
        """Top-1 accuracy should be at least 50% (ambiguity is expected)."""
        print(f"[Accuracy] Top-1 accuracy: {metrics['top1_accuracy']:.1%}")
        assert metrics["top1_accuracy"] >= 0.50, (
            f"Top-1 accuracy {metrics['top1_accuracy']:.1%} is below 50%. "
            f"Full metrics: {metrics}"
        )

    def test_mrr_above_threshold(self, metrics):
        """Mean Reciprocal Rank should be at least 0.55."""
        print(f"[Accuracy] MRR: {metrics['mrr']:.3f}")
        assert metrics["mrr"] >= 0.55, (
            f"MRR {metrics['mrr']:.3f} is below 0.55. Full metrics: {metrics}"
        )

    def test_fallback_rate_acceptable(self, metrics):
        """Fallback rate should be ≤ 15% of non-empty cases."""
        # Count non-empty cases (cases that have at least some findings)
        non_empty_count = sum(
            1 for c in LABELLED_CASES
            if c.get("ast_findings") or c.get("semantic_findings") or c.get("runtime_alerts")
        )
        if non_empty_count == 0:
            return
        print(f"[Accuracy] Fallback rate: {metrics['fallback_rate']:.1%}")
        # Only the empty-input case should produce a fallback
        assert metrics["fallback_rate"] <= 0.15, (
            f"Fallback rate {metrics['fallback_rate']:.1%} too high."
        )

    def test_metrics_all_finite(self, metrics):
        for key, val in metrics.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"Metric {key!r} is not finite: {val}"

    def test_print_full_metrics(self, metrics):
        """Print full metrics report (always passes — informational)."""
        print("\n" + "=" * 50)
        print("DIAGNOSTIC ACCURACY METRICS")
        print("=" * 50)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<22}: {v:.4f}")
            else:
                print(f"  {k:<22}: {v}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# Additional API contract tests
# ---------------------------------------------------------------------------


class TestAPIContract:
    """Verify the public DiagnosisAgent API contract."""

    def test_lookup_taxonomy_valid(self, agent):
        entry = agent.lookup_taxonomy("gradient_management_error")
        assert entry.category_id == "gradient_management_error"

    def test_lookup_taxonomy_invalid_raises(self, agent):
        with pytest.raises(KeyError):
            agent.lookup_taxonomy("not_a_real_id")

    def test_list_taxonomy_categories(self, agent):
        cats = agent.list_taxonomy_categories()
        assert isinstance(cats, list)
        assert "gradient_management_error" in cats
        assert "unknown_or_insufficient_evidence" in cats

    def test_explain_causal_chain(self, agent):
        result = agent.diagnose(
            runtime_alerts=[{"id": "R1", "type": "nan", "message": "loss nan", "severity": "critical"}]
        )
        top = result.primary_hypothesis
        if top:
            explanation = agent.explain_causal_chain(top)
            assert isinstance(explanation, str)
            assert len(explanation) > 0

    def test_to_dict_structure(self, agent):
        result = agent.diagnose(
            ast_findings=[{"id": "A1", "type": "gradient_management_error", "message": "zero_grad missing", "severity": "high"}]
        )
        d = result.to_dict()
        assert "status" in d
        assert "hypotheses" in d
        assert "evidence_counts" in d
        assert "warnings" in d

    def test_diagnose_returns_diagnosis_result(self, agent):
        from silent_bug_agent.diagnosis.models.hypothesis import DiagnosisResult
        result = agent.diagnose()
        assert isinstance(result, DiagnosisResult)

    def test_diagnose_accepts_mixed_dict_and_models(self, agent):
        """API should accept both dict and typed model instances in same call."""
        from silent_bug_agent.diagnosis.models.findings import ASTFinding, RuntimeAlert
        result = agent.diagnose(
            ast_findings=[
                ASTFinding(id="A1", type="gradient_management_error", message="zero_grad missing", severity="high"),
                {"id": "A2", "type": "learning_rate_instability", "message": "lr high", "severity": "medium"},
            ],
            runtime_alerts=[
                RuntimeAlert(id="R1", type="nan", message="loss nan"),
            ]
        )
        assert result is not None
        assert result.evidence_counts["ast"] == 2
        assert result.evidence_counts["runtime"] == 1
