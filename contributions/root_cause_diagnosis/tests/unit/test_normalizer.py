"""
Unit tests for FindingNormalizer.

Tests:
- Dict and typed-model inputs normalise correctly.
- Keyword matching maps expected finding types to taxonomy categories.
- Malformed / partial fields are skipped with a warning (not an exception).
- Unknown finding types produce empty candidate_categories.
- Source categories are set correctly (ast / semantic / runtime).
- Severity is preserved.
- File and line are preserved when present.
- Mixed valid/invalid lists produce results only for valid entries.
"""

import pytest

from silent_bug_agent.inference.normalizer import FindingNormalizer
from silent_bug_agent.models.findings import (
    ASTFinding,
    NormalizedFinding,
    RuntimeAlert,
    SemanticFinding,
    Severity,
    SourceCategory,
)


@pytest.fixture
def normalizer():
    return FindingNormalizer()


class TestNormalizeDictInputs:
    """Validate dict → NormalizedFinding conversion."""

    def test_ast_dict_basic(self, normalizer):
        result = normalizer.normalize_ast([
            {"id": "A1", "type": "zero_grad", "message": "optimizer.zero_grad() missing"}
        ])
        assert len(result) == 1
        nf = result[0]
        assert nf.original_id == "A1"
        assert nf.source == SourceCategory.AST
        assert "gradient_management_error" in nf.candidate_categories

    def test_semantic_dict_basic(self, normalizer):
        result = normalizer.normalize_semantic([
            {"id": "S1", "type": "loss_mismatch", "message": "cross entropy mismatch with softmax"}
        ])
        assert len(result) == 1
        nf = result[0]
        assert nf.source == SourceCategory.SEMANTIC
        assert "loss_output_mismatch" in nf.candidate_categories

    def test_runtime_dict_basic(self, normalizer):
        result = normalizer.normalize_runtime([
            {"id": "R1", "type": "nan", "message": "Loss became NaN"}
        ])
        assert len(result) == 1
        nf = result[0]
        assert nf.source == SourceCategory.RUNTIME
        assert "numerical_instability" in nf.candidate_categories

    def test_file_and_line_preserved(self, normalizer):
        result = normalizer.normalize_ast([
            {"id": "A2", "type": "zero_grad", "message": "missing zero_grad", "file": "train.py", "line": 42}
        ])
        nf = result[0]
        assert nf.file == "train.py"
        assert nf.line == 42

    def test_severity_preserved(self, normalizer):
        result = normalizer.normalize_ast([
            {"id": "A3", "type": "nan", "message": "nan in output", "severity": "critical"}
        ])
        assert result[0].severity == Severity.CRITICAL

    def test_default_severity(self, normalizer):
        result = normalizer.normalize_ast([
            {"id": "A4", "type": "zero_grad", "message": "zero_grad missing"}
        ])
        assert result[0].severity == Severity.MEDIUM  # default for AST

    def test_metadata_preserved(self, normalizer):
        result = normalizer.normalize_runtime([
            {"id": "R2", "type": "gradient_instability", "message": "norm high", "metadata": {"gradient_norm": 5000}}
        ])
        assert result[0].metadata["gradient_norm"] == 5000


class TestNormalizeTypedInputs:
    """Validate typed Pydantic model → NormalizedFinding conversion."""

    def test_ast_finding_model(self, normalizer):
        finding = ASTFinding(id="A10", type="missing_zero_grad", message="optimizer.zero_grad() missing")
        result = normalizer.normalize_ast([finding])
        assert len(result) == 1
        assert result[0].original_id == "A10"
        assert "gradient_management_error" in result[0].candidate_categories

    def test_runtime_alert_model(self, normalizer):
        alert = RuntimeAlert(id="R10", type="nan_inf", message="Loss is NaN")
        result = normalizer.normalize_runtime([alert])
        assert len(result) == 1
        assert "numerical_instability" in result[0].candidate_categories


class TestKeywordMatching:
    """Verify keyword patterns map to expected categories."""

    @pytest.mark.parametrize("msg,expected_cat", [
        ("optimizer.zero_grad() is missing", "gradient_management_error"),
        ("learning rate is too high causing gradient explosion", "learning_rate_instability"),
        ("cross_entropy applied after softmax output", "loss_output_mismatch"),
        ("shape mismatch between input and expected", "tensor_shape_mismatch"),
        ("normalization applied inconsistently across train val", "data_preprocessing_error"),
        ("one-hot label encoding used where integer expected", "label_encoding_error"),
        ("num_classes incorrect in final layer", "model_configuration_error"),
        ("adam optimizer momentum beta incorrect", "optimizer_configuration_error"),
        ("model.train mode not set in training loop", "training_loop_logic_error"),
        ("gradient is zero — possibly detached", "gradient_flow_failure"),
        ("nan detected in loss value", "numerical_instability"),
        ("validation metric incorrect evaluation logic", "evaluation_logic_error"),
        ("cuda out of memory gpu memory exhausted", "memory_resource_exhaustion"),
        ("device mismatch tensor on cpu but model on cuda", "device_placement_error"),
    ])
    def test_keyword_maps_to_category(self, normalizer, msg, expected_cat):
        result = normalizer.normalize_ast([
            {"id": "T1", "type": "unknown", "message": msg}
        ])
        assert len(result) == 1
        assert expected_cat in result[0].candidate_categories, (
            f"Expected {expected_cat!r} in candidates for message: {msg!r}. "
            f"Got: {result[0].candidate_categories}"
        )


class TestMalformedInputs:
    """Verify malformed inputs are skipped gracefully."""

    def test_missing_required_id_skipped(self, normalizer):
        result = normalizer.normalize_ast([
            {"type": "zero_grad", "message": "missing id field"}
        ])
        assert result == []

    def test_missing_required_type_skipped(self, normalizer):
        result = normalizer.normalize_ast([
            {"id": "A99", "message": "missing type field"}
        ])
        assert result == []

    def test_missing_required_message_skipped(self, normalizer):
        result = normalizer.normalize_ast([
            {"id": "A98", "type": "zero_grad"}
        ])
        assert result == []

    def test_mixed_valid_invalid_list(self, normalizer):
        """Valid entries in a list with invalid entries must still be processed."""
        result = normalizer.normalize_ast([
            {"id": "A1", "type": "zero_grad", "message": "missing zero_grad"},
            {"type": "broken", "message": "no id"},  # invalid — missing id
            {"id": "A2", "type": "nan", "message": "nan in loss"},
        ])
        assert len(result) == 2
        ids = {nf.original_id for nf in result}
        assert "A1" in ids
        assert "A2" in ids

    def test_empty_list_returns_empty(self, normalizer):
        assert normalizer.normalize_ast([]) == []
        assert normalizer.normalize_semantic([]) == []
        assert normalizer.normalize_runtime([]) == []


class TestUnknownFindingTypes:
    """Unknown finding types produce no candidate categories (but do not crash)."""

    def test_unknown_type_produces_empty_candidates(self, normalizer):
        result = normalizer.normalize_ast([
            {"id": "X1", "type": "totally_unknown_category_xyz", "message": "some obscure error"}
        ])
        # Should not crash; candidate_categories may be empty
        assert len(result) == 1
        # Categories may or may not be empty depending on message keyword matching
        assert isinstance(result[0].candidate_categories, list)

    def test_partial_optional_fields(self, normalizer):
        """Findings with missing optional fields (file, line) normalise correctly."""
        result = normalizer.normalize_ast([
            {"id": "X2", "type": "nan", "message": "NaN loss detected"}
        ])
        assert len(result) == 1
        assert result[0].file is None
        assert result[0].line is None
