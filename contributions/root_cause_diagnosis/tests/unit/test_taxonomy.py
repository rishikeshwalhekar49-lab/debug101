"""
Unit tests for the taxonomy knowledge base.

Tests:
- All 15 required categories are present.
- Each entry has required fields populated.
- Registry lookup works correctly.
- Unknown category raises KeyError.
- Causal graph edges are valid (source + target exist, weights in range).
- category_ids() returns the correct set.
- causes() / caused_by() helper methods.
- edge_weight() helper.
"""

import pytest

from silent_bug_agent.knowledge.taxonomy import (
    TAXONOMY_REGISTRY,
    CausalEdge,
    TaxonomyEntry,
    TaxonomyRegistry,
)

# ---------------------------------------------------------------------------
# Required categories (from specification)
# ---------------------------------------------------------------------------

REQUIRED_CATEGORIES = {
    "gradient_management_error",
    "learning_rate_instability",
    "loss_output_mismatch",
    "tensor_shape_mismatch",
    "data_preprocessing_error",
    "label_encoding_error",
    "model_configuration_error",
    "optimizer_configuration_error",
    "training_loop_logic_error",
    "gradient_flow_failure",
    "numerical_instability",
    "evaluation_logic_error",
    "memory_resource_exhaustion",
    "device_placement_error",
    "unknown_or_insufficient_evidence",
}


class TestTaxonomyCompleteness:
    """Verify the taxonomy contains all required categories."""

    def test_all_required_categories_present(self):
        """All 15 specification-required categories must exist."""
        actual = TAXONOMY_REGISTRY.category_ids()
        missing = REQUIRED_CATEGORIES - actual
        assert not missing, f"Missing taxonomy categories: {missing}"

    def test_no_duplicate_category_ids(self):
        """Registry must not contain duplicate IDs."""
        all_entries = TAXONOMY_REGISTRY.get_all()
        assert len(all_entries) == len(set(all_entries.keys()))

    def test_total_category_count(self):
        """Should have at least 15 categories."""
        assert len(TAXONOMY_REGISTRY.category_ids()) >= 15


class TestTaxonomyEntryFields:
    """Verify each taxonomy entry has the required fields populated."""

    @pytest.mark.parametrize("cat_id", sorted(REQUIRED_CATEGORIES))
    def test_entry_has_name(self, cat_id):
        entry = TAXONOMY_REGISTRY.lookup(cat_id)
        assert entry.name and isinstance(entry.name, str)

    @pytest.mark.parametrize("cat_id", sorted(REQUIRED_CATEGORIES))
    def test_entry_has_description(self, cat_id):
        entry = TAXONOMY_REGISTRY.lookup(cat_id)
        assert entry.description and len(entry.description) > 20

    @pytest.mark.parametrize("cat_id", sorted(REQUIRED_CATEGORIES))
    def test_entry_has_verification_checks(self, cat_id):
        """All categories except the fallback must have at least one verification check."""
        entry = TAXONOMY_REGISTRY.lookup(cat_id)
        if cat_id != "unknown_or_insufficient_evidence":
            assert len(entry.verification_checks) >= 1

    @pytest.mark.parametrize("cat_id", sorted(REQUIRED_CATEGORIES - {"unknown_or_insufficient_evidence"}))
    def test_non_fallback_has_indicators(self, cat_id):
        """Non-fallback categories should have at least one indicator type."""
        entry = TAXONOMY_REGISTRY.lookup(cat_id)
        total_indicators = (
            len(entry.typical_static_indicators)
            + len(entry.typical_semantic_indicators)
            + len(entry.typical_runtime_indicators)
        )
        assert total_indicators >= 1, (
            f"{cat_id} has no indicators across static/semantic/runtime"
        )


class TestTaxonomyLookup:
    """Test the registry lookup interface."""

    def test_lookup_valid_category(self):
        entry = TAXONOMY_REGISTRY.lookup("gradient_management_error")
        assert isinstance(entry, TaxonomyEntry)
        assert entry.category_id == "gradient_management_error"

    def test_lookup_fallback_category(self):
        entry = TAXONOMY_REGISTRY.lookup("unknown_or_insufficient_evidence")
        assert entry.category_id == "unknown_or_insufficient_evidence"

    def test_lookup_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="not_a_real_category"):
            TAXONOMY_REGISTRY.lookup("not_a_real_category")

    def test_get_all_returns_copy(self):
        a = TAXONOMY_REGISTRY.get_all()
        a["injected"] = None  # type: ignore
        assert "injected" not in TAXONOMY_REGISTRY.get_all()

    def test_category_ids_is_frozenset(self):
        ids = TAXONOMY_REGISTRY.category_ids()
        assert isinstance(ids, frozenset)


class TestCausalGraph:
    """Verify the causal graph edges are internally consistent."""

    def test_all_edge_sources_exist(self):
        """Every edge source must refer to a valid taxonomy category."""
        valid = TAXONOMY_REGISTRY.category_ids()
        for edge in TAXONOMY_REGISTRY.get_edges():
            assert edge.source in valid, (
                f"Edge source {edge.source!r} not in taxonomy"
            )

    def test_all_edge_targets_exist(self):
        """Every edge target must refer to a valid taxonomy category."""
        valid = TAXONOMY_REGISTRY.category_ids()
        for edge in TAXONOMY_REGISTRY.get_edges():
            assert edge.target in valid, (
                f"Edge target {edge.target!r} not in taxonomy"
            )

    def test_edge_weights_in_range(self):
        """Edge weights must be in (0.0, 1.0]."""
        for edge in TAXONOMY_REGISTRY.get_edges():
            assert 0.0 < edge.weight <= 1.0, (
                f"Edge {edge.source}->{edge.target} has weight {edge.weight}"
            )

    def test_no_self_loops(self):
        """A category cannot cause itself."""
        for edge in TAXONOMY_REGISTRY.get_edges():
            assert edge.source != edge.target, (
                f"Self-loop detected for {edge.source!r}"
            )

    def test_causes_helper(self):
        downstream = TAXONOMY_REGISTRY.causes("gradient_management_error")
        assert "numerical_instability" in downstream
        assert 0.0 < downstream["numerical_instability"] <= 1.0

    def test_caused_by_helper(self):
        upstream = TAXONOMY_REGISTRY.caused_by("numerical_instability")
        assert "gradient_management_error" in upstream or "learning_rate_instability" in upstream

    def test_edge_weight_helper_known_edge(self):
        w = TAXONOMY_REGISTRY.edge_weight("gradient_management_error", "numerical_instability")
        assert w > 0.0

    def test_edge_weight_helper_no_edge(self):
        w = TAXONOMY_REGISTRY.edge_weight("tensor_shape_mismatch", "memory_resource_exhaustion")
        assert w == 0.0
