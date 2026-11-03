"""
Finding Normalizer — converts heterogeneous finding dicts or Pydantic models
into a common :class:`~silent_bug_agent.diagnosis.models.findings.NormalizedFinding`
representation used by the causal inference engine.

The normaliser performs two tasks:

1. **Type coercion** — accepts raw ``dict`` inputs or typed Pydantic models.
2. **Keyword matching** — maps the finding's ``type`` field and ``message``
   text to candidate taxonomy category IDs using a deterministic keyword table.

Keyword matching is conservative: multiple categories may be returned when a
finding is ambiguous.  The causal engine then resolves them using context.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Sequence, Tuple, Union

from silent_bug_agent.diagnosis.models.findings import (
    ASTFinding,
    NormalizedFinding,
    RuntimeAlert,
    SemanticFinding,
    Severity,
    SourceCategory,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword → taxonomy category mapping
#
# Each entry is a tuple of (keywords_tuple, category_id).
# Keywords are checked against a lower-cased combined string of:
#   finding.type  +  " "  +  finding.message
#
# Order matters: more specific patterns should appear first.
# ---------------------------------------------------------------------------

# Type-field aliases that directly map to taxonomy IDs (exact / prefix match)
_TYPE_ALIASES: Dict[str, str] = {
    "gradient_management_error": "gradient_management_error",
    "gradient_management": "gradient_management_error",
    "zero_grad": "gradient_management_error",
    "missing_zero_grad": "gradient_management_error",
    "learning_rate_instability": "learning_rate_instability",
    "learning_rate": "learning_rate_instability",
    "exploding_gradient": "learning_rate_instability",
    "gradient_instability": "learning_rate_instability",
    "loss_output_mismatch": "loss_output_mismatch",
    "activation_mismatch": "loss_output_mismatch",
    "loss_mismatch": "loss_output_mismatch",
    "tensor_shape_mismatch": "tensor_shape_mismatch",
    "shape_mismatch": "tensor_shape_mismatch",
    "dimension_mismatch": "tensor_shape_mismatch",
    "data_preprocessing_error": "data_preprocessing_error",
    "preprocessing_error": "data_preprocessing_error",
    "normalization_mismatch": "data_preprocessing_error",
    "label_encoding_error": "label_encoding_error",
    "label_error": "label_encoding_error",
    "encoding_error": "label_encoding_error",
    "model_configuration_error": "model_configuration_error",
    "model_config_error": "model_configuration_error",
    "optimizer_configuration_error": "optimizer_configuration_error",
    "optimizer_config_error": "optimizer_configuration_error",
    "optimizer_error": "optimizer_configuration_error",
    "training_loop_logic_error": "training_loop_logic_error",
    "training_loop_error": "training_loop_logic_error",
    "loop_error": "training_loop_logic_error",
    "gradient_flow_failure": "gradient_flow_failure",
    "gradient_flow": "gradient_flow_failure",
    "detached_graph": "gradient_flow_failure",
    "missing_gradient": "gradient_flow_failure",
    "numerical_instability": "numerical_instability",
    "nan_inf": "numerical_instability",
    "nan": "numerical_instability",
    "inf": "numerical_instability",
    "evaluation_logic_error": "evaluation_logic_error",
    "evaluation_error": "evaluation_logic_error",
    "metric_error": "evaluation_logic_error",
    "memory_resource_exhaustion": "memory_resource_exhaustion",
    "oom": "memory_resource_exhaustion",
    "out_of_memory": "memory_resource_exhaustion",
    "memory_error": "memory_resource_exhaustion",
    "device_placement_error": "device_placement_error",
    "device_error": "device_placement_error",
    "device_mismatch": "device_placement_error",
}

# (keywords_for_text_search, category_id)
_KEYWORD_RULES: List[Tuple[Tuple[str, ...], str]] = [
    # gradient management
    (("zero_grad", "zero grad"), "gradient_management_error"),
    (("gradient accumulation", "stale gradient"), "gradient_management_error"),
    # learning rate / instability
    (("learning rate", "learning_rate", "lr too high", "exploding gradient", "gradient norm"), "learning_rate_instability"),
    (("scheduler", "warmup", "lr schedule"), "learning_rate_instability"),
    # loss / output mismatch
    (("cross_entropy", "cross entropy", "bce", "softmax", "log_softmax", "double softmax"), "loss_output_mismatch"),
    (("activation incompatible", "loss mismatch", "wrong loss", "loss function"), "loss_output_mismatch"),
    (("loss reduction", "loss masking"), "loss_output_mismatch"),
    # tensor shape
    (("shape mismatch", "dimension mismatch", "size mismatch", "reshape", "view(", "matmul", "einsum"), "tensor_shape_mismatch"),
    # data preprocessing
    (("normali", "standardiz", "augmentation", "preprocessing", "transform", "data pipeline"), "data_preprocessing_error"),
    (("train test split", "train val", "data loader"), "data_preprocessing_error"),
    # label encoding
    (("one-hot", "one hot", "label encoding", "label format", "class index", "wrong label"), "label_encoding_error"),
    (("target label", "target interpretation"), "label_encoding_error"),
    # model config
    (("num_classes", "output unit", "out_features", "hidden_size", "architecture"), "model_configuration_error"),
    (("model output shape", "final layer"), "model_configuration_error"),
    # optimizer config
    (("optimizer", "momentum", "weight_decay", "betas", "adam", "sgd", "rmsprop"), "optimizer_configuration_error"),
    # training loop
    (("model.train", "model.eval", "no_grad", "training loop", "backward called"), "training_loop_logic_error"),
    (("train mode", "eval mode", "batch norm", "dropout"), "training_loop_logic_error"),
    # gradient flow
    (("detach", "requires_grad", "frozen", "stop_gradient", "gradient not flowing"), "gradient_flow_failure"),
    (("gradient zero", "gradient none", "gradient missing", "parameters not updating"), "gradient_flow_failure"),
    # numerical instability
    (("nan", "inf", "overflow", "underflow", "division by zero", "log zero", "numerical"), "numerical_instability"),
    # evaluation logic
    (("validation loss", "validation performance", "evaluation logic", "metric", "eval mode"), "evaluation_logic_error"),
    (("train val gap", "performance collapse"), "evaluation_logic_error"),
    # memory
    (("out of memory", "oom", "cuda memory", "memory exhausted", "cache", "memory error"), "memory_resource_exhaustion"),
    (("gpu memory", "batch size", "empty_cache"), "memory_resource_exhaustion"),
    # device
    (("device mismatch", "cpu gpu", "wrong device", "device placement", "tensor on cpu"), "device_placement_error"),
    (("cuda error", "expected device"), "device_placement_error"),
]


def _normalize_type_string(raw_type: str) -> str:
    """Convert a raw type string to a lowercase, underscore-separated key."""
    return re.sub(r"[\s\-]+", "_", raw_type.lower()).strip("_")


def _match_categories(combined_text: str, raw_type: str) -> List[str]:
    """Return candidate taxonomy category IDs for the given finding.

    Args:
        combined_text: Lower-cased ``type + " " + message`` text.
        raw_type: The raw type field from the finding.

    Returns:
        Ordered list of matched category IDs (most specific first, deduped).
    """
    matched: List[str] = []
    seen: set = set()

    # 1. Exact / alias match on normalised type string
    normalised_type = _normalize_type_string(raw_type)
    if normalised_type in _TYPE_ALIASES:
        cat = _TYPE_ALIASES[normalised_type]
        if cat not in seen:
            matched.append(cat)
            seen.add(cat)

    # 2. Prefix match on type aliases
    for alias, cat in _TYPE_ALIASES.items():
        if normalised_type.startswith(alias) or alias.startswith(normalised_type):
            if cat not in seen:
                matched.append(cat)
                seen.add(cat)

    # 3. Keyword text scan over combined text
    for keywords, cat in _KEYWORD_RULES:
        if any(kw in combined_text for kw in keywords):
            if cat not in seen:
                matched.append(cat)
                seen.add(cat)

    return matched


# ---------------------------------------------------------------------------
# Public normaliser class
# ---------------------------------------------------------------------------


class FindingNormalizer:
    """Converts raw findings into :class:`~silent_bug_agent.diagnosis.models.findings.NormalizedFinding` objects.

    Accepts:
    * Typed Pydantic models (:class:`ASTFinding`, :class:`SemanticFinding`,
      :class:`RuntimeAlert`).
    * Plain ``dict`` objects (validated and coerced before normalisation).

    Invalid inputs are logged and skipped — the normaliser never raises on
    individual bad findings so that a single malformed entry does not abort
    the entire diagnosis.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_ast(
        self, findings: Sequence[Union[ASTFinding, Dict[str, Any]]]
    ) -> List[NormalizedFinding]:
        """Normalise a sequence of AST findings."""
        result: List[NormalizedFinding] = []
        for raw in findings:
            nf = self._to_ast(raw)
            if nf is not None:
                result.append(nf)
        return result

    def normalize_semantic(
        self, findings: Sequence[Union[SemanticFinding, Dict[str, Any]]]
    ) -> List[NormalizedFinding]:
        """Normalise a sequence of semantic findings."""
        result: List[NormalizedFinding] = []
        for raw in findings:
            nf = self._to_semantic(raw)
            if nf is not None:
                result.append(nf)
        return result

    def normalize_runtime(
        self, alerts: Sequence[Union[RuntimeAlert, Dict[str, Any]]]
    ) -> List[NormalizedFinding]:
        """Normalise a sequence of runtime alerts."""
        result: List[NormalizedFinding] = []
        for raw in alerts:
            nf = self._to_runtime(raw)
            if nf is not None:
                result.append(nf)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_ast(
        self, raw: Union[ASTFinding, Dict[str, Any]]
    ) -> Optional[NormalizedFinding]:
        try:
            finding = raw if isinstance(raw, ASTFinding) else ASTFinding(**raw)
        except Exception as exc:
            logger.warning("Skipping invalid AST finding %r: %s", raw, exc)
            return None
        return self._build_normalized(finding, SourceCategory.AST)

    def _to_semantic(
        self, raw: Union[SemanticFinding, Dict[str, Any]]
    ) -> Optional[NormalizedFinding]:
        try:
            finding = (
                raw if isinstance(raw, SemanticFinding) else SemanticFinding(**raw)
            )
        except Exception as exc:
            logger.warning("Skipping invalid semantic finding %r: %s", raw, exc)
            return None
        return self._build_normalized(finding, SourceCategory.SEMANTIC)

    def _to_runtime(
        self, raw: Union[RuntimeAlert, Dict[str, Any]]
    ) -> Optional[NormalizedFinding]:
        try:
            alert = raw if isinstance(raw, RuntimeAlert) else RuntimeAlert(**raw)
        except Exception as exc:
            logger.warning("Skipping invalid runtime alert %r: %s", raw, exc)
            return None
        return self._build_normalized(alert, SourceCategory.RUNTIME)

    def _build_normalized(
        self,
        finding: Union[ASTFinding, SemanticFinding, RuntimeAlert],
        source: SourceCategory,
    ) -> NormalizedFinding:
        combined = (finding.type + " " + finding.message).lower()
        candidates = _match_categories(combined, finding.type)

        file_val = getattr(finding, "file", None)
        line_val = getattr(finding, "line", None)

        return NormalizedFinding(
            original_id=finding.id,
            source=source,
            raw_type=finding.type,
            message_lower=finding.message.lower(),
            severity=finding.severity,
            file=file_val,
            line=line_val,
            metadata=finding.metadata if hasattr(finding, "metadata") else {},
            candidate_categories=candidates,
        )
