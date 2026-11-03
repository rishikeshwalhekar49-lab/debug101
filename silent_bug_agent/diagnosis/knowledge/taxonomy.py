"""
ML/DL Bug Root-Cause Taxonomy Knowledge Base.

This module defines the complete taxonomy of machine-learning and deep-learning
root-cause categories together with a lightweight deterministic causal graph
expressed as weighted directed edges.

The :class:`TaxonomyRegistry` singleton provides stable lookup by category ID
and exposes the causal graph for the inference engine.

Usage::

    from silent_bug_agent.diagnosis.knowledge.taxonomy import TAXONOMY_REGISTRY

    entry = TAXONOMY_REGISTRY.lookup("gradient_management_error")
    print(entry.name)
    print(entry.typical_static_indicators)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaxonomyEntry:
    """A single root-cause category in the ML/DL bug taxonomy.

    Attributes:
        category_id: Stable, machine-readable identifier.
        name: Human-readable short name.
        description: Longer explanation of what this root cause means.
        typical_static_indicators: Keywords or phrases found in AST findings.
        typical_semantic_indicators: Keywords or phrases found in semantic reviews.
        typical_runtime_indicators: Keywords or phrases found in runtime alerts.
        causal_parents: Category IDs that often *cause* this category (upstream).
        causal_children: Category IDs that this category often *causes* (downstream).
        verification_checks: Actionable checks an engineer can perform.
    """

    category_id: str
    name: str
    description: str
    typical_static_indicators: Tuple[str, ...] = field(default_factory=tuple)
    typical_semantic_indicators: Tuple[str, ...] = field(default_factory=tuple)
    typical_runtime_indicators: Tuple[str, ...] = field(default_factory=tuple)
    causal_parents: Tuple[str, ...] = field(default_factory=tuple)
    causal_children: Tuple[str, ...] = field(default_factory=tuple)
    verification_checks: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Causal graph edge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalEdge:
    """A weighted directed causal relationship between two taxonomy categories.

    ``source`` typically *causes* ``target``.  A weight of 1.0 is a strong
    causal link; 0.3 is a weak association.

    Attributes:
        source: Category ID of the upstream cause.
        target: Category ID of the downstream symptom/effect.
        weight: Strength of the causal link in [0.0, 1.0].
        rationale: Short description of why this link exists.
    """

    source: str
    target: str
    weight: float
    rationale: str


# ---------------------------------------------------------------------------
# Raw taxonomy data
# ---------------------------------------------------------------------------

_ENTRIES: List[TaxonomyEntry] = [
    TaxonomyEntry(
        category_id="gradient_management_error",
        name="Gradient Management Error",
        description=(
            "Gradients are not correctly zeroed, accumulated, or managed across "
            "training steps.  The most common manifestation is a missing call to "
            "optimizer.zero_grad() before the backward pass, causing gradient "
            "accumulation across batches."
        ),
        typical_static_indicators=(
            "zero_grad",
            "missing zero_grad",
            "optimizer.zero_grad",
            "gradient accumulation",
            "backward",
            "retain_graph",
        ),
        typical_semantic_indicators=(
            "gradients accumulated across batches",
            "gradient update incorrect",
            "stale gradients",
            "gradient not cleared",
        ),
        typical_runtime_indicators=(
            "gradient norm increasing",
            "gradient norm large",
            "gradient norm spike",
            "loss not decreasing",
            "loss oscillating",
            "gradient accumulation detected",
        ),
        causal_parents=(),
        causal_children=("numerical_instability", "learning_rate_instability"),
        verification_checks=(
            "Confirm optimizer.zero_grad() is called before loss.backward() in every training step.",
            "Check whether gradient accumulation is intentional (e.g., gradient_accumulation_steps > 1).",
            "Log gradient norms per step and verify they remain stable.",
            "Inspect retained computation graphs for unintentional retain_graph=True usage.",
        ),
    ),
    TaxonomyEntry(
        category_id="learning_rate_instability",
        name="Learning Rate Instability",
        description=(
            "The learning rate is set too high, decays incorrectly, or is scheduled "
            "inappropriately, causing divergent training, loss spikes, or oscillation."
        ),
        typical_static_indicators=(
            "learning_rate",
            "lr",
            "optimizer",
            "scheduler",
            "step_size",
            "warmup",
            "lr_scheduler",
        ),
        typical_semantic_indicators=(
            "excessive learning rate",
            "lr too high",
            "scheduler misconfigured",
            "warmup missing",
            "lr schedule incorrect",
        ),
        typical_runtime_indicators=(
            "exploding gradient",
            "gradient norm very large",
            "loss diverging",
            "loss nan",
            "loss inf",
            "loss oscillating",
            "training unstable",
            "gradient norm spike",
        ),
        causal_parents=("gradient_management_error",),
        causal_children=("numerical_instability",),
        verification_checks=(
            "Reduce the learning rate by an order of magnitude and observe loss behaviour.",
            "Add gradient clipping (torch.nn.utils.clip_grad_norm_) with a suitable threshold.",
            "Log the effective learning rate at each step and compare to expected schedule.",
            "Verify scheduler.step() is called at the correct frequency (per batch vs per epoch).",
        ),
    ),
    TaxonomyEntry(
        category_id="loss_output_mismatch",
        name="Loss / Output Mismatch",
        description=(
            "The model's output activation function, output shape, or value range is "
            "incompatible with the selected loss function, or the loss is applied "
            "incorrectly (e.g., applying log-softmax before CrossEntropyLoss, which "
            "already applies log-softmax internally)."
        ),
        typical_static_indicators=(
            "loss function",
            "cross_entropy",
            "bce",
            "nll_loss",
            "softmax",
            "sigmoid",
            "activation",
            "log_softmax",
            "output shape",
        ),
        typical_semantic_indicators=(
            "output activation incompatible",
            "loss function mismatch",
            "double softmax",
            "sigmoid before bce",
            "raw logits expected",
            "loss reduction incorrect",
            "wrong loss function",
        ),
        typical_runtime_indicators=(
            "loss nan",
            "loss inf",
            "loss constant",
            "loss not decreasing",
            "negative loss",
            "loss value unexpected",
        ),
        causal_parents=(),
        causal_children=("numerical_instability",),
        verification_checks=(
            "Verify that CrossEntropyLoss receives raw logits (not softmax outputs).",
            "Verify that BCELoss receives values in (0, 1) — use BCEWithLogitsLoss for raw logits.",
            "Check output tensor shape matches what the loss function expects.",
            "Print model output statistics (min, max, mean) before loss computation.",
        ),
    ),
    TaxonomyEntry(
        category_id="tensor_shape_mismatch",
        name="Tensor Shape Mismatch",
        description=(
            "Tensors involved in an operation have incompatible shapes.  This can arise "
            "from incorrect reshaping, wrong batch dimensions, misaligned feature sizes, "
            "or incorrect broadcasting assumptions."
        ),
        typical_static_indicators=(
            "reshape",
            "view",
            "transpose",
            "permute",
            "shape",
            "dimension",
            "size mismatch",
            "matmul",
            "mm",
            "einsum",
        ),
        typical_semantic_indicators=(
            "tensor shape incorrect",
            "dimension mismatch",
            "shape incompatible",
            "wrong batch dimension",
            "feature size mismatch",
        ),
        typical_runtime_indicators=(
            "shape mismatch",
            "size mismatch",
            "dimension error",
            "runtime error shape",
            "tensor shape error",
        ),
        causal_parents=(),
        causal_children=(),
        verification_checks=(
            "Add assert statements or logging for tensor shapes at each critical step.",
            "Use torchsummary or model.eval() forward pass with dummy inputs to verify shapes.",
            "Check reshape/view calls preserve the total number of elements.",
            "Verify batch dimension handling when batch_size=1 (potential squeeze issues).",
        ),
    ),
    TaxonomyEntry(
        category_id="data_preprocessing_error",
        name="Data Preprocessing Error",
        description=(
            "The training or evaluation data is not preprocessed consistently — "
            "normalisation statistics differ between train and val/test splits, "
            "augmentation is applied at inference, or preprocessing steps are "
            "applied in the wrong order."
        ),
        typical_static_indicators=(
            "transform",
            "normalize",
            "standardize",
            "preprocessing",
            "augmentation",
            "data_loader",
            "dataset",
        ),
        typical_semantic_indicators=(
            "preprocessing inconsistent",
            "normalization mismatch",
            "augmentation at test time",
            "train test preprocessing different",
            "data pipeline incorrect",
        ),
        typical_runtime_indicators=(
            "validation loss high",
            "train val gap",
            "performance collapse",
            "unexpected input range",
        ),
        causal_parents=(),
        causal_children=("evaluation_logic_error",),
        verification_checks=(
            "Compare training and validation/test preprocessing pipelines side by side.",
            "Verify normalisation statistics (mean, std) are computed on train set only.",
            "Confirm data augmentation is disabled at validation/test time.",
            "Check the order of transforms (e.g., ToTensor before Normalize).",
        ),
    ),
    TaxonomyEntry(
        category_id="label_encoding_error",
        name="Label Encoding Error",
        description=(
            "Class labels are encoded in a format incompatible with the loss function "
            "or model output.  Common issues include one-hot encoding where integer "
            "indices are expected, off-by-one label indices, or incorrect label dtype."
        ),
        typical_static_indicators=(
            "label",
            "target",
            "one_hot",
            "class_index",
            "encoding",
            "dtype",
            "long",
            "float",
        ),
        typical_semantic_indicators=(
            "label encoding incorrect",
            "one-hot vs index",
            "label format mismatch",
            "target interpretation incorrect",
            "wrong label dtype",
        ),
        typical_runtime_indicators=(
            "loss nan",
            "index out of range",
            "label error",
            "target shape mismatch",
        ),
        causal_parents=(),
        causal_children=("loss_output_mismatch",),
        verification_checks=(
            "Confirm CrossEntropyLoss receives integer class indices, not one-hot vectors.",
            "Verify BCELoss/BCEWithLogitsLoss receives float targets in {0.0, 1.0}.",
            "Check label range: indices must be in [0, num_classes - 1].",
            "Inspect a sample batch of (input, target) pairs before training starts.",
        ),
    ),
    TaxonomyEntry(
        category_id="model_configuration_error",
        name="Model Configuration Error",
        description=(
            "The model architecture is mis-configured: wrong number of output units, "
            "incorrect activation functions, missing or duplicate layers, or "
            "incompatible layer parameters."
        ),
        typical_static_indicators=(
            "num_classes",
            "out_features",
            "activation",
            "layer",
            "architecture",
            "hidden_size",
            "in_channels",
            "out_channels",
        ),
        typical_semantic_indicators=(
            "output units incorrect",
            "activation mismatch",
            "layer missing",
            "architecture wrong",
            "model output shape wrong",
        ),
        typical_runtime_indicators=(
            "output shape mismatch",
            "performance not improving",
            "model not learning",
        ),
        causal_parents=(),
        causal_children=("loss_output_mismatch", "tensor_shape_mismatch"),
        verification_checks=(
            "Verify the final layer output size matches num_classes.",
            "Check that the activation function in the final layer is compatible with the loss.",
            "Print model architecture (print(model)) and inspect layer shapes.",
            "Run a single forward pass and check output shape.",
        ),
    ),
    TaxonomyEntry(
        category_id="optimizer_configuration_error",
        name="Optimizer Configuration Error",
        description=(
            "The optimizer is configured with incorrect hyperparameters (e.g., wrong "
            "momentum, weight decay, betas) or the wrong optimizer type is used for "
            "the task."
        ),
        typical_static_indicators=(
            "optimizer",
            "adam",
            "sgd",
            "rmsprop",
            "momentum",
            "weight_decay",
            "betas",
            "eps",
        ),
        typical_semantic_indicators=(
            "optimizer misconfigured",
            "wrong optimizer",
            "momentum incorrect",
            "weight decay too high",
            "betas incorrect",
        ),
        typical_runtime_indicators=(
            "loss not converging",
            "training slow",
            "loss oscillating",
        ),
        causal_parents=(),
        causal_children=("learning_rate_instability",),
        verification_checks=(
            "Review optimizer hyperparameters against published baselines for the task.",
            "Try Adam with default betas=(0.9, 0.999) if unsure.",
            "Check that all model parameters are passed to the optimizer.",
            "Verify weight_decay value is appropriate (e.g., 1e-4 to 1e-2 for regularisation).",
        ),
    ),
    TaxonomyEntry(
        category_id="training_loop_logic_error",
        name="Training Loop Logic Error",
        description=(
            "The training loop itself is incorrect: the backward pass is called at the "
            "wrong point, evaluation is mixed with training mode, the model is not set "
            "to train() mode during training, or the loop structure is fundamentally wrong."
        ),
        typical_static_indicators=(
            "model.train",
            "model.eval",
            "with torch.no_grad",
            "loss.backward",
            "optimizer.step",
            "loop",
            "epoch",
            "batch",
        ),
        typical_semantic_indicators=(
            "training and evaluation logic inconsistent",
            "model not in train mode",
            "backward called incorrectly",
            "training loop incorrect",
            "no_grad context missing",
        ),
        typical_runtime_indicators=(
            "loss not decreasing",
            "performance constant",
            "batch norm behaviour unexpected",
            "dropout active at test time",
        ),
        causal_parents=(),
        causal_children=(
            "gradient_management_error",
            "evaluation_logic_error",
        ),
        verification_checks=(
            "Verify model.train() is called before the training loop and model.eval() before validation.",
            "Confirm loss.backward() is only called during training, not validation.",
            "Check that optimizer.step() follows loss.backward() and zero_grad() in the correct order.",
            "Use torch.no_grad() context for all validation/test forward passes.",
        ),
    ),
    TaxonomyEntry(
        category_id="gradient_flow_failure",
        name="Gradient Flow Failure",
        description=(
            "Gradients do not flow through part or all of the computation graph.  "
            "Common causes include .detach() calls breaking the graph, non-differentiable "
            "operations, or requires_grad=False on parameters that should be trained."
        ),
        typical_static_indicators=(
            "detach",
            "requires_grad",
            "no_grad",
            "stop_gradient",
            "leaf variable",
            "frozen",
            "non-differentiable",
        ),
        typical_semantic_indicators=(
            "gradient not flowing",
            "detached computation graph",
            "missing detach",
            "incorrectly detached",
            "requires_grad false",
            "frozen layers incorrect",
        ),
        typical_runtime_indicators=(
            "gradient zero",
            "gradient none",
            "gradient missing",
            "parameters not updating",
            "loss constant",
        ),
        causal_parents=(),
        causal_children=("gradient_management_error",),
        verification_checks=(
            "Register gradient hooks and verify gradients are non-zero for key parameters.",
            "Check for unintentional .detach() calls in the forward pass.",
            "Verify requires_grad=True for all parameters intended to be trained.",
            "Use model.named_parameters() to list parameters and their requires_grad status.",
        ),
    ),
    TaxonomyEntry(
        category_id="numerical_instability",
        name="Numerical Instability",
        description=(
            "Computations produce NaN or Inf values due to numerical overflow, underflow, "
            "division by zero, log of a non-positive number, or other floating-point issues."
        ),
        typical_static_indicators=(
            "nan",
            "inf",
            "log",
            "sqrt",
            "exp",
            "division",
            "clamp",
            "epsilon",
        ),
        typical_semantic_indicators=(
            "numerical instability",
            "nan value",
            "inf value",
            "overflow",
            "underflow",
            "division by zero",
            "log zero",
        ),
        typical_runtime_indicators=(
            "loss nan",
            "loss inf",
            "gradient nan",
            "gradient inf",
            "weight nan",
            "activation nan",
            "weight inf",
            "activation inf",
            "nan detected",
            "inf detected",
        ),
        causal_parents=(
            "gradient_management_error",
            "learning_rate_instability",
            "loss_output_mismatch",
        ),
        causal_children=(),
        verification_checks=(
            "Add torch.autograd.set_detect_anomaly(True) to find the first NaN-producing op.",
            "Clamp inputs to log/sqrt to a small positive epsilon.",
            "Use mixed precision (torch.cuda.amp) carefully — check for underflow in fp16.",
            "Reduce the learning rate and observe whether NaN disappears.",
            "Check initial weight scales (e.g., very large initialisation can cause immediate NaN).",
        ),
    ),
    TaxonomyEntry(
        category_id="evaluation_logic_error",
        name="Evaluation Logic Error",
        description=(
            "The evaluation procedure is incorrect: metrics are computed on training data "
            "instead of validation data, the model is not set to eval() mode, dropout or "
            "batch normalisation behave as in training, or metric implementation is wrong."
        ),
        typical_static_indicators=(
            "model.eval",
            "metric",
            "accuracy",
            "validation",
            "test",
            "predict",
        ),
        typical_semantic_indicators=(
            "evaluation logic incorrect",
            "metric incorrect",
            "validation measured on train data",
            "model not in eval mode",
            "evaluation inconsistent",
        ),
        typical_runtime_indicators=(
            "validation performance collapses",
            "train val gap large",
            "performance not improving",
            "overfitting sudden",
        ),
        causal_parents=("training_loop_logic_error", "data_preprocessing_error"),
        causal_children=(),
        verification_checks=(
            "Confirm model.eval() is called before any validation/test loop.",
            "Verify metrics are accumulated on validation set, not training set.",
            "Use a fixed validation set that is never used for hyperparameter selection.",
            "Check that metric implementation handles edge cases (empty batch, class imbalance).",
        ),
    ),
    TaxonomyEntry(
        category_id="memory_resource_exhaustion",
        name="Memory / Resource Exhaustion",
        description=(
            "The GPU or CPU runs out of memory, typically because batch size is too large, "
            "computation graphs are not freed, or tensors are accumulated in lists "
            "without detaching."
        ),
        typical_static_indicators=(
            "batch_size",
            "cuda",
            "memory",
            "oom",
            "out of memory",
            "cache",
            "del",
            "empty_cache",
        ),
        typical_semantic_indicators=(
            "batch size too large",
            "memory leak",
            "graph not freed",
            "tensor accumulation",
        ),
        typical_runtime_indicators=(
            "cuda out of memory",
            "oom",
            "memory error",
            "killed",
            "gpu memory exhausted",
            "runtime error memory",
        ),
        causal_parents=(),
        causal_children=(),
        verification_checks=(
            "Reduce batch size by half and observe if OOM disappears.",
            "Call torch.cuda.empty_cache() between epochs.",
            "Check for loss/metric accumulation loops that forget .item() (which would retain graphs).",
            "Use gradient checkpointing for large models.",
        ),
    ),
    TaxonomyEntry(
        category_id="device_placement_error",
        name="Device Placement Error",
        description=(
            "Tensors or model parameters are on different devices (CPU vs GPU) causing "
            "runtime errors, or mixed-device operations that silently fall back to CPU."
        ),
        typical_static_indicators=(
            "cuda",
            "cpu",
            "device",
            "to(device)",
            ".cuda()",
            ".cpu()",
        ),
        typical_semantic_indicators=(
            "device mismatch",
            "tensor on wrong device",
            "cpu gpu mismatch",
            "device placement incorrect",
        ),
        typical_runtime_indicators=(
            "device error",
            "expected device",
            "tensor on cpu",
            "cuda error",
            "device mismatch",
        ),
        causal_parents=(),
        causal_children=(),
        verification_checks=(
            "Verify all tensors and model are moved to the same device before use.",
            "Use a device variable (device = torch.device('cuda' if available else 'cpu')).",
            "Check data loader output is moved to device before passing to model.",
        ),
    ),
    TaxonomyEntry(
        category_id="unknown_or_insufficient_evidence",
        name="Unknown / Insufficient Evidence",
        description=(
            "The available evidence is insufficient, ambiguous, or contradictory to "
            "support a confident root-cause diagnosis.  Further investigation is required."
        ),
        typical_static_indicators=(),
        typical_semantic_indicators=(),
        typical_runtime_indicators=(),
        causal_parents=(),
        causal_children=(),
        verification_checks=(
            "Collect more diagnostic information: logs, profiling traces, runtime alerts.",
            "Reproduce the issue in a minimal script to isolate the failing component.",
            "Enable verbose logging and anomaly detection (torch.autograd.set_detect_anomaly(True)).",
        ),
    ),
]

# ---------------------------------------------------------------------------
# Causal graph edges
# ---------------------------------------------------------------------------

_EDGES: List[CausalEdge] = [
    # gradient_management_error → gradient_flow_failure (accumulated gradients make huge steps that kill units)
    CausalEdge(
        source="gradient_management_error",
        target="gradient_flow_failure",
        weight=0.60,
        rationale="Gradients that are never cleared grow every step; the oversized updates push ReLU units dead.",
    ),
    # loss_output_mismatch → learning_rate_instability (a summed loss scales every step like a larger LR)
    CausalEdge(
        source="loss_output_mismatch",
        target="learning_rate_instability",
        weight=0.50,
        rationale="A summed loss makes every update batch-size times larger, which looks like a learning rate set too high.",
    ),
    # gradient_management_error → numerical_instability
    CausalEdge(
        source="gradient_management_error",
        target="numerical_instability",
        weight=0.70,
        rationale="Accumulated gradients can explode, producing NaN/Inf.",
    ),
    # gradient_management_error → learning_rate_instability (symptom)
    CausalEdge(
        source="gradient_management_error",
        target="learning_rate_instability",
        weight=0.50,
        rationale="Large accumulated gradients mimic the effect of a high learning rate.",
    ),
    # learning_rate_instability → numerical_instability
    CausalEdge(
        source="learning_rate_instability",
        target="numerical_instability",
        weight=0.75,
        rationale="An excessive LR causes gradient explosion → NaN.",
    ),
    # loss_output_mismatch → numerical_instability
    CausalEdge(
        source="loss_output_mismatch",
        target="numerical_instability",
        weight=0.65,
        rationale="Double softmax / incompatible activations often yield log(0) → -Inf.",
    ),
    # label_encoding_error → loss_output_mismatch
    CausalEdge(
        source="label_encoding_error",
        target="loss_output_mismatch",
        weight=0.60,
        rationale="Wrong label format causes the loss function to produce incorrect values.",
    ),
    # model_configuration_error → loss_output_mismatch
    CausalEdge(
        source="model_configuration_error",
        target="loss_output_mismatch",
        weight=0.55,
        rationale="Wrong output activation or shape makes the output incompatible with the loss.",
    ),
    # model_configuration_error → tensor_shape_mismatch
    CausalEdge(
        source="model_configuration_error",
        target="tensor_shape_mismatch",
        weight=0.45,
        rationale="Misconfigured layer dimensions cause shape errors at runtime.",
    ),
    # optimizer_configuration_error → learning_rate_instability
    CausalEdge(
        source="optimizer_configuration_error",
        target="learning_rate_instability",
        weight=0.55,
        rationale="Wrong optimizer settings (e.g., momentum) can destabilise training.",
    ),
    # training_loop_logic_error → gradient_management_error
    CausalEdge(
        source="training_loop_logic_error",
        target="gradient_management_error",
        weight=0.60,
        rationale="Incorrect loop ordering can cause zero_grad() to be skipped.",
    ),
    # training_loop_logic_error → evaluation_logic_error
    CausalEdge(
        source="training_loop_logic_error",
        target="evaluation_logic_error",
        weight=0.50,
        rationale="A broken loop often affects both train and eval phases.",
    ),
    # gradient_flow_failure → gradient_management_error
    CausalEdge(
        source="gradient_flow_failure",
        target="gradient_management_error",
        weight=0.45,
        rationale="Detached graphs can appear as zero gradients or missing updates.",
    ),
    # data_preprocessing_error → evaluation_logic_error
    CausalEdge(
        source="data_preprocessing_error",
        target="evaluation_logic_error",
        weight=0.55,
        rationale="Inconsistent preprocessing makes validation metrics unreliable.",
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TaxonomyRegistry:
    """Singleton registry for ML/DL bug taxonomy entries and causal graph.

    Access the global instance via ``TAXONOMY_REGISTRY``; do not instantiate
    this class directly in production code.
    """

    def __init__(
        self,
        entries: List[TaxonomyEntry],
        edges: List[CausalEdge],
    ) -> None:
        self._entries: Dict[str, TaxonomyEntry] = {e.category_id: e for e in entries}
        self._edges: List[CausalEdge] = edges
        # Pre-build adjacency: source → {target: weight}
        self._causes: Dict[str, Dict[str, float]] = {}
        self._caused_by: Dict[str, Dict[str, float]] = {}
        for edge in edges:
            self._causes.setdefault(edge.source, {})[edge.target] = edge.weight
            self._caused_by.setdefault(edge.target, {})[edge.source] = edge.weight

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def lookup(self, category_id: str) -> TaxonomyEntry:
        """Return the :class:`TaxonomyEntry` for *category_id*.

        Raises:
            KeyError: If *category_id* is not in the taxonomy.
        """
        try:
            return self._entries[category_id]
        except KeyError:
            valid = ", ".join(sorted(self._entries.keys()))
            raise KeyError(
                f"Unknown taxonomy category: {category_id!r}. "
                f"Valid categories: {valid}"
            ) from None

    def get_all(self) -> Dict[str, TaxonomyEntry]:
        """Return a shallow copy of all taxonomy entries keyed by category ID."""
        return dict(self._entries)

    def category_ids(self) -> FrozenSet[str]:
        """Return the set of all valid category IDs."""
        return frozenset(self._entries.keys())

    # ------------------------------------------------------------------
    # Causal graph helpers
    # ------------------------------------------------------------------

    def causes(self, category_id: str) -> Dict[str, float]:
        """Return downstream categories that *category_id* typically causes.

        Returns a dict mapping downstream category ID → edge weight.
        """
        return dict(self._causes.get(category_id, {}))

    def caused_by(self, category_id: str) -> Dict[str, float]:
        """Return upstream categories that typically cause *category_id*.

        Returns a dict mapping upstream category ID → edge weight.
        """
        return dict(self._caused_by.get(category_id, {}))

    def get_edges(self) -> List[CausalEdge]:
        """Return all causal edges in the graph."""
        return list(self._edges)

    def edge_weight(self, source: str, target: str) -> float:
        """Return the causal edge weight from *source* → *target*, or 0.0."""
        return self._causes.get(source, {}).get(target, 0.0)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

TAXONOMY_REGISTRY: TaxonomyRegistry = TaxonomyRegistry(
    entries=_ENTRIES,
    edges=_EDGES,
)
