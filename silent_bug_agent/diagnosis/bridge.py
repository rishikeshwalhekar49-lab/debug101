"""Connects the causal engine to the team's shared contracts.

The engine ranks 15 root-cause categories; the rest of the system speaks the 21
`bug_id`s of the taxonomy. This maps findings into the engine and its ranking back
out as `DiagnosticHypothesis` objects, so diagnosis uses the same bug list as
every detector.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..findings import Finding, Report
from ..invariants.base import Severity
from ..schemas import DiagnosticHypothesis, RepairStrategy

BUG_CATEGORY: Dict[str, str] = {
    "missing_zero_grad": "gradient_management_error",
    "gradient_accumulation_miscount": "gradient_management_error",
    "train_eval_mode_mismatch": "evaluation_logic_error",
    "incorrect_metric_averaging": "evaluation_logic_error",
    "frozen_layer_no_grad": "optimizer_configuration_error",
    "optimizer_state_leak_across_folds": "optimizer_configuration_error",
    "wrong_loss_reduction": "loss_output_mismatch",
    "mismatched_loss_task": "loss_output_mismatch",
    "softmax_before_cross_entropy": "loss_output_mismatch",
    "label_data_misalignment": "label_encoding_error",
    "data_leakage_train_test": "data_preprocessing_error",
    "non_normalized_inputs": "data_preprocessing_error",
    "duplicate_augmentation_applied_twice": "data_preprocessing_error",
    "exploding_gradients": "numerical_instability",
    "mixed_precision_overflow": "numerical_instability",
    "vanishing_gradients": "gradient_flow_failure",
    "dead_relu": "gradient_flow_failure",
    "learning_rate_misconfigured": "learning_rate_instability",
    "lr_scheduler_step_order_bug": "learning_rate_instability",
    "silent_device_fallback": "device_placement_error",
    "unset_random_seed": "training_loop_logic_error",
}

_STRATEGY = {"learning_rate_misconfigured": RepairStrategy.HYPERPARAM, "lr_scheduler_step_order_bug": RepairStrategy.REFACTOR}
_SEVERITY = {Severity.INFO: "low", Severity.WARNING: "medium", Severity.CRITICAL: "high"}
_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
# repeatability settings are worth fixing, but never before a bug that changes what the model learns
_LOW_PRIORITY = {"unset_random_seed": 0.3, "silent_device_fallback": 0.6}


def category_of(bug_id: str) -> str:
    return BUG_CATEGORY.get(bug_id, "unknown_or_insufficient_evidence")


def engine_inputs(findings: Sequence[Finding]) -> Dict[str, List[dict]]:
    """Split shared `Finding`s into the three lists `DiagnosisAgent.diagnose` takes."""
    inputs: Dict[str, List[dict]] = {"ast": [], "semantic": [], "runtime": []}
    for index, finding in enumerate(findings):
        common = {"id": f"F{index}", "type": category_of(finding.bug_id), "message": finding.message[:400],
                  "severity": _SEVERITY[finding.severity]}
        located = {"file": finding.file, "line": finding.line if finding.line >= 1 else None}
        if finding.detector == "runtime":
            inputs["runtime"].append({**common, "metadata": {"bug_id": finding.bug_id}})
        elif finding.detector == "semantic":
            inputs["semantic"].append({**common, **located})
        else:
            inputs["ast"].append({**common, **located})
    return inputs


def _noisy_or(values: Sequence[float]) -> float:
    miss = 1.0
    for value in values:
        miss *= 1.0 - max(0.0, min(1.0, value))
    return 1.0 - miss


def hypotheses_from_report(report: Report, category_confidence: Dict[str, float]) -> List[DiagnosticHypothesis]:
    """One hypothesis per detected bug, ranked by detector evidence and the engine's causal score."""
    grouped: Dict[str, List[Finding]] = {}
    for finding in report.findings:
        grouped.setdefault(finding.bug_id, []).append(finding)
    hypotheses: List[DiagnosticHypothesis] = []
    for bug_id, findings in grouped.items():
        evidence = _noisy_or([f.confidence for f in findings])
        causal = category_confidence.get(category_of(bug_id), 0.0)
        critical = any(f.severity == Severity.CRITICAL for f in findings)
        detectors = {f.detector for f in findings}
        score = 0.5 * evidence + 0.5 * causal + (0.1 if critical else 0.0) + (0.05 if len(detectors) > 1 else 0.0)
        score = min(0.99, score) * _LOW_PRIORITY.get(bug_id, 1.0)
        lead = max(findings, key=lambda f: f.confidence)
        hypotheses.append(DiagnosticHypothesis(
            bug_id=bug_id,
            description=lead.message[:300],
            confidence=round(score, 3),
            evidence_finding_keys=[f.key() for f in findings],
            suggested_strategy=_STRATEGY.get(bug_id, RepairStrategy.DIRECT_EDIT),
            extra={"category": category_of(bug_id), "detectors": sorted(detectors), "line": lead.line,
                   "max_severity": max((f.severity for f in findings), key=lambda s: _RANK[s]).value,
                   "alternatives": list(lead.extra.get("alternative_bug_ids", [])), "diagnosed_by": "causal_engine"},
        ))
    _causes_before_symptoms(hypotheses)
    hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    return hypotheses


def _causes_before_symptoms(hypotheses: List[DiagnosticHypothesis]) -> None:
    """If one detected bug can cause another (an edge in the causal graph), rank the cause first.

    A loss summed instead of averaged makes every step too large, which the runtime
    monitor sees as a learning-rate problem; repairing the symptom would hide the cause."""
    from .knowledge.taxonomy import TAXONOMY_REGISTRY

    for cause in hypotheses:
        for symptom in hypotheses:
            if cause is symptom or cause.bug_id in _LOW_PRIORITY:
                continue
            forward = TAXONOMY_REGISTRY.edge_weight(category_of(cause.bug_id), category_of(symptom.bug_id))
            backward = TAXONOMY_REGISTRY.edge_weight(category_of(symptom.bug_id), category_of(cause.bug_id))
            if forward > backward and cause.confidence <= symptom.confidence:
                cause.confidence = round(min(0.99, symptom.confidence + 0.01), 3)
                cause.extra.setdefault("explains", []).append(symptom.bug_id)
