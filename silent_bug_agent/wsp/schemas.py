"""Data contracts for the read-only weight inspector."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


class WeightAnomalyType(str, Enum):
    NAN_INF = "nan_inf"                # non-finite values
    EXPLODED = "exploded"              # finite but implausibly large values
    ALL_ZERO = "all_zero"              # a weight matrix that is entirely zero
    BIAS_COLLAPSE = "bias_collapse"    # biases so negative that a following ReLU cannot fire
    RANK_COLLAPSE = "rank_collapse"    # a matrix that uses a small fraction of its dimensions
    WEIGHT_OUTLIER = "weight_outlier"  # a few entries far larger than the rest


@dataclass
class WeightFinding:
    """One measured anomaly in a named tensor."""
    layer_name: str
    param_shape: Tuple[int, ...]
    anomaly_type: WeightAnomalyType
    severity: str                      # "critical" | "warning" | "info"
    metric_value: float
    threshold: float
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "param_shape": list(self.param_shape),
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "message": self.message,
            "evidence": self.evidence,
            "suggested_action": self.suggested_action,
        }


@dataclass
class WeightReport:
    """Every finding for one set of weights, and where those weights came from."""
    model_name: str
    total_parameters: int = 0
    source: str = ""                   # e.g. "checkpoint model.pt" or "weights after 60 training steps"
    findings: List[WeightFinding] = field(default_factory=list)
    layer_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    inspected: bool = False            # False when no weights could be read at all

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.findings)

    @property
    def has_issues(self) -> bool:
        return any(f.severity in ("critical", "warning") for f in self.findings)

    def sorted_findings(self) -> List[WeightFinding]:
        return sorted(self.findings, key=lambda f: (SEVERITY_ORDER.get(f.severity, 3), f.layer_name))

    def to_dict(self) -> Dict[str, Any]:
        counts = {name: sum(f.severity == name for f in self.findings) for name in SEVERITY_ORDER}
        return {
            "model_name": self.model_name,
            "source": self.source,
            "inspected": self.inspected,
            "total_parameters": self.total_parameters,
            "summary": {"by_severity": counts},
            "findings": [f.to_dict() for f in self.sorted_findings()],
            "layer_summaries": self.layer_summaries,
            "diagnostics": self.diagnostics,
        }

    def format_text(self) -> str:
        lines = [f"Weight inspection of {self.model_name} (read-only)", f"Source: {self.source or 'unknown'}"]
        if self.inspected:
            lines.append(f"Values inspected: {self.total_parameters:,} in {len(self.layer_summaries)} matrices")
            if not self.findings:
                lines.append("No weight anomalies found.")
        for finding in self.sorted_findings():
            lines.append(f"[{finding.severity.upper()}] {finding.anomaly_type.value} in {finding.layer_name} "
                         f"{list(finding.param_shape)}")
            lines.append(f"    {finding.message}")
            if finding.suggested_action:
                lines.append(f"    next: {finding.suggested_action}")
        lines += [f"note: {note}" for note in self.diagnostics]
        return "\n".join(lines)
