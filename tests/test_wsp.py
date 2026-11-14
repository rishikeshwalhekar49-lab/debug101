"""Unit tests for the Weight-Space Programming (WSP) module."""

import math
import pytest
import torch
import torch.nn as nn

from silent_bug_agent.wsp.inspector import WeightInspector
from silent_bug_agent.wsp.editor import WeightEditor
from silent_bug_agent.wsp.schemas import (
    WeightAnomalyType,
    WeightSurgeryType,
    DualSolution,
)


class SimpleBuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        with torch.no_grad():
            self.fc1.bias.fill_(-100.0)  # Catastrophic negative bias
        self.fc2 = nn.Linear(16, 2)


class RankCollapsedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            # Rank 1 matrix in a 16x16 parameter
            u = torch.randn(16, 1)
            v = torch.randn(16, 1)
            self.fc.weight.copy_(torch.matmul(u, v.T))


class OutlierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self.fc.weight.normal_(0, 0.02)
            # Inject extreme kurtosis outlier
            self.fc.weight[0, 0] = 50.0


def test_inspector_detects_bias_collapse():
    inspector = WeightInspector()
    model = SimpleBuggyModel()
    report = inspector.inspect_model(model)

    assert report.has_critical
    bias_findings = [f for f in report.findings if f.anomaly_type == WeightAnomalyType.BIAS_COLLAPSE]
    assert len(bias_findings) >= 1
    assert bias_findings[0].layer_name == "fc1.bias"
    assert bias_findings[0].metric_value == -100.0


def test_inspector_detects_rank_collapse():
    inspector = WeightInspector(min_effective_rank_ratio=0.2)
    model = RankCollapsedModel()
    report = inspector.inspect_model(model)

    rank_findings = [f for f in report.findings if f.anomaly_type == WeightAnomalyType.RANK_COLLAPSE]
    assert len(rank_findings) >= 1
    assert rank_findings[0].layer_name == "fc.weight"


def test_inspector_detects_weight_outliers():
    inspector = WeightInspector(max_kurtosis=20.0)
    model = OutlierModel()
    report = inspector.inspect_model(model)

    outlier_findings = [f for f in report.findings if f.anomaly_type == WeightAnomalyType.WEIGHT_OUTLIER]
    assert len(outlier_findings) >= 1
    assert outlier_findings[0].layer_name == "fc.weight"


def test_editor_bias_center_patch():
    model = SimpleBuggyModel()
    inspector = WeightInspector()
    report = inspector.inspect_model(model)
    finding = next(f for f in report.findings if f.anomaly_type == WeightAnomalyType.BIAS_COLLAPSE)

    editor = WeightEditor()
    patch = editor.synthesize_patch(finding, model.fc1.bias)
    assert patch is not None
    assert patch.surgery_type == WeightSurgeryType.BIAS_CENTER

    # Apply patch
    success = editor.apply_patch(model, patch)
    assert success
    assert (model.fc1.bias == 0.0).all()

    # Re-inspect to confirm clean
    re_report = inspector.inspect_model(model)
    assert not any(f.anomaly_type == WeightAnomalyType.BIAS_COLLAPSE for f in re_report.findings)


def test_editor_subspace_excision():
    matrix = torch.randn(16, 16)
    # Add a massive dominant rank-1 direction
    v = torch.randn(16, 1)
    v = v / torch.norm(v)
    u = torch.randn(16, 1) * 20.0
    matrix = matrix + torch.matmul(u, v.T)

    editor = WeightEditor()
    patch = editor.create_subspace_excision_patch("layer.weight", matrix, direction_v=v)
    assert patch.surgery_type == WeightSurgeryType.SUBSPACE_EXCISION

    # Apply transformation
    patched_matrix = patch.transform_fn(matrix)
    # Projection along v should now be zero: W' v = 0
    proj = torch.matmul(patched_matrix, v)
    assert torch.norm(proj).item() < 1e-4


def test_editor_outlier_clamp():
    model = OutlierModel()
    inspector = WeightInspector(max_kurtosis=20.0)
    report = inspector.inspect_model(model)
    finding = next(f for f in report.findings if f.anomaly_type == WeightAnomalyType.WEIGHT_OUTLIER)

    editor = WeightEditor()
    patch = editor.synthesize_patch(finding, model.fc.weight)
    assert patch is not None
    assert patch.surgery_type == WeightSurgeryType.OUTLIER_CLAMP

    # Apply patch
    success = editor.apply_patch(model, patch)
    assert success
    assert model.fc.weight[0, 0].item() < 10.0

