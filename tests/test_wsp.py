"""The read-only weight inspector: what it flags, what it must not flag, and that it never runs or edits anything."""

import json
import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from silent_bug_agent.cli import main  # noqa: E402
from silent_bug_agent.wsp import WeightAnomalyType, WeightInspector, inspect_script  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "examples" / "sample_scripts"


def kinds(report, *severities):
    return {f.anomaly_type for f in report.findings if not severities or f.severity in severities}


class DeadBias(nn.Module):
    def __init__(self, dead=16):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        with torch.no_grad():
            self.fc1.bias[:dead] = -100.0
        self.fc2 = nn.Linear(16, 2)


class Healthy(nn.Module):
    """Layers that a careless check flags: zero biases, normalisation statistics, an embedding."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 32)
        self.conv = nn.Conv2d(3, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.norm = nn.LayerNorm(32)
        self.fc = nn.Linear(32, 32)
        self.head = nn.Linear(32, 10)
        nn.init.zeros_(self.fc.bias)
        nn.init.zeros_(self.head.bias)


# ------------------------------------------------------------- what it flags
def test_a_fully_dead_bias_is_critical():
    report = WeightInspector().inspect_model(DeadBias())
    finding = next(f for f in report.findings if f.anomaly_type == WeightAnomalyType.BIAS_COLLAPSE)
    assert finding.layer_name == "fc1.bias" and finding.severity == "critical" and finding.metric_value == -100.0
    assert report.has_critical and report.has_issues


def test_one_dead_bias_in_many_is_only_a_note():
    model = nn.Linear(8, 64)
    with torch.no_grad():
        model.bias[0] = -100.0
    report = WeightInspector().inspect_model(model)
    finding = next(f for f in report.findings if f.anomaly_type == WeightAnomalyType.BIAS_COLLAPSE)
    assert finding.severity == "info" and not report.has_issues


def test_rank_collapse_is_found():
    torch.manual_seed(0)
    layer = nn.Linear(16, 16, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(16, 1) @ torch.randn(1, 16))
    report = WeightInspector(min_effective_rank_ratio=0.2).inspect_model(layer)
    assert WeightAnomalyType.RANK_COLLAPSE in kinds(report, "warning", "critical")


def test_a_single_extreme_weight_is_a_warning():
    torch.manual_seed(0)
    layer = nn.Linear(32, 32, bias=False)
    with torch.no_grad():
        layer.weight.normal_(0, 0.02)
        layer.weight[0, 0] = 50.0
    report = WeightInspector().inspect_model(layer)
    finding = next(f for f in report.findings if f.anomaly_type == WeightAnomalyType.WEIGHT_OUTLIER)
    assert finding.severity == "warning" and finding.layer_name == "weight"


def test_nan_exploded_and_zeroed_weights_are_found():
    torch.manual_seed(0)
    state = {"a.weight": torch.randn(8, 8), "b.weight": torch.randn(8, 8) * 1e6, "c.weight": torch.zeros(8, 8)}
    state["a.weight"][0, 0] = float("nan")
    report = WeightInspector().inspect_state_dict(state)
    by_layer = {f.layer_name: f for f in report.findings}
    assert by_layer["a.weight"].anomaly_type == WeightAnomalyType.NAN_INF and by_layer["a.weight"].severity == "critical"
    assert by_layer["b.weight"].anomaly_type == WeightAnomalyType.EXPLODED
    assert by_layer["c.weight"].anomaly_type == WeightAnomalyType.ALL_ZERO


# ------------------------------------------------------ what it must not flag
def test_a_healthy_model_raises_no_warning():
    """Zero biases and normalisation statistics are normal, at initialisation and after training."""
    torch.manual_seed(0)
    model = Healthy()
    assert not WeightInspector().inspect_model(model).has_issues
    model.bn.running_var.fill_(60000.0)  # unnormalised pixel inputs give statistics this large
    report = WeightInspector().inspect_model(model)
    assert not report.has_issues, report.format_text()
    assert report.inspected and report.total_parameters > 0


# ---------------------------------------------------------------- checkpoints
def test_checkpoint_formats_are_read(tmp_path):
    model = DeadBias()
    plain, nested = tmp_path / "plain.pt", tmp_path / "nested.ckpt"
    torch.save(model.state_dict(), plain)
    torch.save({"epoch": 3, "state_dict": model.state_dict()}, nested)
    for path in (plain, nested):
        report = WeightInspector().inspect_checkpoint(path)
        assert report.inspected and WeightAnomalyType.BIAS_COLLAPSE in kinds(report), path.name
        assert path.name in report.source


class _RunsCodeWhenLoaded:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (open, (self.marker, "w"))


def test_a_malicious_checkpoint_is_not_executed(tmp_path):
    """A pickled object can run code when loaded. The inspector must refuse it, not run it."""
    marker, path = tmp_path / "pwned.txt", tmp_path / "evil.pt"
    with open(path, "wb") as handle:
        pickle.dump({"state_dict": _RunsCodeWhenLoaded(marker)}, handle, protocol=2)
    report = WeightInspector().inspect_checkpoint(path)
    assert not marker.exists()
    assert not report.inspected and "safely" in report.diagnostics[0]


def test_a_missing_checkpoint_is_reported(tmp_path):
    report = WeightInspector().inspect_checkpoint(tmp_path / "nope.pt")
    assert not report.inspected and "not found" in report.diagnostics[0]


# -------------------------------------------------------------------- scripts
def test_script_weights_are_read_after_training_and_the_script_is_untouched():
    script = SAMPLES / "benchmark_06_dead_relu.py"
    before = script.read_bytes()
    report = inspect_script(script, max_steps=20, timeout=120)
    assert script.read_bytes() == before
    assert report.inspected and "after 20 training step(s)" in report.source
    finding = next(f for f in report.findings if f.anomaly_type == WeightAnomalyType.BIAS_COLLAPSE)
    assert finding.severity == "critical" and finding.layer_name.endswith("bias")


def test_a_script_without_training_gives_no_weights(tmp_path):
    script = tmp_path / "no_training.py"
    script.write_text("import torch\nprint(torch.ones(2).sum())\n", encoding="utf-8")
    report = inspect_script(script, max_steps=5, timeout=60)
    assert not report.inspected and not report.findings
    assert "No weights were captured" in report.diagnostics[0]


# ------------------------------------------------------------------------ CLI
def test_cli_reports_and_exits_by_severity(tmp_path, capsys):
    bad, good = tmp_path / "bad.pt", tmp_path / "good.pt"
    torch.manual_seed(0)
    torch.save(DeadBias().state_dict(), bad)
    torch.save(Healthy().state_dict(), good)

    assert main(["wsp", "--checkpoint", str(bad), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["by_severity"]["critical"] == 1 and data["findings"][0]["anomaly_type"] == "bias_collapse"

    assert main(["wsp", "--checkpoint", str(good)]) == 0
    assert "No weight anomalies found" in capsys.readouterr().out

    assert main(["wsp"]) == 2
    assert main(["wsp", "--checkpoint", str(tmp_path / "missing.pt")]) == 2


def test_cli_still_accepts_the_old_flags_and_changes_nothing(tmp_path, capsys):
    checkpoint = tmp_path / "bad.pt"
    torch.save(DeadBias().state_dict(), checkpoint)
    before = checkpoint.read_bytes()
    assert main(["wsp", "--checkpoint", str(checkpoint), "--dry-run", "--yes", "--max-iter", "2"]) == 1
    assert "no longer do anything" in capsys.readouterr().err
    assert checkpoint.read_bytes() == before
