"""Version 4: code features, backward-pass tracing, hand-updated scripts, Colab scripts, label checks, Gate 2."""

import ast
import sys
from pathlib import Path

import pytest

from silent_bug_agent.learned.code import CODE_FEATURES, code_features
from silent_bug_agent.mutation.dataset import has_effect
from silent_bug_agent.mutation.operators import OPERATORS, mutants

REPO = Path(__file__).resolve().parent.parent

LOOP = """
import torch
model = torch.nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
x, y = torch.randn(64, 4), torch.randint(0, 2, (64,))
for epoch in range(3):
    model.train()
    for i in range(0, 64, 16):
        opt.zero_grad()
        loss = loss_fn(model(x[i:i + 16]), y[i:i + 16])
        loss.backward()
        opt.step()
    model.eval()
"""

MANUAL = """
import torch
torch.manual_seed(0)
x = torch.randn(256, 8)
y = x @ torch.randn(8, 1)
w = torch.zeros(8, 1, requires_grad=True)
lr = 0.05
for step in range(200):
    loss = ((x @ w - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()
"""


def test_code_features_read_the_facts():
    healthy = code_features(LOOP)
    assert set(healthy) == set(CODE_FEATURES) and healthy["code_parsed"] == 1.0
    assert healthy["code_backward_covered"] == 1.0 and healthy["code_has_eval"] == 1.0
    assert healthy["code_log_lr"] == pytest.approx(-1.0) and healthy["code_optimizer_partial"] == 0.0
    # zero_grad() called once, before the loop: the backward call is no longer covered
    once = LOOP.replace("        opt.zero_grad()\n", "").replace("for epoch in range(3):", "opt.zero_grad()\nfor epoch in range(3):")
    assert code_features(once)["code_backward_covered"] == 0.0
    # clearing by hand counts too, and so does a helper function that clears
    assert code_features(MANUAL)["code_backward_covered"] == 1.0
    helper = MANUAL.replace("        w.grad.zero_()\n", "").replace("for step", "def clear():\n    w.grad = None\n\n\nfor step")
    helper = helper.replace("        w -= lr * w.grad\n", "        w -= lr * w.grad\n    clear()\n")
    assert code_features(helper)["code_backward_covered"] == 1.0
    summed = LOOP.replace("CrossEntropyLoss()", 'CrossEntropyLoss(reduction="sum")').replace(
        "model.parameters()", "model.weight.parameters()")
    values = code_features(summed)
    assert values["code_reduction_sum"] == 1.0 and values["code_optimizer_partial"] == 1.0
    assert code_features("y = y[torch.randperm(len(y))]")["code_labels_permuted"] == 1.0
    assert code_features("perm = torch.randperm(10)\nx, y = x[perm], y[perm]")["code_labels_permuted"] == 0.0
    assert code_features("x = (pixels / 255.0 - 0.5)")["code_scales_inputs"] == 1.0
    assert code_features("this is not python (")["code_parsed"] == 0.0


def test_operators_plant_bugs_in_hand_updated_scripts():
    planted = mutants(MANUAL)
    assert "missing_zero_grad" in planted and "w.grad.zero_()" not in planted["missing_zero_grad"]
    assert "learning_rate_misconfigured" in planted
    lr = ast.parse(planted["learning_rate_misconfigured"])
    value = next(n.value.value for n in ast.walk(lr) if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "lr")
    assert value in (5.0, 5e-06)  # x100, or 10,000 times too small
    scaled = "import torch\npixels = torch.rand(10, 784) * 255\nx_train = pixels / 255.0\n"
    assert "/ 255" not in OPERATORS["non_normalized_inputs"](scaled)


def _report(losses, val=None):
    calls = [{"train": False, "value": v} for v in (val or [])]
    return {"losses": losses, "trace": {"loss_calls": calls}}


def test_a_planted_bug_must_make_the_run_worse():
    healthy = _report([2.0, 1.5, 1.0, 0.5] * 3, val=[0.6, 0.5])
    assert has_effect("label_data_misalignment", _report([2.0] * 12, val=[2.0]), healthy)
    assert not has_effect("learning_rate_misconfigured", _report([2.0, 1.0, 0.5, 0.4] * 3, val=[0.5]), healthy)
    assert has_effect("learning_rate_misconfigured", _report([2.0, float("nan")]), healthy)
    # a summed loss is on another scale: compare how much of it training removed
    assert not has_effect("wrong_loss_reduction", _report([64.0, 48.0, 32.0, 16.0] * 3), healthy)
    assert has_effect("wrong_loss_reduction", _report([64.0, 63.0, 62.0, 61.0] * 3), healthy)
    assert has_effect("train_eval_mode_mismatch", _report([2.0, 1.5, 1.0, 0.5] * 3, val=[0.9, 0.8]), healthy)
    assert has_effect("train_eval_mode_mismatch", _report([1.0] * 4), _report([1.0] * 4))  # no validation: kept


def _trace(code, tmp_path, steps=40):
    pytest.importorskip("torch")
    from silent_bug_agent.mutation.dataset import trace_run

    return trace_run(code, max_steps=steps, timeout=120)


def test_hand_updated_scripts_are_traced_and_stopped(tmp_path):
    report = _trace(MANUAL, tmp_path)
    trace = report["trace"]
    assert report["status"] == "stopped" and report["backward_calls"] == 41  # 40 passes, the 41st stops the run
    assert trace["updates_seen"] >= 35 and trace["stale_after_update"] == 0
    assert len(trace["bw_grad_norms"]) == 40 and trace["bw_step_over_grad"]
    broken = _trace(MANUAL.replace("        w.grad.zero_()\n", ""), tmp_path)["trace"]
    assert broken["stale_after_update"] >= 0.9 * broken["updates_seen"] > 0  # gradients piled up after every update


def test_gradients_cleared_by_setting_them_to_none_are_not_a_missing_zero_grad(tmp_path):
    cleared = _trace(MANUAL.replace("w.grad.zero_()", "w.grad = None"), tmp_path)["trace"]
    assert cleared["stale_after_update"] == 0 and cleared["updates_seen"] > 0
    accumulated = LOOP.replace("        opt.zero_grad()\n        loss", "        loss").replace(
        "        opt.step()\n", "        if i % 32 == 16:\n            opt.step()\n            opt.zero_grad()\n")
    trace = _trace(accumulated, tmp_path)["trace"]
    assert trace["stale_any"] > 0 and trace["stale_after_update"] == 0  # accumulating on purpose is not the bug


def test_colab_scripts_have_valid_edits_and_point_at_the_local_data(tmp_path):
    sys.path.insert(0, str(REPO / "examples" / "colab"))
    import variants

    versions = variants.versions(folder=tmp_path)  # no data there: scripts that need it are left out
    assert versions and all("/content/sample_data" not in v["code"] for v in versions)
    for version in versions:
        ast.parse(version["code"])
    names = {v["script"] for v in versions}
    assert "colab_07_polynomial_fit.py" in names and "colab_01_quickstart_mnist.py" not in names
    splits = {}
    for name, spec in variants.VARIANTS.items():
        splits.setdefault(spec["split"], set()).add(name)
    assert not (splits["dev"] & splits["test"]) and not (set(variants.TRAIN) & (splits["dev"] | splits["test"]))


def test_gate_two_judges_eval_fixes_on_validation_loss():
    from silent_bug_agent.validation.multiseedagent import VALIDATION_METRIC_BUGS, MultiseedAgent
    from silent_bug_agent.validation.sandbox import ExecutionResult

    assert "train_eval_mode_mismatch" in VALIDATION_METRIC_BUGS
    result = ExecutionResult(returncode=0, stdout="", stderr="", run_report={"trace": {"loss_calls": [
        {"train": True, "value": 9.0}, {"train": False, "value": 1.0}, {"train": False, "value": 2.0},
        {"train": False, "value": 3.0}, {"train": False, "value": 4.0}]}})
    assert MultiseedAgent._validation_loss(result) == pytest.approx(3.0)
    assert MultiseedAgent._validation_loss(ExecutionResult(returncode=0, stdout="", stderr="")) is None


def test_compare_versions_reads_a_version_with_arguments():
    sys.path.insert(0, str(REPO / "examples"))
    from compare_versions import parse_version

    name, path, extra = parse_version("v4=.:--learned active")
    assert name == "v4" and path.is_absolute() and extra == ["--learned", "active"]
