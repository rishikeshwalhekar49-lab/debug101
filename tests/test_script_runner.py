"""Measured runs: any script, any seed, a step budget, and one-batch overfitting."""

import textwrap

import pytest

pytest.importorskip("torch")

from silent_bug_agent.validation.overfit_agent import OverfitOracleAgent  # noqa: E402
from silent_bug_agent.validation.sandbox import SubprocessSandbox  # noqa: E402

SCRIPT = textwrap.dedent("""
    import torch
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(STEPS):
        x = torch.randn(16, 4)
        y = x.sum(dim=1, keepdim=True)
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    """)


def run(code, **kwargs):
    return SubprocessSandbox().run_measured(code, timeout=120, **kwargs)


def test_losses_and_steps_are_recorded_without_help_from_the_script():
    result = run(SCRIPT.replace("STEPS", "20"), seed=1)
    report = result.run_report
    assert result.success and report["status"] == "ok"
    assert report["steps"] == 20 and len(report["losses"]) == 20
    assert report["params_trainable"] == 2 and report["params_with_gradient"] == 2


def test_hard_coded_seed_still_varies_with_the_run_seed():
    first = run(SCRIPT.replace("STEPS", "3"), seed=1).run_report["losses"][0]
    second = run(SCRIPT.replace("STEPS", "3"), seed=2).run_report["losses"][0]
    again = run(SCRIPT.replace("STEPS", "3"), seed=1).run_report["losses"][0]
    assert first != second and first == again


def test_step_budget_stops_training():
    assert run(SCRIPT.replace("STEPS", "1000"), seed=0, max_steps=7).run_report["steps"] == 7


def test_overfit_mode_trains_on_one_batch():
    report = run(SCRIPT.replace("STEPS", "150"), seed=0, mode="overfit").run_report
    assert report["loss_ratio"] < 0.05


def test_script_errors_are_reported():
    result = run("raise ValueError('boom')", seed=0)
    assert not result.success and "boom" in result.error_message


def test_overfit_gate_passes_learning_code_and_fails_frozen_code():
    gate = OverfitOracleAgent(steps=80, timeout=120)
    assert gate.check_code(SCRIPT.replace("STEPS", "200"))["passed"] is True
    frozen = SCRIPT.replace("STEPS", "50").replace("lr=0.1", "lr=0.0")
    outcome = gate.check_code(frozen)
    assert outcome["passed"] is False and "loss" in outcome["reason"]
    broken = SCRIPT.replace("STEPS", "50").replace("loss = ((model(x) - y) ** 2).mean()",
                                                   "loss = ((model(x) - y) ** 2).mean() * float('nan')")
    assert "NaN" in gate.check_code(broken)["reason"]
