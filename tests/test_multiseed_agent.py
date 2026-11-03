"""Unit and integration tests for MultiseedAgent and statistical validation."""

import textwrap
import pytest

from silent_bug_agent.schemas import (
    AgentState,
    AgentStatus,
    PatchCandidate,
    RepairStrategy,
    ValidationVerdict,
)
from silent_bug_agent.validation.multiseedagent import (
    MultiseedAgent,
    apply_patch_string,
    compare_distributions,
    compute_cohens_d,
    compute_confidence_interval,
)
from silent_bug_agent.validation.sandbox import ExecutionResult, SubprocessSandbox


def _make_candidate(diff: str = "", bug_id: str = "BUG_MISSING_ZERO_GRAD") -> PatchCandidate:
    return PatchCandidate(
        hypothesis_bug_id=bug_id,
        diff=diff,
        strategy=RepairStrategy.DIRECT_EDIT,
        source="rules",
        confidence=0.9,
        changed_files=["train.py"],
        changed_lines=1,
    )


# ---------------------------------------------------------------------------
# Statistical Tools Tests
# ---------------------------------------------------------------------------

def test_confidence_interval_basic():
    diffs = [1.0, 1.2, 0.9, 1.1, 1.0]
    ci_low, ci_high = compute_confidence_interval(diffs, confidence=0.95)
    assert ci_low < ci_high
    assert ci_low > 0.0  # consistently positive improvement


def test_confidence_interval_single_sample():
    diffs = [2.5]
    ci_low, ci_high = compute_confidence_interval(diffs, confidence=0.95)
    assert ci_low == 2.5
    assert ci_high == 2.5


def test_cohens_d_effect_size():
    # Large improvement
    b = [10.0, 10.5, 9.8, 10.2, 10.1]
    p = [2.0, 2.1, 1.9, 2.2, 2.0]
    d = compute_cohens_d(b, p)
    assert d > 5.0  # huge effect size


def test_compare_distributions_detects_significant_fix():
    # Baseline has consistently high loss, patched has consistently low loss
    baseline = [4.5, 4.8, 4.3, 4.6, 4.7]
    patched = [0.2, 0.3, 0.25, 0.28, 0.22]
    res = compare_distributions(baseline, patched, alpha=0.05)
    assert res.is_significant
    assert res.p_value < 0.001
    assert res.mean_difference > 4.0
    assert res.confidence_interval_95[0] > 0.0


def test_compare_distributions_rejects_random_seed_luck():
    # Difference is small noise around zero across seeds
    baseline = [1.05, 0.98, 1.02, 1.01, 0.99]
    patched = [1.04, 1.01, 1.00, 0.98, 1.02]  # basically the same
    res = compare_distributions(baseline, patched, alpha=0.05)
    assert not res.is_significant
    assert res.p_value >= 0.05


def test_compare_distributions_rejects_regression():
    # Patched is worse than baseline
    baseline = [1.0, 1.1, 1.05, 0.95, 1.0]
    patched = [3.0, 3.2, 2.9, 3.1, 3.0]
    res = compare_distributions(baseline, patched, alpha=0.05)
    assert not res.is_significant
    assert res.mean_difference < 0.0


def test_compare_distributions_ks_test():
    baseline = [5.0, 5.2, 4.9, 5.1, 5.0]
    patched = [1.0, 1.1, 0.9, 1.0, 1.2]
    res = compare_distributions(baseline, patched, alpha=0.05, prefer_test="ks_test")
    assert res.is_significant
    assert res.test_name == "kolmogorov_smirnov"


# ---------------------------------------------------------------------------
# Patch String Application Tests
# ---------------------------------------------------------------------------

def test_apply_patch_string_simple():
    original = "x = 1\ny = 2\n"
    diff = "--- a/test.py\n+++ b/test.py\n@@ -2,1 +2,1 @@\n-y = 2\n+y = 20\n"
    candidate = _make_candidate(diff=diff)
    patched = apply_patch_string(original, candidate)
    assert "y = 20" in patched


def test_apply_patch_string_object_with_apply():
    class CustomPatch:
        def apply(self, src):
            return src.replace("foo", "bar")

    res = apply_patch_string("test foo baz", CustomPatch())
    assert res == "test bar baz"


# ---------------------------------------------------------------------------
# MultiseedAgent Sandboxed Verification Tests
# ---------------------------------------------------------------------------

def test_multiseed_agent_verify_genuine_patch(tmp_path):
    target = tmp_path / "script.py"
    target.write_text(textwrap.dedent("""
        import os
        import sys
        # Emulates bug where step is unoptimized unless fixed
        flag = False
        loss = 2.5
        print(f'__METRICS__:{{"final_loss": {loss}}}')
    """))

    candidate = _make_candidate(diff="""--- a/script.py
+++ b/script.py
@@ -5,2 +5,2 @@
-flag = False
-loss = 2.5
+flag = True
+loss = 0.2
""")

    agent = MultiseedAgent(seeds=[42, 1337, 2024], timeout_per_run=10.0)
    verdict = agent.verify_patch(baseline_code=target.read_text(), candidate=candidate)

    assert verdict.accepted is True
    assert verdict.validator == "multiseed_stats"
    assert verdict.p_value < 0.05
    assert verdict.seeds_run == 3
    assert verdict.effect_size > 0


def test_multiseed_agent_rejects_failing_patch():
    baseline_code = """
loss = 1.0
print(f'__METRICS__:{"final_loss": 1.0}')
"""
    # Patch introduces a crash
    candidate = _make_candidate(diff="""--- a/script.py
+++ b/script.py
@@ -2,1 +2,1 @@
-loss = 1.0
+raise RuntimeError("Syntax/runtime error introduced by bad patch")
""")

    agent = MultiseedAgent(seeds=[42, 1337, 2024], timeout_per_run=5.0)
    verdict = agent.verify_patch(baseline_code=baseline_code, candidate=candidate)

    assert verdict.accepted is False
    assert "failed" in verdict.notes.lower()
    assert verdict.p_value == 1.0


# ---------------------------------------------------------------------------
# AgentState & Supervisor Integration Tests
# ---------------------------------------------------------------------------

def test_multiseed_agent_call_contract(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("loss = 2.0\nprint(f'__METRICS__:{{\"final_loss\": {loss}}}')\n", encoding="utf-8")
    candidate = _make_candidate(diff="""--- a/train.py
+++ b/train.py
@@ -1,1 +1,1 @@
-loss = 2.0
+loss = 1.0
""")
    state = AgentState(
        run_id="run-test-01",
        target_file=str(script),
        patch_candidates=[candidate],
    )

    agent = MultiseedAgent(seeds=[42, 1337, 2024])
    updated_state = agent.call(state)

    assert len(updated_state.validation_verdicts) == 1
    v = updated_state.validation_verdicts[0]
    assert v.validator == "multiseed_stats"
    assert v.patch_candidate == candidate
    assert v.seeds_run == 3


def test_multiseed_agent_without_a_script_never_accepts():
    candidate = _make_candidate()
    state = AgentState(run_id="run-test-00", target_file="", patch_candidates=[candidate])
    verdict = MultiseedAgent(seeds=[42, 1337, 2024]).call(state).validation_verdicts[0]
    assert verdict.accepted is False and verdict.seeds_run == 0


def test_multiseed_agent_respects_overfit_oracle_rejection():
    candidate = _make_candidate()
    state = AgentState(
        run_id="run-test-02",
        target_file="",
        patch_candidates=[candidate],
        validation_verdicts=[
            # OverfitOracle ran first and rejected candidate
            ValidationVerdict(
                patch_candidate=candidate,
                accepted=False,
                validator="overfit_oracle",
                notes="Model failed one-batch overfit check.",
            )
        ],
    )

    agent = MultiseedAgent()
    updated_state = agent.call(state)

    # Agent appends its verdict reflecting early rejection to conserve budget
    assert len(updated_state.validation_verdicts) == 2
    ms_verdict = updated_state.validation_verdicts[1]
    assert ms_verdict.validator == "multiseed_stats"
    assert ms_verdict.accepted is False
    assert ms_verdict.seeds_run == 0
    assert "Overfit Oracle rejected" in ms_verdict.notes


def test_multiseed_agent_callable_interface():
    agent = MultiseedAgent()
    state = AgentState(run_id="call-test")
    # Verify __call__ works identically to call
    out = agent(state)
    assert out.run_id == "call-test"


def test_supervisor_agent_validation_pipeline_registration(tmp_path):
    from silent_bug_agent.supervisor import SupervisorAgent
    from silent_bug_agent.validation.overfit_agent import OverfitOracleAgent

    torch = pytest.importorskip("torch")
    script = tmp_path / "train.py"
    script.write_text(textwrap.dedent("""
        import torch
        model = torch.nn.Linear(4, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        for step in range(40):
            x = torch.randn(32, 4)
            opt.zero_grad()
            loss = ((model(x) - x.sum(dim=1, keepdim=True)) ** 2).mean()
            loss.backward()
            opt.step()
        print(f'__METRICS__:{{"final_loss": {loss.item()}}}')
        """), encoding="utf-8")
    candidate = _make_candidate(diff="""--- a/train.py
+++ b/train.py
@@ -3,3 +3,3 @@
 model = torch.nn.Linear(4, 1)
-opt = torch.optim.SGD(model.parameters(), lr=0.0)
+opt = torch.optim.SGD(model.parameters(), lr=0.1)
 for step in range(40):
""")

    # Register as SupervisorAgent(validationagents=[OverfitOracleAgent(), MultiseedAgent()])
    # exactly as specified in the brief's contract
    supervisor = SupervisorAgent(validationagents=[OverfitOracleAgent(steps=40), MultiseedAgent(seeds=[42, 1337, 2024])])
    state = AgentState(run_id="sup-test-01", target_file=str(script), patch_candidates=[candidate])

    result_state = supervisor.run(state)
    assert len(result_state.validation_verdicts) == 2
    # First verdict is from OverfitOracle
    assert result_state.validation_verdicts[0].validator == "overfit_oracle"
    # Second verdict is from MultiseedAgent
    assert result_state.validation_verdicts[1].validator == "multiseed_stats"
    assert result_state.status == AgentStatus.DONE


def test_supervisor_rejects_unknown_arguments():
    from silent_bug_agent.supervisor import SupervisorAgent

    with pytest.raises(TypeError):
        SupervisorAgent(detection_agent=object())
