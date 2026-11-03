from pathlib import Path

from silent_bug_agent.schemas import (
    AgentState,
    DiagnosticHypothesis,
    RepairStrategy,
)
from silent_bug_agent.agents.fix_synthesizer_agent import FixSynthesizerAgent
from silent_bug_agent.learning.bandit import ContextualBandit
from silent_bug_agent.repair.loop import AutonomousRepairLoop


def test_synthesizer_generates_patch_without_modifying_source(tmp_path):
    target = tmp_path / "example.py"
    original = "learning_rate = 0.1\n"
    target.write_text(original, encoding="utf-8")

    hypothesis = DiagnosticHypothesis(
        bug_id="TEST_LR",
        description="Use the specified learning rate.",
        confidence=0.9,
        suggested_strategy=RepairStrategy.DIRECT_EDIT,
        extra={
            "repair_spec": {
                "search": "learning_rate = 0.1",
                "replacement": "learning_rate = 0.01",
            }
        },
    )
    state = AgentState(target_file=str(target), hypotheses=[hypothesis])

    FixSynthesizerAgent().run(state)

    assert len(state.patch_candidates) == 1
    assert "learning_rate = 0.01" in state.patch_candidates[0].diff
    assert target.read_text(encoding="utf-8") == original


def test_synthesizer_rejects_ambiguous_edit(tmp_path):
    target = tmp_path / "example.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")

    hypothesis = DiagnosticHypothesis(
        bug_id="TEST_AMBIGUOUS",
        description="Change x.",
        confidence=0.8,
        extra={
            "repair_spec": {
                "search": "x = 1",
                "replacement": "x = 2",
            }
        },
    )
    state = AgentState(target_file=str(target), hypotheses=[hypothesis])

    FixSynthesizerAgent().run(state)

    assert state.patch_candidates == []
    assert "found 2" in state.extra["repair_error"]


def test_bandit_records_and_selects_valid_strategy():
    bandit = ContextualBandit()
    extra = {}
    hypothesis = DiagnosticHypothesis(
        bug_id="TEST_BANDIT",
        description="Test strategy selection.",
        confidence=0.8,
        suggested_strategy=RepairStrategy.DIRECT_EDIT,
    )

    bandit.record_reward(extra, RepairStrategy.DIRECT_EDIT, 1.0)
    bandit.record_reward(extra, RepairStrategy.REFACTOR, -1.0)

    selected = bandit.select_strategy(
        hypothesis,
        extra,
        available=[RepairStrategy.DIRECT_EDIT, RepairStrategy.REFACTOR],
    )

    assert selected in [RepairStrategy.DIRECT_EDIT, RepairStrategy.REFACTOR]
    assert extra["repair_bandit"]["direct_edit"]["count"] == 1.0
    assert extra["repair_bandit"]["refactor"]["count"] == 1.0


def test_repair_loop_validates_candidate_without_changing_original(tmp_path):
    target = tmp_path / "example.py"
    original = "learning_rate = 0.1\n"
    target.write_text(original, encoding="utf-8")

    hypothesis = DiagnosticHypothesis(
        bug_id="TEST_LOOP",
        description="Correct the learning rate.",
        confidence=0.9,
        suggested_strategy=RepairStrategy.DIRECT_EDIT,
        extra={
            "repair_spec": {
                "search": "learning_rate = 0.1",
                "replacement": "learning_rate = 0.01",
            }
        },
    )
    state = AgentState(target_file=str(target), hypotheses=[hypothesis])
    loop = AutonomousRepairLoop(FixSynthesizerAgent())

    loop.run(state)

    assert len(state.patch_candidates) == 1
    assert len(state.validation_verdicts) == 1
    assert state.validation_verdicts[0].accepted is False
    assert state.validation_verdicts[0].extra["patch_applied_to_temp"] is True
    assert target.read_text(encoding="utf-8") == original
