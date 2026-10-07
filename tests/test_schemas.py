"""Contract validation tests for the multi-agent shared schemas (Task 1.1).

These tests verify that:
- All agent data contracts can be instantiated with minimal args.
- to_dict / from_dict round-trips are lossless.
- AgentState field ownership is correctly typed.
- Stubs for every inter-agent message format are exercised.

Run with:  pytest tests/test_schemas.py -v
"""

from silent_bug_agent.invariants.base import Severity
from silent_bug_agent.findings import Finding, Report
from silent_bug_agent.schemas import (
    AgentState,
    AgentStatus,
    DiagnosticHypothesis,
    PatchCandidate,
    RepairStrategy,
    ValidationVerdict,
)


# ---------------------------------------------------------------------------
# DiagnosticHypothesis
# ---------------------------------------------------------------------------

def _make_hypothesis(**kwargs) -> DiagnosticHypothesis:
    defaults = dict(
        bug_id="BUG_MISSING_ZERO_GRAD",
        description="optimizer.zero_grad() never called; gradients accumulate.",
        confidence=0.92,
        evidence_finding_keys=[("BUG_MISSING_ZERO_GRAD", "train.py", 42)],
        suggested_strategy=RepairStrategy.DIRECT_EDIT,
    )
    defaults.update(kwargs)
    return DiagnosticHypothesis(**defaults)


def test_diagnostic_hypothesis_instantiation():
    h = _make_hypothesis()
    assert h.confidence == 0.92
    assert h.suggested_strategy == RepairStrategy.DIRECT_EDIT


def test_diagnostic_hypothesis_roundtrip():
    h = _make_hypothesis()
    h2 = DiagnosticHypothesis.from_dict(h.to_dict())
    assert h2.bug_id == h.bug_id
    assert h2.confidence == h.confidence
    assert h2.suggested_strategy == h.suggested_strategy
    assert h2.evidence_finding_keys == h.evidence_finding_keys


# ---------------------------------------------------------------------------
# PatchCandidate
# ---------------------------------------------------------------------------

def _make_patch(**kwargs) -> PatchCandidate:
    defaults = dict(
        hypothesis_bug_id="BUG_MISSING_ZERO_GRAD",
        diff="--- a/train.py\n+++ b/train.py\n@@ -40,0 +41 @@\n+    optimizer.zero_grad()\n",
        strategy=RepairStrategy.DIRECT_EDIT,
        source="rules",
        confidence=0.88,
        changed_files=["train.py"],
        changed_lines=1,
    )
    defaults.update(kwargs)
    return PatchCandidate(**defaults)


def test_patch_candidate_instantiation():
    p = _make_patch()
    assert p.source == "rules"
    assert p.strategy == RepairStrategy.DIRECT_EDIT


def test_patch_candidate_roundtrip():
    p = _make_patch()
    p2 = PatchCandidate.from_dict(p.to_dict())
    assert p2.hypothesis_bug_id == p.hypothesis_bug_id
    assert p2.diff == p.diff
    assert p2.strategy == p.strategy
    assert p2.changed_lines == p.changed_lines


# ---------------------------------------------------------------------------
# ValidationVerdict
# ---------------------------------------------------------------------------

def _make_verdict(accepted: bool = True, validator: str = "overfit_oracle", **kwargs) -> ValidationVerdict:
    return ValidationVerdict(
        patch_candidate=_make_patch(),
        accepted=accepted,
        validator=validator,
        p_value=0.03,
        effect_size=0.12,
        seeds_run=3,
        compute_budget_used=0.4,
        **kwargs,
    )


def test_validation_verdict_accepted():
    v = _make_verdict(accepted=True)
    assert v.accepted is True
    assert v.validator == "overfit_oracle"


def test_validation_verdict_rejected():
    v = _make_verdict(accepted=False, validator="multiseed_stats")
    assert v.accepted is False


def test_validation_verdict_roundtrip():
    v = _make_verdict()
    v2 = ValidationVerdict.from_dict(v.to_dict())
    assert v2.accepted == v.accepted
    assert v2.p_value == v.p_value
    assert v2.patch_candidate.diff == v.patch_candidate.diff


# ---------------------------------------------------------------------------
# AgentState — field ownership stubs
# ---------------------------------------------------------------------------

def test_agent_state_initial():
    state = AgentState(run_id="run-001", target_file="train.py")
    assert state.status == AgentStatus.PENDING
    assert state.detection_report is None
    assert state.hypotheses == []
    assert state.patch_candidates == []
    assert state.validation_verdicts == []
    assert state.should_abort() is False


def test_agent_state_person2_writes_report():
    """Stub: Rishikesh's RuntimeMonitorAgent writes detection_report."""
    state = AgentState(run_id="run-001", target_file="train.py")
    finding = Finding(
        bug_id="BUG_DEAD_RELU",
        detector="runtime",
        message="100% dead ReLU units detected in layer fc1.",
        severity=Severity.CRITICAL,
        confidence=0.95,
    )
    state.detection_report = Report(findings=[finding], files_scanned=["train.py"])
    assert len(state.detection_report.findings) == 1
    assert state.detection_report.findings[0].detector == "runtime"


def test_agent_state_person4_writes_hypotheses():
    """Stub: Person 4's DiagnosisAgent appends hypotheses."""
    state = AgentState(run_id="run-001", target_file="train.py")
    state.hypotheses.append(_make_hypothesis(confidence=0.91))
    state.hypotheses.append(_make_hypothesis(bug_id="BUG_LR_TOO_HIGH", confidence=0.55))
    assert state.best_hypothesis.confidence == 0.91


def test_agent_state_person4_writes_patches():
    """Stub: Person 4's FixSynthesizerAgent appends patch candidates."""
    state = AgentState(run_id="run-001", target_file="train.py")
    state.patch_candidates.append(_make_patch())
    assert len(state.patch_candidates) == 1


def test_agent_state_person3_writes_verdicts():
    """Stub: Person 3's validation agents append verdicts."""
    state = AgentState(run_id="run-001", target_file="train.py")
    state.validation_verdicts.append(_make_verdict(accepted=True))
    assert len(state.accepted_patches) == 1


def test_agent_state_supervisor_aborts():
    """Stub: Supervisor sets abort flag when max_iterations exceeded."""
    state = AgentState(run_id="run-001", target_file="train.py", max_iterations=2)
    state.iteration = 2
    assert state.should_abort() is True


def test_agent_state_status_transitions():
    state = AgentState(run_id="run-001", target_file="train.py")
    state.status = AgentStatus.RUNNING
    assert state.status == AgentStatus.RUNNING
    state.status = AgentStatus.DONE
    assert state.status == AgentStatus.DONE
