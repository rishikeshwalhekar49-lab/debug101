"""Approval is real: `fix --dry-run` writes nothing, `apply` writes only the validated patch,
`fix --another` looks for a different one, and the event stream the TUI reads says what happened."""

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from silent_bug_agent.agents import supervisor as supervisor_module  # noqa: E402
from silent_bug_agent.agents.fix_synthesizer_agent import FixSynthesizerAgent  # noqa: E402
from silent_bug_agent.agents.supervisor import PipelineConfig, SupervisorAgent, describe_state  # noqa: E402
from silent_bug_agent.cli import main  # noqa: E402
from silent_bug_agent.learning.experience_store import ExperienceStore  # noqa: E402
from silent_bug_agent.repair.patch_utils import generate_minimal_diff  # noqa: E402
from silent_bug_agent.repair.pending import (PendingPatchError, apply_pending, bundled_example,  # noqa: E402
                                             load_pending, load_rejected, reject_pending, save_pending)
from silent_bug_agent.repair.recipes import recipe_for  # noqa: E402
from silent_bug_agent.schemas import AgentState, DiagnosticHypothesis  # noqa: E402
from silent_bug_agent.semantic.client import FakeLLMClient  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "examples" / "sample_scripts"
BENCHMARK = "benchmark_04_softmax_before_crossentropy.py"


def copy(tmp_path, name=BENCHMARK):
    target = tmp_path / name
    shutil.copy(SAMPLES / name, target)
    return target


def events_of(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def test_dry_run_then_apply_writes_exactly_the_validated_patch(tmp_path, capsys, monkeypatch):
    # the CLI builds its own memory; keep it off ChromaDB so the test never downloads an embedding model
    monkeypatch.setattr(ExperienceStore, "_vectors", lambda self: None)
    script = copy(tmp_path)
    original = script.read_text(encoding="utf-8")

    assert main(["fix", str(script), "--dry-run", "--no-llm", "--seeds", "3", "--events"]) == 0
    events = events_of(capsys.readouterr().out)
    counts = Counter(e["event"] for e in events)
    result = next(e for e in events if e["event"] == "result")["data"]["runs"][0]

    # nothing was written, and the stream says so
    assert script.read_text(encoding="utf-8") == original and not list(tmp_path.glob("*.orig"))
    accepted = next(e for e in events if e["event"] == "accepted_patch")
    assert accepted["applied"] is False and accepted["pending"] is True and accepted["copy_to"] is None
    assert accepted["bug_id"] == "softmax_before_cross_entropy" and accepted["source"] == "rules"
    assert accepted["diff"] == result["accepted_diff"]

    # each hypothesis, candidate and verdict is sent once, with real values
    assert counts["hypothesis"] == len(result["hypotheses"])
    assert counts["verdict"] == len(result["attempts"]) and counts["accepted_patch"] == 1
    assert all(e["validator"] for e in events if e["event"] == "verdict")
    assert all(e["bug_id"] and e["strategy"] != "unknown" for e in events if e["event"] == "patch")
    assert events[-1] == {**events[-1], "event": "done", "exit_code": 0}

    # apply writes that patch, keeps a backup, and can only be used once
    assert main(["apply", str(script), "--events"]) == 0
    applied = next(e for e in events_of(capsys.readouterr().out) if e["event"] == "applied")
    assert "torch.softmax(out, dim=-1)" not in script.read_text(encoding="utf-8")
    assert applied["copy"] is False and Path(applied["written"]) == script.resolve()
    assert Path(applied["backup"]).read_text(encoding="utf-8") == original
    assert main(["apply", str(script)]) == 1
    assert "no validated patch" in capsys.readouterr().err


def test_apply_refuses_a_patch_for_a_file_that_changed(tmp_path, capsys):
    script = copy(tmp_path)
    source = script.read_text(encoding="utf-8")
    diff = generate_minimal_diff(source, source.replace("torch.softmax(out, dim=-1)", "out"), script.name)
    save_pending(script, diff, bug_id="softmax_before_cross_entropy")
    script.write_text(source + "\n# edited by hand after validation\n", encoding="utf-8")
    edited = script.read_text(encoding="utf-8")

    assert main(["apply", str(script)]) == 1
    assert "has changed since the patch was validated" in capsys.readouterr().err
    assert script.read_text(encoding="utf-8") == edited and load_pending(script) is None


def test_apply_without_a_validated_patch_does_nothing(tmp_path, capsys):
    script = copy(tmp_path)
    before = script.read_text(encoding="utf-8")
    assert main(["apply", str(script)]) == 1
    assert script.read_text(encoding="utf-8") == before
    assert main(["apply", str(tmp_path / "missing.py")]) == 2


def recipe_diff(script, bug="softmax_before_cross_entropy"):
    source = script.read_text(encoding="utf-8")
    return generate_minimal_diff(source, recipe_for(bug, source).source, script.name)


def test_applying_to_a_bundled_example_writes_a_copy_and_leaves_it_alone(tmp_path, monkeypatch, capsys):
    bundled = SAMPLES / BENCHMARK
    assert bundled_example(bundled) and bundled_example(SAMPLES.parent / "planted_bugs_torch.py")
    assert not bundled_example(copy(tmp_path))
    before = bundled.read_bytes()
    monkeypatch.chdir(tmp_path)  # the copy is written beside where the command runs

    with pytest.raises(PendingPatchError, match="no validated patch"):
        apply_pending(bundled)
    save_pending(bundled, recipe_diff(bundled), bug_id="softmax_before_cross_entropy")
    assert main(["apply", str(bundled), "--events"]) == 0
    applied = next(e for e in events_of(capsys.readouterr().out) if e["event"] == "applied")

    written = tmp_path / "fixed_examples" / BENCHMARK
    assert applied["copy"] is True and Path(applied["written"]) == written.resolve() and applied["backup"] is None
    assert "torch.softmax(out, dim=-1)" not in written.read_text(encoding="utf-8")
    assert bundled.read_bytes() == before and not bundled.with_name(BENCHMARK + ".orig").exists()
    assert load_pending(bundled) is None


def test_the_pipeline_never_edits_a_bundled_example_but_keeps_the_patch_for_apply(tmp_path, monkeypatch):
    """`fix` without --dry-run on a bundled example validates a patch, leaves the file, and offers a copy."""
    script = copy(tmp_path, "benchmark_01_missing_zero_grad.py")
    before = script.read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(supervisor_module, "bundled_example", lambda path: True)
    config = PipelineConfig(offline=True, seeds=(42, 1337, 2024), monitor_steps=60, validation_steps=40,
                            overfit_steps=60, timeout_per_run=120, use_vectors=False)
    states = SupervisorAgent(config, memory=ExperienceStore(tmp_path / "memory", use_vectors=False)
                             ).run_until_clean(str(script), max_rounds=3)
    info = describe_state(states[-1])
    assert len(states) == 1 and info["outcome"] == "fixed" and info["accepted_diff"]
    assert not info["patch_applied"] and info["pending_patch"]
    assert Path(info["copy_to"]) == (tmp_path / "fixed_examples" / script.name).resolve()
    assert script.read_text(encoding="utf-8") == before and not list(tmp_path.glob("*.orig"))
    assert load_pending(script)["diff"] == info["accepted_diff"]


# ------------------------------------------------------------- try another fix
def test_another_fix_never_offers_the_declined_patch_again(tmp_path, capsys):
    """Offline there is one rule-based fix per bug, so turning it down leaves nothing, and says so."""
    script = copy(tmp_path)
    before = script.read_text(encoding="utf-8")
    declined = recipe_diff(script)
    save_pending(script, declined, bug_id="softmax_before_cross_entropy")

    assert main(["fix", str(script), "--dry-run", "--another", "--no-llm", "--seeds", "3", "--events"]) == 1
    events = events_of(capsys.readouterr().out)
    assert not [e for e in events if e["event"] in ("accepted_patch", "patch")]
    note = next(e for e in events if e["event"] == "result")["data"]["runs"][-1]["another_note"]
    assert "No other fix passed validation" in note and "Without an LLM key" in note
    assert script.read_text(encoding="utf-8") == before
    assert load_pending(script) is None and [p["diff"] for p in load_rejected(script)] == [declined]
    assert main(["apply", str(script)]) == 1  # the declined patch can no longer be applied by accident

    # an ordinary run starts fresh, so the first patch is offered again
    assert main(["fix", str(script), "--dry-run", "--no-llm", "--seeds", "3", "--events"]) == 0
    accepted = next(e for e in events_of(capsys.readouterr().out) if e["event"] == "accepted_patch")
    assert accepted["diff"] == declined and load_rejected(script) == []


def test_another_fix_needs_an_earlier_patch_to_replace(tmp_path, capsys):
    script = copy(tmp_path)
    assert main(["fix", str(script), "--dry-run", "--another", "--no-llm"]) == 2
    assert "no earlier validated patch" in capsys.readouterr().err


def test_declined_patches_are_forgotten_when_the_file_changes(tmp_path):
    script = copy(tmp_path)
    save_pending(script, recipe_diff(script), bug_id="softmax_before_cross_entropy")
    assert len(reject_pending(script)) == 1 and len(load_rejected(script)) == 1
    script.write_text(script.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert load_rejected(script) == []


def _fix_agent_run(tmp_path, reply, declined_sources):
    """One repair step with a model that answers `reply`, after the user declined `declined_sources`."""
    script = copy(tmp_path)
    source = script.read_text(encoding="utf-8")
    state = AgentState(target_file=str(script), hypotheses=[DiagnosticHypothesis(
        bug_id="softmax_before_cross_entropy", description="softmax is applied before CrossEntropyLoss", confidence=0.9)])
    state.extra["tried_patches"] = [FixSynthesizerAgent._digest(text(source)) for text in declined_sources]
    agent = FixSynthesizerAgent(client=FakeLLMClient(responses=[json.dumps(reply)]))
    return agent.run(state), source


def test_the_model_is_asked_for_a_different_patch_and_a_repeat_is_refused(tmp_path):
    line = "            return torch.softmax(out, dim=-1)"
    rule_fix = lambda source: recipe_for("softmax_before_cross_entropy", source).source  # noqa: E731

    # a different edit from the model is accepted as the new candidate
    other = {"thought": "t", "strategy": "direct_edit", "confidence": 0.8, "explanation": "return the logits",
             "edits": [{"search": line, "replacement": "            return out  # raw logits for CrossEntropyLoss"}]}
    state, source = _fix_agent_run(tmp_path, other, [rule_fix])
    assert len(state.patch_candidates) == 1 and state.patch_candidates[0].source == "llm"
    assert "raw logits" in state.patch_candidates[0].diff

    # the model repeating the declined patch is refused, and the rule gives the same patch, so nothing is offered
    same = dict(other, edits=[{"search": line, "replacement": "            return out"}])
    assert rule_fix(source).count("            return out\n") >= 1
    state, _ = _fix_agent_run(tmp_path, same, [rule_fix])
    assert not state.patch_candidates and "already tried" in json.dumps(state.extra.get("repair_feedback"))
