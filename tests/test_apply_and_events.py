"""Approval is real: `fix --dry-run` writes nothing, `apply` writes only the validated patch,
and the event stream the TUI reads says exactly what happened."""

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from silent_bug_agent.agents import supervisor as supervisor_module  # noqa: E402
from silent_bug_agent.agents.supervisor import PipelineConfig, SupervisorAgent, describe_state  # noqa: E402
from silent_bug_agent.cli import main  # noqa: E402
from silent_bug_agent.learning.experience_store import ExperienceStore  # noqa: E402
from silent_bug_agent.repair.patch_utils import generate_minimal_diff  # noqa: E402
from silent_bug_agent.repair.pending import (PendingPatchError, apply_pending, bundled_example,  # noqa: E402
                                             load_pending, save_pending)

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
    assert accepted["applied"] is False and accepted["pending"] is True and accepted["blocked"] is None
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


def test_bundled_examples_are_recognised_and_never_applied_to(tmp_path):
    bundled = SAMPLES / BENCHMARK
    assert bundled_example(bundled) and bundled_example(SAMPLES.parent / "planted_bugs_torch.py")
    assert not bundled_example(copy(tmp_path))
    before = bundled.read_bytes()
    with pytest.raises(PendingPatchError, match="bundled examples"):
        apply_pending(bundled)
    assert bundled.read_bytes() == before


def test_the_pipeline_does_not_edit_a_protected_file(tmp_path, monkeypatch):
    """`fix` without --dry-run on a bundled example validates a patch but leaves the file as it is."""
    script = copy(tmp_path, "benchmark_01_missing_zero_grad.py")
    before = script.read_text(encoding="utf-8")
    monkeypatch.setattr(supervisor_module, "bundled_example", lambda path: True)
    config = PipelineConfig(offline=True, seeds=(42, 1337, 2024), monitor_steps=60, validation_steps=40,
                            overfit_steps=60, timeout_per_run=120, use_vectors=False)
    states = SupervisorAgent(config, memory=ExperienceStore(tmp_path / "memory", use_vectors=False)
                             ).run_until_clean(str(script), max_rounds=3)
    info = describe_state(states[-1])
    assert len(states) == 1 and info["outcome"] == "fixed" and info["accepted_diff"]
    assert not info["patch_applied"] and "bundled examples" in info["apply_blocked"]
    assert script.read_text(encoding="utf-8") == before and not list(tmp_path.glob("*.orig"))
