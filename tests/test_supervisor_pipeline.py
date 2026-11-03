"""The LangGraph supervisor end to end: offline, and live against a fake provider."""

import json
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("langgraph")

from fake_llm_server import FakeProvider, chat_reply  # noqa: E402

from silent_bug_agent.agents.supervisor import PipelineConfig, SupervisorAgent, describe_state  # noqa: E402
from silent_bug_agent.learning.experience_store import ExperienceStore  # noqa: E402
from silent_bug_agent.schemas import AgentStatus  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "examples" / "sample_scripts"
FAST = dict(seeds=(42, 1337, 2024), monitor_steps=60, validation_steps=40, overfit_steps=60, timeout_per_run=120,
            use_vectors=False)


def copy(tmp_path, name):
    target = tmp_path / name
    shutil.copy(SAMPLES / name, target)
    return target


def supervisor(tmp_path, **options):
    config = PipelineConfig(**{**FAST, **options})
    return SupervisorAgent(config, memory=ExperienceStore(tmp_path / "memory", use_vectors=False))


def test_offline_fix_is_validated_and_applied_with_a_backup(tmp_path):
    script = copy(tmp_path, "benchmark_01_missing_zero_grad.py")
    state = supervisor(tmp_path, offline=True).run(str(script))
    info = describe_state(state)
    assert state.status == AgentStatus.DONE and info["outcome"] == "fixed"
    # with only 3 seeds a real fix may miss p < 0.05; it then passes because the bug is gone and nothing got worse
    assert info["attempts"][-1]["accepted"] and info["attempts"][-1]["route"] in ("statistics", "bug_removed")
    assert "optimizer.zero_grad()" in info["accepted_diff"]
    assert Path(info["backup"]).read_text(encoding="utf-8") == (SAMPLES / script.name).read_text(encoding="utf-8")
    assert script.read_text(encoding="utf-8") != (SAMPLES / script.name).read_text(encoding="utf-8")
    assert info["llm"]["mode"] == "offline" and info["llm"]["calls"] == 0


def test_dry_run_leaves_the_file_alone(tmp_path):
    script = copy(tmp_path, "benchmark_03_frozen_layer.py")
    before = script.read_text(encoding="utf-8")
    state = supervisor(tmp_path, offline=True, dry_run=True).run(str(script))
    assert describe_state(state)["outcome"] == "fixed"
    assert script.read_text(encoding="utf-8") == before


def test_healthy_script_is_left_alone(tmp_path):
    script = copy(tmp_path, "benchmark_01_missing_zero_grad.py")
    state = supervisor(tmp_path, offline=True, script_args=["--fix"]).run(str(script))
    assert describe_state(state)["outcome"] == "no_bug_found"
    assert not list(tmp_path.glob("*.orig"))


def test_a_script_that_crashes_is_reported_not_patched(tmp_path):
    script = tmp_path / "broken.py"
    script.write_text("import torch\nraise RuntimeError('data file missing')\n", encoding="utf-8")
    state = supervisor(tmp_path, offline=True).run(str(script))
    assert state.status == AgentStatus.FAILED
    assert "data file missing" in state.abort_reason or describe_state(state)["outcome"] in ("no_bug_found", "script_error")


def _llm_handler(source_text):
    """Answers like a real model would, keyed on which agent is asking."""
    def handler(body):
        system = body["messages"][0]["content"] if body["messages"][0]["role"] == "system" else ""
        if "Perception Agent" in system:
            return chat_reply(json.dumps({"summary": "Gradients are never cleared.",
                                          "primary_bug_ids": ["missing_zero_grad"], "side_effect_bug_ids": []}))
        if "Diagnosis Agent" in system:
            return chat_reply(json.dumps({"reasoning": "No zero_grad in the loop.", "ranked": [
                {"bug_id": "missing_zero_grad", "confidence": 0.95, "explanation": "grads accumulate", "evidence_line": 46}]}))
        if "Fix Synthesizer" in system:
            return chat_reply(json.dumps({"thought": "Clear the gradients each step.", "strategy": "direct_edit",
                                          "edits": [{"search": "        pred = model(x)\n",
                                                     "replacement": "        optimizer.zero_grad()\n        pred = model(x)\n"}],
                                          "confidence": 0.9, "explanation": "add zero_grad"}))
        if "Validation Analyzer" in system:
            return chat_reply(json.dumps({"verdict": "true_fix", "reason": "large consistent gap"}))
        if "code reviewer" in system:
            return chat_reply("[]")
        return chat_reply("{}")
    return handler


def test_live_pipeline_uses_the_models_for_every_agent(tmp_path, monkeypatch):
    script = copy(tmp_path, "benchmark_01_missing_zero_grad.py")
    server = FakeProvider(_llm_handler(script.read_text(encoding="utf-8")))
    try:
        monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
        monkeypatch.setenv("GROQ_BASE_URL", server.url)
        state = supervisor(tmp_path, dry_run=True).run(str(script))
    finally:
        server.close()
    info = describe_state(state)
    assert info["outcome"] == "fixed"
    assert info["hypotheses"][0]["diagnosed_by"] == "causal_engine+llm"
    assert info["attempts"][-1]["source"] == "llm" and info["attempts"][-1]["accepted"]
    systems = [r["body"]["messages"][0]["content"] for r in server.requests]
    for agent in ("code reviewer", "Perception Agent", "Diagnosis Agent", "Fix Synthesizer", "Validation Analyzer"):
        assert any(agent in s for s in systems), agent
    models = {r["body"]["model"] for r in server.requests}
    assert models == {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}
    assert info["llm"]["mode"] == "live" and info["llm"]["calls"] >= 5


def test_a_harmful_model_patch_is_rejected_and_the_loop_recovers(tmp_path, monkeypatch):
    script = copy(tmp_path, "benchmark_01_missing_zero_grad.py")

    def handler(body):
        system = body["messages"][0]["content"]
        if "Fix Synthesizer" in system:  # a high-confidence patch that only slows learning down
            return chat_reply(json.dumps({"thought": "t", "strategy": "hyperparam",
                                          "edits": [{"search": "lr=0.01", "replacement": "lr=0.0000001"}],
                                          "confidence": 0.99, "explanation": "bad"}))
        return _llm_handler("")(body)

    server = FakeProvider(handler)
    try:
        monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
        monkeypatch.setenv("GROQ_BASE_URL", server.url)
        state = supervisor(tmp_path, dry_run=True, max_iterations=4).run(str(script))
    finally:
        server.close()
    info = describe_state(state)
    rejected = [a for a in info["attempts"] if not a["accepted"]]
    assert rejected and rejected[0]["source"] == "llm"
    assert info["outcome"] == "fixed" and info["attempts"][-1]["source"] == "rules"
    assert "optimizer.zero_grad()" in info["accepted_diff"]
