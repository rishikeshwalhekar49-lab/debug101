"""Experience memory (SQLite, with a word-overlap fallback) and the Meta-Evolution agent."""

import pytest

from silent_bug_agent.agents.meta_evolution import MetaEvolutionAgent
from silent_bug_agent.learning.experience_store import ExperienceStore
from silent_bug_agent.semantic.client import FakeLLMClient


def store(tmp_path):
    return ExperienceStore(tmp_path / "memory", use_vectors=False)


def attempt(memory, bug="missing_zero_grad", accepted=True, used=(), notes="", strategy="direct_edit"):
    return memory.remember({"bug_id": bug, "category": "gradient_management_error", "strategy": strategy,
                            "accepted": accepted, "reward": 1.0 if accepted else -0.5, "summary": f"fix {bug}",
                            "diff": "+    optimizer.zero_grad()\n", "notes": notes, "exemplar_ids": list(used)})


def test_only_accepted_repairs_are_recalled(tmp_path):
    memory = store(tmp_path)
    good = attempt(memory)
    attempt(memory, accepted=False)
    found = memory.recall("missing_zero_grad gradients accumulate", k=3)
    assert [f["id"] for f in found] == [good]


def test_bandit_statistics_persist(tmp_path):
    memory = store(tmp_path)
    memory.update_bandit("ctx", "direct_edit", 1.0)
    memory.update_bandit("ctx", "direct_edit", -0.5)
    again = ExperienceStore(tmp_path / "memory", use_vectors=False)
    assert again.load_bandit("ctx") == {"direct_edit": {"count": 2.0, "reward_sum": 0.5}}


def test_meta_prunes_an_exemplar_that_hurts(tmp_path):
    memory = store(tmp_path)
    harmful = attempt(memory)
    for _ in range(3):
        attempt(memory, accepted=False, used=[harmful], notes="Gate 2 failed: no improvement")
    for _ in range(3):
        attempt(memory, accepted=True)
    report = MetaEvolutionAgent(memory).run()
    assert harmful in report.pruned_exemplars
    assert harmful not in [r["id"] for r in memory.recall("missing_zero_grad", k=10)]


def test_meta_writes_lessons_with_an_llm_and_without(tmp_path):
    memory = store(tmp_path)
    attempt(memory, accepted=False, notes="Gate 1 failed: loss did not fall")
    report = MetaEvolutionAgent(memory).run()
    assert report.lesson_source == "rules" and memory.active_lessons("fix")
    llm = FakeLLMClient(responses=['{"reasoning": "r", "lessons": ["Prefer the smallest edit."]}'])
    attempt(memory, accepted=False, notes="patch too large")
    report = MetaEvolutionAgent(memory, client=llm).run()
    assert report.lesson_source == "llm" and memory.active_lessons("fix") == ["Prefer the smallest edit."]


def test_meta_rolls_back_lessons_that_made_things_worse(tmp_path):
    memory = store(tmp_path)
    memory.set_lessons("fix", ["old"], "rules")
    for accepted in [True] * 5:
        memory.score_lessons("fix", accepted)
    memory.set_lessons("fix", ["new"], "llm")
    for accepted in [False] * 5:
        memory.score_lessons("fix", accepted)
    attempt(memory)
    assert MetaEvolutionAgent(memory).run().rolled_back is True
    assert memory.active_lessons("fix") == ["old"]


def test_chroma_vectors_when_available(tmp_path):
    pytest.importorskip("chromadb")
    memory = ExperienceStore(tmp_path / "vectors", use_vectors=True)
    attempt(memory, bug="dead_relu")
    found = memory.recall("relu units output zero", k=1)
    if memory.notes:  # no network for the embedding model: the fallback must still work
        assert "word overlap" in memory.notes[0]
    assert found and found[0]["bug_id"] == "dead_relu"


def test_memory_keeps_working_in_ram_when_its_file_cannot_be_opened(tmp_path):
    """A folder that cannot hold the database must not crash the run, on the first use or any later one."""
    blocker = tmp_path / "a_file_not_a_folder"
    blocker.write_text("x", encoding="utf-8")
    memory = ExperienceStore(blocker / "memory", use_vectors=False)
    assert "held in RAM" in memory.notes[0]
    first = attempt(memory)
    memory.update_bandit("ctx", "direct_edit", 1.0)
    memory.update_bandit("ctx", "direct_edit", 0.5)
    assert memory.count() == 1 and memory.load_bandit("ctx")["direct_edit"] == {"count": 2.0, "reward_sum": 1.5}
    assert [f["id"] for f in memory.recall("missing_zero_grad", k=3)] == [first]
