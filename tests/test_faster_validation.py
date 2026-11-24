"""Gate 2 runs seeds in parallel, reuses baseline runs, and may stop after 3 seeds.

The two looks share the error budget (1% early, 4% at the end), so a patch that
does nothing is still accepted at most 5% of the time."""

import threading
from collections import Counter

import pytest

from silent_bug_agent.schemas import PatchCandidate, RepairStrategy
from silent_bug_agent.validation.multiseedagent import MultiseedAgent, writes_files
from silent_bug_agent.validation.sandbox import ExecutionResult

BASE = "print('baseline')\n"


class FakeSandbox:
    """Returns a final loss chosen per (code, seed); counts every run."""

    def __init__(self, losses):
        self.losses = losses  # code -> {seed: loss or None}
        self.calls = Counter()
        self.lock = threading.Lock()
        self.config = type("Config", (), {"max_cpu_cores": None})()

    def run_code(self, code, seed=None, timeout=None):
        with self.lock:
            self.calls[(code, seed)] += 1
        loss = self.losses[code][seed]
        if loss is None:
            return ExecutionResult(returncode=1, stdout="", stderr="boom", error_message="boom")
        return ExecutionResult(returncode=0, stdout="", stderr="", metrics={"final_loss": loss})


def candidate(code):
    return PatchCandidate(hypothesis_bug_id="missing_zero_grad", diff="", strategy=list(RepairStrategy)[0],
                          source="rules", confidence=0.9, extra={"patched_code": code})


def agent(sandbox, **kwargs):
    return MultiseedAgent(seeds=[1, 2, 3, 4, 5], sandbox=sandbox, **kwargs)


@pytest.fixture(autouse=True)
def patched_source(monkeypatch):
    # the candidate carries its patched code directly, so no diff has to be applied
    monkeypatch.setattr("silent_bug_agent.validation.multiseedagent.apply_patch_string",
                        lambda baseline, cand: cand.extra["patched_code"])


def test_a_clear_improvement_stops_after_three_seeds():
    good = "print('good')\n"
    sandbox = FakeSandbox({BASE: {s: 1.0 + 0.01 * s for s in range(1, 6)},
                           good: {s: 0.2 + 0.01 * s for s in range(1, 6)}})
    verdict = agent(sandbox).verify_patch(BASE, candidate(good))
    assert verdict.accepted and verdict.seeds_run == 3
    assert verdict.extra["sequential"]["stopped_early"] == "accept"
    assert sum(sandbox.calls.values()) == 6 and "Stopped after 3 of 5 seeds" in verdict.notes


def test_no_improvement_stops_early_and_is_rejected():
    worse = "print('worse')\n"
    sandbox = FakeSandbox({BASE: {s: 1.0 for s in range(1, 6)}, worse: {s: 1.2 + 0.01 * s for s in range(1, 6)}})
    verdict = agent(sandbox).verify_patch(BASE, candidate(worse))
    assert not verdict.accepted and verdict.seeds_run == 3
    assert verdict.extra["sequential"]["stopped_early"] == "futility"


def test_an_unclear_result_runs_every_seed_at_the_stricter_final_level():
    noisy = "print('noisy')\n"
    base = {1: 1.00, 2: 1.10, 3: 0.95, 4: 1.05, 5: 1.02}
    patched = {1: 0.90, 2: 1.05, 3: 0.97, 4: 0.92, 5: 0.99}
    sandbox = FakeSandbox({BASE: base, noisy: patched})
    verdict = agent(sandbox).verify_patch(BASE, candidate(noisy))
    info = verdict.extra["sequential"]
    assert verdict.seeds_run == 5 and info["stopped_early"] is None and info["alpha_used"] == pytest.approx(0.04)
    assert sum(sandbox.calls.values()) == 10


def test_a_patch_that_crashes_is_rejected_after_the_first_seeds():
    broken = "print('broken')\n"
    sandbox = FakeSandbox({BASE: {s: 1.0 for s in range(1, 6)}, broken: {s: None for s in range(1, 6)}})
    verdict = agent(sandbox).verify_patch(BASE, candidate(broken))
    assert not verdict.accepted and verdict.seeds_run == 3


def test_baseline_runs_are_reused_for_every_candidate_and_order_is_kept():
    a, b = "print('a')\n", "print('b')\n"
    sandbox = FakeSandbox({BASE: {s: 1.0 + 0.01 * s for s in range(1, 6)},
                           a: {s: 0.2 + 0.01 * s for s in range(1, 6)},
                           b: {s: 0.3 + 0.01 * s for s in range(1, 6)}})
    gate = agent(sandbox, workers=4)
    first, second = gate.verify_patch(BASE, candidate(a)), gate.verify_patch(BASE, candidate(b))
    assert first.accepted and second.accepted
    assert all(sandbox.calls[(BASE, s)] == 1 for s in (1, 2, 3))  # never re-run
    # pairs stay matched by seed even though the runs finished in any order
    assert first.extra["baseline_values"] == pytest.approx([1.01, 1.02, 1.03])
    assert second.extra["patched_values"] == pytest.approx([0.31, 0.32, 0.33])


def test_three_seeds_are_one_look_at_the_usual_level():
    good = "print('good')\n"
    sandbox = FakeSandbox({BASE: {s: 1.0 + 0.01 * s for s in (1, 2, 3)}, good: {s: 0.9 + 0.012 * s for s in (1, 2, 3)}})
    verdict = MultiseedAgent(seeds=[1, 2, 3], sandbox=sandbox).verify_patch(BASE, candidate(good))
    assert verdict.extra["sequential"] == {"staged": False, "seeds_planned": 3, "seeds_run": 3,
                                           "stopped_early": None, "alpha_used": 0.05}


def test_scripts_that_write_files_run_one_at_a_time():
    assert writes_files("torch.save(model.state_dict(), 'model.pt')")
    assert writes_files("with open('log.txt', 'w') as f:\n    f.write('x')")
    assert writes_files("df.to_csv('out.csv')")
    assert writes_files("datasets.MNIST('data', train=True, download=True)")  # parallel downloads would collide
    assert not writes_files("x = torch.randn(3)\nprint(open('data.txt').read())")
    sandbox = FakeSandbox({})
    gate = MultiseedAgent(seeds=[1, 2, 3], sandbox=sandbox)
    assert gate._workers_for(["torch.save(m, 'a.pt')"], 6) == 1
    assert gate._workers_for(["print(1)"], 6) >= 1
