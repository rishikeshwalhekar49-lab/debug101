"""Tests for the Module 1 runtime hooks (silent_bug_agent/runtime/).

Skipped automatically when torch is not installed, so the core suite still
runs torch-free.  Run: python -m pytest tests/test_runtime_hooks.py -v
"""

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from silent_bug_agent import TrainingState, get_default_registry  # noqa: E402
from silent_bug_agent.runtime import (  # noqa: E402
    EmaZScore,
    HookConfig,
    RuntimeHooks,
    SnapshotStream,
    TensorStatsExtractor,
    module_label,
    select_layers,
    spike_indices,
)
from silent_bug_agent.runtime.validate import (  # noqa: E402
    ARCHS,
    EXPECTED,
    run_planted,
    validate_architecture,
    validate_compiled,
    validate_extra_signals,
    validate_stream_consumer,
)

CPU = torch.device("cpu")


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_architecture_validation(arch):
    r = validate_architecture(arch, steps=3, device=CPU)
    assert r["passed"], r


@pytest.mark.parametrize("bug", sorted(EXPECTED))
def test_planted_bugs_same_verdict_as_adapter(bug):
    r = run_planted(bug, device=CPU)
    assert r["ok"], r


def test_extra_signals_stream_and_compile():
    res = validate_extra_signals(CPU)
    assert all(res.values()), res
    assert validate_stream_consumer(CPU)
    assert validate_compiled(CPU)


def test_snapshot_contract_and_labels():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.1), nn.Linear(8, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with RuntimeHooks(model, opt) as hooks:
        for _ in range(3):
            opt.zero_grad()
            F.cross_entropy(model(torch.randn(16, 4)), torch.randint(0, 2, (16,))).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            model(torch.randn(16, 4))
            model(torch.randn(16, 4))  # second eval batch: no extra snapshot by default
    states = hooks.stream.drain()
    assert [s.step for s in states] == [1, 2, 3, 3] and [s.is_eval_phase for s in states] == [False] * 3 + [True]
    train, ev = states[0], states[-1]
    assert isinstance(train, TrainingState)
    assert set(train.grad_norms) == set(train.weight_deltas) == {n for n, _ in model.named_parameters()}
    assert all(train.requires_grad.values())
    stats = train.activation_stats["1[BatchNorm1d]"]
    assert {"mean", "std", "nan_count", "inf_count", "zero_frac", "grad_norm"} <= set(stats)
    assert train.module_modes["3[Dropout]"] == "train" and ev.module_modes["3[Dropout]"] == "eval"
    assert ev.grad_norms == {} and ev.weight_deltas == {} and ev.activation_stats
    assert len(train.loss_history) == 1 and len(states[2].loss_history) == 3
    assert train.extra["loss"]["value"] == pytest.approx(train.loss_history[-1])


def test_every_n_steps_and_manual_end_step():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    hooks = RuntimeHooks(model, opt, HookConfig(every_n_steps=2)).attach()
    for _ in range(6):
        opt.zero_grad()
        model(torch.randn(8, 4)).pow(2).mean().backward()
        opt.step()
    hooks.detach()
    assert [s.step for s in hooks.stream.drain()] == [1, 3, 5]

    hooks = RuntimeHooks(model).attach()  # no optimizer: manual SGD + end_step()
    for _ in range(2):
        model.zero_grad()
        model(torch.randn(8, 4)).pow(2).mean().backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= 0.1 * p.grad
        hooks.end_step()
    hooks.detach()
    states = hooks.stream.drain()
    assert len(states) == 2 and all(s.grad_norms and min(s.weight_deltas.values()) > 0 for s in states)


def test_invariant_callback_and_failures():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with torch.no_grad():
        model[0].bias.fill_(-100.0)  # dead ReLU
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    seen = []
    hooks = RuntimeHooks(model, opt, invariants=get_default_registry(), on_failure=lambda s, r: seen.append(r.invariant_id))
    with hooks:
        for _ in range(2):
            opt.zero_grad()
            F.cross_entropy(model(torch.randn(8, 4)), torch.randint(0, 2, (8,))).backward()
            opt.step()
    assert "activation_health" in seen and hooks.failures


def test_stream_ring_buffer():
    stream = SnapshotStream(capacity=2)
    got = []
    stream.subscribe(lambda s: got.append(s.step))
    for i in range(5):
        stream.publish(TrainingState(step=i))
    assert stream.dropped == 3 and [s.step for s in stream.drain()] == [3, 4]
    assert got == [0, 1, 2, 3, 4] and stream.latest.step == 4
    stream.close()
    assert stream.get(timeout=0.1) is None


def test_extractor_sampling_nan_inf_and_dtypes():
    ex = TensorStatsExtractor(max_elems=1000)
    x = torch.randn(100_000) * 3 + 1
    mean, std = ex.activation(x)[:2].tolist()
    assert abs(mean - 1) < 0.3 and abs(std - 3) < 0.3
    for bad, idx in ((float("nan"), 8), (float("inf"), 9)):
        y = torch.zeros(10_000)
        y[1] = bad  # stride 10 never samples index 1
        assert ex.activation(y)[idx].item() >= 1
        assert ex.gradient(y)[idx - 5].item() >= 1
    st = ex.activation(torch.tensor([1.0, float("nan"), 0.0], dtype=torch.bfloat16)).tolist()
    assert st[8] == 1.0 and st[5] == pytest.approx(0.5)
    assert math.isclose(ex.gradient(torch.ones(4))[0].item(), 2.0)


def test_layer_selection_and_labels():
    layer = nn.TransformerEncoderLayer(8, 2, 16, batch_first=True)
    names = [n for n, _ in select_layers(layer)]
    assert "self_attn" in names and not any(n.startswith("self_attn.") for n in names)
    assert module_label("bn1", nn.BatchNorm2d(3)) == "bn1[BatchNorm2d]"


def test_numerics_spike_detection():
    assert spike_indices([1.0] * 20 + [50.0] + [1.0] * 5) == [20]
    det = EmaZScore(warmup=3)
    for v in (1.0, 1.1, 0.9, 1.0):
        det.update(v)
    assert det.update(float("nan"))[1]
