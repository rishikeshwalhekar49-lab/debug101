"""Fixes for the four serious problems found in the earlier review of the runtime hooks."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from silent_bug_agent.runtime.hooks import RuntimeHooks  # noqa: E402


def test_sparse_gradients_no_longer_stop_training():
    model = nn.EmbeddingBag(50, 8, sparse=True)
    opt = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    hooks = RuntimeHooks(model, opt).attach()
    for _ in range(3):
        opt.zero_grad()
        model(torch.randint(0, 50, (16, 4))).pow(2).mean().backward()
        opt.step()
    hooks.detach()
    assert hooks.step == 3
    state = hooks.stream.drain()[-1]
    assert state.grad_norms["weight"] > 0


def test_an_internal_error_switches_monitoring_off_and_training_continues(monkeypatch):
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    hooks = RuntimeHooks(model, opt).attach()

    def broken():
        raise RuntimeError("simulated monitoring bug")

    monkeypatch.setattr(hooks.param_coll, "before_step", broken)
    before = model.weight.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 4)).pow(2).mean().backward()
        opt.step()
    hooks.detach()
    assert hooks.step == 3 and not torch.equal(before, model.weight)
    assert hooks.errors and "simulated monitoring bug" in hooks.errors[0]


def test_closure_gradients_are_measured_after_the_closure_runs():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(32, 4)
    hooks = RuntimeHooks(model, opt).attach()
    truth = []

    def closure():
        opt.zero_grad()
        loss = (model(x) - 1.0).pow(2).mean()
        loss.backward()
        truth.append(model.weight.grad.norm().item())
        return loss

    for _ in range(3):
        opt.step(closure)
    hooks.detach()
    states = hooks.stream.drain()
    assert states[0].grad_norms["weight"] == pytest.approx(truth[0], rel=1e-5)
    assert states[-1].grad_norms["weight"] == pytest.approx(truth[-1], rel=1e-5)


def test_grad_scaler_loss_is_recorded_unscaled():
    scaler_type = getattr(torch.amp, "GradScaler", None)
    if scaler_type is None:
        pytest.skip("torch.amp.GradScaler is not available")
    try:
        scaler = scaler_type("cpu", init_scale=1024.0)
    except Exception as exc:  # older torch: no CPU scaler
        pytest.skip(f"no CPU GradScaler: {exc}")
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    hooks = RuntimeHooks(model, opt).attach()
    x = torch.randn(16, 4)
    loss = (model(x) - 1.0).pow(2).mean()
    true_loss = loss.item()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    hooks.detach()
    assert hooks.loss_history[0] == pytest.approx(true_loss, rel=1e-4)
