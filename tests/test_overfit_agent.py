import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from trainmedic.repair.patch import Patch

from silent_bug_agent.validation.overfit_agent import (
    calculate_loss_slope,
    run_overfit_check,
    overfit_one_batch,
    evaluate_candidate_patches,
    should_stop_early,
)


def test_loss_slope_is_negative_when_loss_falls():
    assert calculate_loss_slope([4.0, 2.0, 1.0]) < 0


def test_early_stop_detects_a_plateau():
    assert should_stop_early([1.0, 0.8, 0.8, 0.8], patience=2)


def test_model_can_overfit_one_batch():
    torch.manual_seed(0)

    inputs = torch.linspace(-1, 1, 32).unsqueeze(1)
    targets = 3 * inputs + 2
    data_loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=32,
        shuffle=False,
    )
    model = nn.Linear(1, 1)

    result = run_overfit_check(
        model=model,
        data_loader=data_loader,
        loss_fn=nn.MSELoss(),
        optimizer_factory=lambda parameters: torch.optim.SGD(
            parameters, lr=0.1
        ),
        max_steps=200,
        patience=30,
    )

    assert result["passed"]
    assert result["final_loss"] < result["initial_loss"]
    assert result["loss_slope"] < 0

def test_check_does_not_change_original_model():
    inputs = torch.tensor([[1.0], [2.0]])
    targets = 2 * inputs
    data_loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=2,
    )
    model = nn.Linear(1, 1)
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    run_overfit_check(
        model=model,
        data_loader=data_loader,
        loss_fn=nn.MSELoss(),
        max_steps=3,
        patience=5,
    )

    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_empty_data_loader_gives_clear_error():
    import pytest

    empty_data = TensorDataset(
        torch.empty((0, 1)),
        torch.empty((0, 1)),
    )
    data_loader = DataLoader(empty_data, batch_size=2)

    with pytest.raises(ValueError, match="empty"):
        run_overfit_check(
            model=nn.Linear(1, 1),
            data_loader=data_loader,
            loss_fn=nn.MSELoss(),
        )
def test_overfit_one_batch_accepts_a_batch():
    inputs = torch.linspace(-1, 1, 32).unsqueeze(1)
    targets = 3 * inputs + 2
    model = nn.Linear(1, 1)

    result = overfit_one_batch(
        model=model,
        batch=(inputs, targets),
        loss_fn=nn.MSELoss(),
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        steps=200,
    )

    assert result["probe"] == "overfit_one_batch"
    assert result["passed"]
    assert result["first_loss"] == result["initial_loss"]        

def test_candidate_patch_is_applied_before_overfit_check():
    buggy_code = "weight = 0.0\n"
    patched_code = "weight = 2.5\n"
    patch = Patch.from_sources("model.py", buggy_code, patched_code)

    inputs = torch.linspace(-1, 1, 32).unsqueeze(1)
    targets = 3 * inputs + 2
    received_sources = []
    def model_factory(sources):
        received_sources.append(sources["model.py"])
        weight = float(sources["model.py"].split("=")[1])
        model = nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(weight)
            model.bias.fill_(2.0)
        return model

    results = evaluate_candidate_patches(
        buggy_sources={"model.py": buggy_code},
        candidate_patches=[patch],
        model_factory=model_factory,
        batch=(inputs, targets),
        loss_fn=nn.MSELoss(),
        optimizer_factory=lambda parameters: torch.optim.SGD(
            parameters, lr=0.1
        ),
        steps=200,
    )

    assert len(results) == 1
    assert results[0]["candidate_index"] == 0
    assert results[0]["patch_applied"]
    assert results[0]["passed"]
def test_overfit_check_rejects_optimizer_that_does_not_learn():
    inputs = torch.linspace(-1, 1, 32).unsqueeze(1)
    targets = 3 * inputs + 2
    data_loader = DataLoader(
        TensorDataset(inputs, targets),
        batch_size=32,
        shuffle=False,
    )
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)

    result = run_overfit_check(
        model=model,
        data_loader=data_loader,
        loss_fn=nn.MSELoss(),
        optimizer_factory=lambda parameters: torch.optim.SGD(
            parameters, lr=0.0
        ),
        max_steps=200,
        patience=5,
    )

    assert result["passed"] is False
    assert result["final_loss"] == result["initial_loss"]
