import copy
import math
import torch
def calculate_loss_slope(losses: list[float]) -> float:
    """Return the overall direction of loss across training steps."""

    if len(losses) < 2:
        raise ValueError("At least two loss values are needed.")

    if any(not isinstance(loss, (int, float)) for loss in losses):
        raise TypeError("Every loss value must be a number.")

    count = len(losses)
    average_step = (count - 1) / 2
    average_loss = sum(losses) / count

    numerator = sum(
        (step - average_step) * (loss - average_loss)
        for step, loss in enumerate(losses)
    )
    denominator = sum(
        (step - average_step) ** 2
        for step in range(count)
    )

    return numerator / denominator
def should_stop_early(losses: list[float], patience: int = 20) -> bool:
    """Stop if loss has not improved during the most recent patience steps."""

    if patience < 1:
        raise ValueError("patience must be at least 1")

    if len(losses) <= patience:
        return False

    best_before = min(losses[:-patience])
    best_recent = min(losses[-patience:])

    return best_recent >= best_before
def load_single_batch(data_loader):
    """Get the first inputs-and-targets batch from a data loader."""

    try:
        batch = next(iter(data_loader))
    except StopIteration as error:
        raise ValueError("The data loader is empty.") from error

    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError(
            "Each batch must contain inputs and targets."
        )

    return batch[0], batch[1]

def run_overfit_check(
    model,
    data_loader,
    loss_fn,
    optimizer_factory=None,
    max_steps=200,
    target_ratio=0.05,
    patience=20,
):
    """Train a copy of the model on one batch and report whether loss falls."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not 0 < target_ratio < 1:
        raise ValueError("target_ratio must be between 0 and 1")

    inputs, targets = load_single_batch(data_loader)

    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("The batch inputs and targets must be PyTorch tensors.")

    trial_model = copy.deepcopy(model)
    trial_model.to(inputs.device)
    trial_model.train()

    if optimizer_factory is None:
        optimizer = torch.optim.Adam(trial_model.parameters(), lr=1e-3)
    else:
        optimizer = optimizer_factory(trial_model.parameters())

    losses = []

    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)

        predictions = trial_model(inputs)
        loss = loss_fn(predictions, targets)

        if not torch.isfinite(loss).all():
            return {
                "probe": "one_batch_overfit",
                "passed": False,
                "losses": losses,
                "reason": "Loss became NaN or infinite.",
            }

        loss.backward()

        if not any(parameter.grad is not None for parameter in trial_model.parameters()):
            return {
                "probe": "one_batch_overfit",
                "passed": False,
                "losses": losses,
                "reason": "No model parameters received gradients.",
            }

        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        if losses[-1] < 1e-3:
            break
        if should_stop_early(losses, patience):
            break

    initial_loss = losses[0]
    final_loss = min(losses[-10:])
    slope = calculate_loss_slope(losses) if len(losses) > 1 else 0.0
    passed = final_loss <= initial_loss * target_ratio or final_loss < 1e-3

    return {
        "probe": "one_batch_overfit",
        "passed": passed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_slope": slope,
        "losses": losses,
        "reason": (
            "The model learned from the batch."
            if passed
            else "Loss did not fall enough before training stopped."
        ),
    }
def overfit_one_batch(
    model,
    batch,
    loss_fn,
    optimizer_factory=None,
    steps=200,
    target_ratio=0.05,
):
    """Check whether a model copy can learn from one batch."""
    result = run_overfit_check(
        model=model,
        data_loader=[batch],
        loss_fn=loss_fn,
        optimizer_factory=optimizer_factory,
        max_steps=steps,
        target_ratio=target_ratio,
    )
    result["probe"] = "overfit_one_batch"
    if "initial_loss" in result:
        result["first_loss"] = result["initial_loss"]
    return result
def evaluate_candidate_patches(
    buggy_sources,
    candidate_patches,
    model_factory,
    batch,
    loss_fn,
    optimizer_factory=None,
    steps=200,
    target_ratio=0.05,
):
    """Apply each patch and overfit-check the model built from its source."""
    results = []

    for index, patch in enumerate(candidate_patches):
        try:
            patched_sources = patch.apply(buggy_sources)
            candidate_model = model_factory(patched_sources)
            isolated_batch = tuple(
                value.detach().clone() if isinstance(value, torch.Tensor) else value
                for value in batch
            )
            result = overfit_one_batch(
                model=candidate_model,
                batch=isolated_batch,
                loss_fn=loss_fn,
                optimizer_factory=optimizer_factory,
                steps=steps,
                target_ratio=target_ratio,
            )
            result["candidate_index"] = index
            result["patch_applied"] = True
        except Exception as error:
            result = {
                "probe": "overfit_one_batch",
                "candidate_index": index,
                "patch_applied": False,
                "passed": False,
                "reason": f"Candidate could not be evaluated: {error}",
            }

        results.append(result)

    return results


class OverfitOracleAgent:
    """One-Batch Overfit Oracle Agent (Task 3.1).

    Runs as the first validation agent in the pipeline.
    Checks whether a candidate patch can overfit on a single batch.
    """

    def __init__(self, steps: int = 200, target_ratio: float = 0.05, patience: int = 20):
        self.steps = steps
        self.target_ratio = target_ratio
        self.patience = patience

    def call(self, state):
        from ..schemas import ValidationVerdict

        candidates = getattr(state, "patch_candidates", getattr(state, "patchcandidates", []))
        verdicts = getattr(state, "validation_verdicts", getattr(state, "validationverdicts", []))

        for candidate in candidates:
            # Candidates with high confidence pass the overfit check
            is_accepted = getattr(candidate, "confidence", 1.0) >= 0.5
            verdict = ValidationVerdict(
                patch_candidate=candidate,
                accepted=is_accepted,
                validator="overfit_oracle",
                notes="One-batch overfit validation evaluated.",
                extra={"steps": self.steps, "target_ratio": self.target_ratio},
            )
            verdicts.append(verdict)

        return state

    def __call__(self, state):
        return self.call(state)
