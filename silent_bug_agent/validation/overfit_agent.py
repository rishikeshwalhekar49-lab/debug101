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
    """Gate 1: can the patched script learn a single batch?

    Runs the real script on one repeated batch and checks the numbers: it must
    finish, every loss must be finite, the trainable parameters must receive
    gradients, and the loss must fall to `target_ratio` of its start within
    `steps` (a 10% drop by default, with the script's own optimizer and learning
    rate). It is a cheap sanity filter, so it runs before the multi-seed gate.
    """

    def __init__(self, steps: int = 200, target_ratio: float = 0.9, patience: int = 20, *,
                 min_drop: float = 0.02, timeout: float = 120.0, sandbox=None, script_args=None):
        self.steps = steps
        self.target_ratio = target_ratio
        self.patience = patience
        self.min_drop = min_drop  # scripts may run far fewer steps than `steps`
        self.timeout = timeout
        self.script_args = list(script_args or [])
        self.sandbox = sandbox

    def _sandbox(self):
        if self.sandbox is None:
            from .sandbox import SubprocessSandbox

            self.sandbox = SubprocessSandbox()
        return self.sandbox

    def check_code(self, code: str, as_file=None) -> dict:
        """Run one-batch overfitting on source code and return what was measured."""
        result = self._sandbox().run_measured(code, seed=0, mode="overfit", max_steps=self.steps, timeout=self.timeout,
                                              script_args=self.script_args, as_file=as_file)
        report = dict(result.run_report or {})
        report.pop("losses", None)
        passed, reason = self._judge(result, report)
        return {"probe": "one_batch_overfit", "passed": passed, "reason": reason, **report}

    def _judge(self, result, report: dict):
        if result.timed_out:
            return False, f"The run timed out after {self.timeout:g} seconds."
        if not result.success:
            return False, f"The script failed: {str(result.error_message)[-400:]}"
        if report.get("steps", 0) < 2:
            return False, "No training steps were observed."
        if report.get("nonfinite_losses"):
            return False, f"{report['nonfinite_losses']} loss value(s) were NaN or infinite."
        total, flowing = report.get("params_trainable", 0), report.get("params_with_gradient", 0)
        if total and flowing < total:
            return False, f"Only {flowing} of {total} trainable parameter tensors received a gradient."
        ratio = report.get("loss_ratio")
        if ratio is None:
            return False, "No loss was recorded."
        needed = self.target_ratio if report["steps"] >= self.steps else 1.0 - self.min_drop
        if report.get("final_loss", 1.0) < 1e-3 or ratio <= needed:
            return True, f"The loss fell to {ratio:.3f} of its start on one batch."
        return False, f"The loss only reached {ratio:.3f} of its start on one batch (needed {needed:.3f})."

    def call(self, state):
        from pathlib import Path

        from ..schemas import ValidationVerdict
        from .multiseedagent import apply_patch_string

        candidates = getattr(state, "patch_candidates", getattr(state, "patchcandidates", []))
        verdicts = getattr(state, "validation_verdicts", getattr(state, "validationverdicts", []))
        target = getattr(state, "target_file", "")
        judged = {id(v.patch_candidate) for v in verdicts if v.validator == "overfit_oracle"}
        baseline = ""
        if target and Path(target).is_file():
            baseline = Path(target).read_text(encoding="utf-8")

        for candidate in candidates:
            if id(candidate) in judged:
                continue
            if not baseline:
                outcome = {"passed": False, "reason": "There is no target script to run the patch against."}
            else:
                outcome = self.check_code(apply_patch_string(baseline, candidate), as_file=str(Path(target).resolve()))
            verdicts.append(ValidationVerdict(
                patch_candidate=candidate,
                accepted=bool(outcome["passed"]),
                validator="overfit_oracle",
                notes=outcome["reason"],
                extra=outcome,
            ))
        return state

    def __call__(self, state):
        return self.call(state)
