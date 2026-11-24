"""Training-trace statistics for the learned bug classifier, recorded inside the run.

`script_runner --trace` installs a Tracer next to its Recorder. The tracer keeps a
few numbers per optimizer step and per forward pass, never the tensors, so it is
cheap: the gradient norm, how much of the model got a gradient, how far the step
moved the weights, whether the model's outputs look like probabilities, how many
ReLU outputs are zero, the scale of the inputs, and whether validation passes ran
with the model still in training mode. `features.py` turns these into the
classifier's input vector.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

MAX_STEPS_KEPT = 2000
MAX_FORWARDS_MEASURED = 300
MAX_TRACKED_VALUES = 2_000_000  # weights copied per step to measure the update; larger models skip it


def _first_tensor(value: Any, torch: Any) -> Optional[Any]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item, torch)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item, torch)
            if found is not None:
                return found
    return None


class Tracer:
    def __init__(self) -> None:
        self.grad_norms: List[float] = []
        self.no_grad_fraction: List[float] = []  # per step: share of parameter tensors without a gradient
        self.update_ratios: List[float] = []
        self.zero_grad_calls = 0
        self.lr: Optional[float] = None
        self.forwards = self.no_grad_forwards = self.no_grad_training_forwards = 0
        self.measured_outputs = self.unit_outputs = self.rowsum_one_outputs = 0
        self.input_abs_means: List[float] = []
        self.input_stds: List[float] = []
        self.relu_zero: List[float] = []
        self.ce_calls = self.ce_probability_inputs = 0  # cross_entropy given probabilities: softmax applied twice
        self._models: Dict[int, Any] = {}
        self._handles: List[Any] = []
        self._undo: List[Any] = []
        self._before: Dict[int, Any] = {}
        self._changed: Dict[int, bool] = {}  # id(parameter in an optimizer) -> ever moved
        self._optimizers: set = set()

    # ---------------------------------------------------------------- install
    def install(self) -> None:
        import torch

        tracer = self
        originals = (torch.optim.Optimizer.zero_grad, torch.nn.Module.zero_grad)

        def optimizer_zero_grad(optimizer: Any, *args: Any, **kwargs: Any) -> Any:
            tracer.zero_grad_calls += 1
            return originals[0](optimizer, *args, **kwargs)

        def module_zero_grad(module: Any, *args: Any, **kwargs: Any) -> Any:
            tracer.zero_grad_calls += 1
            return originals[1](module, *args, **kwargs)

        torch.optim.Optimizer.zero_grad = optimizer_zero_grad
        torch.nn.Module.zero_grad = module_zero_grad
        self._undo += [lambda: setattr(torch.optim.Optimizer, "zero_grad", originals[0]),
                       lambda: setattr(torch.nn.Module, "zero_grad", originals[1])]

        import torch.nn.functional as F

        original_ce = F.cross_entropy

        def cross_entropy(logits: Any, *args: Any, **kwargs: Any) -> Any:
            if tracer.ce_calls < MAX_FORWARDS_MEASURED and isinstance(logits, torch.Tensor) \
                    and logits.is_floating_point() and logits.dim() == 2 and logits.numel():
                tracer.ce_calls += 1
                with torch.no_grad():
                    x = logits.detach().float()
                    if float(x.min()) >= 0.0 and float(x.max()) <= 1.0 \
                            and float((x.sum(dim=1) - 1.0).abs().max()) < 1e-3:
                        tracer.ce_probability_inputs += 1
            return original_ce(logits, *args, **kwargs)

        F.cross_entropy = cross_entropy  # nn.CrossEntropyLoss.forward calls F.cross_entropy
        self._undo.append(lambda: setattr(F, "cross_entropy", original_ce))

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()

    # ------------------------------------------------------------ optimizer
    def before_step(self, optimizer: Any, models: Dict[int, Any]) -> None:
        import torch

        if id(optimizer) not in self._optimizers:
            self._optimizers.add(id(optimizer))
            rates = [float(g["lr"]) for g in optimizer.param_groups if isinstance(g.get("lr"), (int, float))]
            self.lr = max(rates) if rates else None
        for key, model in models.items():
            if key not in self._models:
                self._watch(model, torch)
        params = [p for g in optimizer.param_groups for p in g["params"]]
        if len(self.grad_norms) >= MAX_STEPS_KEPT or not params:
            return
        squares, missing = 0.0, 0
        for p in params:
            grad = p.grad
            if grad is None or grad.is_sparse:
                missing += 1
                continue
            norm = float(grad.detach().float().norm())
            if norm == 0.0:
                missing += 1
            squares += norm * norm
        self.grad_norms.append(math.sqrt(squares))
        self.no_grad_fraction.append(missing / len(params))
        if sum(p.numel() for p in params) <= MAX_TRACKED_VALUES:
            self._before = {id(p): p.detach().clone() for p in params}

    def after_step(self, optimizer: Any) -> None:
        if not self._before:
            return
        moved, total = 0.0, 0.0
        for group in optimizer.param_groups:
            for p in group["params"]:
                old = self._before.get(id(p))
                if old is None:
                    continue
                delta = float((p.detach() - old).float().norm())
                moved += delta * delta
                total += float(old.float().norm()) ** 2
                self._changed[id(p)] = self._changed.get(id(p), False) or delta > 0.0
        self._before = {}
        if len(self.update_ratios) < MAX_STEPS_KEPT:
            self.update_ratios.append(math.sqrt(moved) / (math.sqrt(total) + 1e-12))

    # -------------------------------------------------------------- forward
    def _watch(self, model: Any, torch: Any) -> None:
        self._models[id(model)] = model
        tracer = self

        def on_forward(module: Any, inputs: Any, output: Any) -> None:
            tracer.forwards += 1
            if not torch.is_grad_enabled():
                tracer.no_grad_forwards += 1
                if module.training:
                    tracer.no_grad_training_forwards += 1
            if tracer.measured_outputs >= MAX_FORWARDS_MEASURED:
                return
            with torch.no_grad():
                x = _first_tensor(inputs, torch)
                if x is not None and x.is_floating_point() and x.numel():
                    tracer.input_abs_means.append(float(x.detach().float().abs().mean()))
                    tracer.input_stds.append(float(x.detach().float().std())) if x.numel() > 1 else None
                y = _first_tensor(output, torch)
                if y is not None and y.is_floating_point() and y.dim() == 2 and y.numel():
                    y = y.detach().float()
                    tracer.measured_outputs += 1
                    if float(y.min()) >= 0.0 and float(y.max()) <= 1.0:
                        tracer.unit_outputs += 1
                        if float((y.sum(dim=1) - 1.0).abs().max()) < 1e-3:
                            tracer.rowsum_one_outputs += 1

        def on_relu(module: Any, inputs: Any, output: Any) -> None:
            if len(tracer.relu_zero) < MAX_FORWARDS_MEASURED and isinstance(output, torch.Tensor) and output.numel():
                tracer.relu_zero.append(float((output.detach() == 0).float().mean()))

        self._handles.append(model.register_forward_hook(on_forward))
        for child in model.modules():
            if isinstance(child, torch.nn.ReLU):
                self._handles.append(child.register_forward_hook(on_relu))
        # functional relu inside forward() is invisible to module hooks; F.relu is patched for that
        if not any(isinstance(m, torch.nn.ReLU) for m in model.modules()) and not self._undo_has_relu:
            self._patch_functional_relu(torch)

    _undo_has_relu = False

    def _patch_functional_relu(self, torch: Any) -> None:
        import torch.nn.functional as F

        tracer, original, original_torch = self, F.relu, torch.relu

        def relu(x: Any, *args: Any, **kwargs: Any) -> Any:
            out = original(x, *args, **kwargs)
            if len(tracer.relu_zero) < MAX_FORWARDS_MEASURED and isinstance(out, torch.Tensor) and out.numel():
                tracer.relu_zero.append(float((out.detach() == 0).float().mean()))
            return out

        def torch_relu(x: Any) -> Any:
            out = original_torch(x)
            if len(tracer.relu_zero) < MAX_FORWARDS_MEASURED and isinstance(out, torch.Tensor) and out.numel():
                tracer.relu_zero.append(float((out.detach() == 0).float().mean()))
            return out

        F.relu, torch.relu = relu, torch_relu
        self._undo_has_relu = True
        self._undo += [lambda: setattr(F, "relu", original), lambda: setattr(torch, "relu", original_torch)]

    # --------------------------------------------------------------- report
    def report(self) -> Dict[str, Any]:
        biases: List[float] = []
        for model in self._models.values():
            for name, p in model.named_parameters():
                if name.endswith("bias") and p.numel():
                    biases.append(float(p.detach().float().mean()))
        changed = list(self._changed.values())
        return {
            "grad_norms": self.grad_norms, "no_grad_fraction": self.no_grad_fraction,
            "update_ratios": self.update_ratios, "zero_grad_calls": self.zero_grad_calls, "lr": self.lr,
            "frozen_fraction": (1.0 - sum(changed) / len(changed)) if changed else None,
            "forwards": self.forwards, "no_grad_forwards": self.no_grad_forwards,
            "no_grad_training_forwards": self.no_grad_training_forwards,
            "measured_outputs": self.measured_outputs, "unit_outputs": self.unit_outputs,
            "rowsum_one_outputs": self.rowsum_one_outputs,
            "ce_calls": self.ce_calls, "ce_probability_inputs": self.ce_probability_inputs,
            "input_abs_mean": sum(self.input_abs_means) / len(self.input_abs_means) if self.input_abs_means else None,
            "input_std": sum(self.input_stds) / len(self.input_stds) if self.input_stds else None,
            "relu_zero_fraction": sum(self.relu_zero) / len(self.relu_zero) if self.relu_zero else None,
            "bias_mean": sum(biases) / len(biases) if biases else None,
        }
