"""Training-trace statistics for the learned bug classifier, recorded inside the run.

`script_runner --trace` installs a Tracer next to its Recorder. The tracer keeps a
few numbers per optimizer step and per forward pass, never the tensors, so it is
cheap: the gradient norm, how much of the model got a gradient, how far the step
moved the weights, whether the model's outputs look like probabilities, how many
ReLU outputs are zero, the scale of the inputs, whether validation passes ran
with the model still in training mode, and every call to a loss function: its value,
its `reduction`, the batch size and (for classification) the number of classes, for
training and validation separately. `features.py` turns these into the
classifier's input vector.

Version 4 also watches every `backward()` call, so it measures effects rather than
API calls, and sees scripts that update their weights by hand with no optimizer:

* whether the weights a backward pass reaches still held a gradient from before
  although they had been updated since (gradients pile up across updates: a
  missing `zero_grad()`, however the script clears gradients when it does);
* the gradient norm after each backward pass, how far the weights moved between
  two passes and how that compares with the gradient (the step size actually used);
* the first layer's gradient against the last one's (vanishing gradients);
* how saturated tanh and sigmoid outputs are, the optimizer's kind, and ReLU zeros
  seen through `F.relu`/`torch.relu` whatever the model is.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

MAX_STEPS_KEPT = 2000
MAX_FORWARDS_MEASURED = 300
MAX_TRACKED_VALUES = 2_000_000  # weights copied per step to measure the update; larger models skip it
MAX_LOSS_CALLS = 2000
MAX_BACKWARD_MEASURED = 300
MAX_GRAPH_NODES = 20_000
ADAPTIVE_OPTIMIZERS = ("Adam", "AdamW", "RMSprop", "Adagrad", "Adamax", "NAdam", "RAdam", "Adadelta", "SparseAdam")
# loss functions in torch.nn.functional; the nn.*Loss modules call these, so patching them covers both
LOSS_FUNCTIONS = ("cross_entropy", "nll_loss", "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss",
                  "binary_cross_entropy", "binary_cross_entropy_with_logits")
CLASSIFICATION = ("cross_entropy", "nll_loss")


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
        self.loss_calls: List[Dict[str, Any]] = []  # {train, reduction, value, batch, classes, fn}
        self.mode_layers = False  # a watched model has dropout or batch norm, so train()/eval() matters
        self._model_params: set = set()  # id of every trainable parameter of the watched models
        self._optimizer_params: set = set()  # id of every parameter some optimizer updates
        self._models: Dict[int, Any] = {}
        self._handles: List[Any] = []
        self._undo: List[Any] = []
        self._before: Dict[int, Any] = {}
        self._changed: Dict[int, bool] = {}  # id(parameter in an optimizer) -> ever moved
        self._optimizers: set = set()
        # version 4: backward passes
        self.backward_calls = 0
        self.stale_checks = self.stale_after_update = self.stale_any = self.updates_seen = 0
        self.bw_grad_norms: List[float] = []
        self.bw_update_ratios: List[float] = []
        self.bw_step_over_grad: List[float] = []
        self.layer_grad_ratios: List[float] = []  # log10(first weight's gradient norm / last weight's)
        self.saturated: List[float] = []  # share of tanh/sigmoid outputs within 1% of their limits
        self.relu_zero_all: List[float] = []
        self.optimizer_kinds: set = set()
        self._leaf_before: Dict[int, Any] = {}  # id(leaf) -> copy at the end of the previous backward pass
        self._leaf_versions: Dict[int, int] = {}
        self._last_grad_norm: Optional[float] = None

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

        self._patch_activations(torch, F)
        for name in LOSS_FUNCTIONS:
            original = getattr(F, name, None)
            if original is not None:
                setattr(F, name, self._watch_loss(name, original, torch))
                self._undo.append(lambda name=name, original=original: setattr(F, name, original))

    def _watch_loss(self, name: str, original: Any, torch: Any) -> Any:
        tracer = self

        def loss(prediction: Any, *args: Any, **kwargs: Any) -> Any:
            out = original(prediction, *args, **kwargs)
            try:
                tracer._record_loss(name, prediction, kwargs, out, torch)
            except Exception:  # noqa: BLE001 - measuring must never break the user's training
                pass
            return out

        return loss

    def _record_loss(self, name: str, prediction: Any, kwargs: Dict[str, Any], out: Any, torch: Any) -> None:
        if len(self.loss_calls) >= MAX_LOSS_CALLS or not isinstance(prediction, torch.Tensor):
            return
        reduction = kwargs.get("reduction", "mean")
        if kwargs.get("size_average") is False or kwargs.get("reduce") is False:
            reduction = "sum"  # the old spellings of reduction="sum"
        value = float(out.detach()) if isinstance(out, torch.Tensor) and out.numel() == 1 else None
        batch = int(prediction.shape[0]) if prediction.dim() >= 1 else 1
        classes = int(prediction.shape[1]) if name in CLASSIFICATION and prediction.dim() >= 2 else None
        self.loss_calls.append({"fn": name, "train": bool(torch.is_grad_enabled()), "reduction": str(reduction),
                                "value": value, "batch": batch, "classes": classes})
        if name == "cross_entropy" and self.ce_calls < MAX_FORWARDS_MEASURED and prediction.dim() == 2 \
                and prediction.is_floating_point() and prediction.numel():
            self.ce_calls += 1
            with torch.no_grad():
                x = prediction.detach().float()
                if float(x.min()) >= 0.0 and float(x.max()) <= 1.0 \
                        and float((x.sum(dim=1) - 1.0).abs().max()) < 1e-3:
                    self.ce_probability_inputs += 1

    def _patch_activations(self, torch: Any, F: Any) -> None:
        """tanh/sigmoid saturation and ReLU zeros, whatever builds the model (nn modules call these)."""
        tracer = self

        def measured(original: Any, kind: str) -> Any:
            def wrapper(x: Any, *args: Any, **kwargs: Any) -> Any:
                out = original(x, *args, **kwargs)
                try:
                    if isinstance(out, torch.Tensor) and out.numel() and out.is_floating_point() \
                            and torch.is_grad_enabled():
                        with torch.no_grad():
                            y = out.detach()
                            if kind == "relu" and len(tracer.relu_zero_all) < MAX_FORWARDS_MEASURED:
                                tracer.relu_zero_all.append(float((y == 0).float().mean()))
                            elif kind == "tanh" and len(tracer.saturated) < MAX_FORWARDS_MEASURED:
                                tracer.saturated.append(float((y.abs() > 0.99).float().mean()))
                            elif kind == "sigmoid" and len(tracer.saturated) < MAX_FORWARDS_MEASURED:
                                tracer.saturated.append(float(((y < 0.01) | (y > 0.99)).float().mean()))
                except Exception:  # noqa: BLE001 - measuring must never break the user's training
                    pass
                return out
            return wrapper

        for owner, name, kind in ((F, "relu", "relu"), (torch, "relu", "relu"), (torch, "tanh", "tanh"),
                                  (torch, "sigmoid", "sigmoid"), (F, "sigmoid", "sigmoid"), (F, "tanh", "tanh")):
            original = getattr(owner, name, None)
            if original is None:
                continue
            setattr(owner, name, measured(original, kind))
            self._undo.append(lambda owner=owner, name=name, original=original: setattr(owner, name, original))

    # ------------------------------------------------------------ backward
    def _leaves(self, tensor: Any) -> List[Any]:
        """The tensors a backward pass from `tensor` will give gradients to, in the order the graph reaches them."""
        fn = getattr(tensor, "grad_fn", None)
        if fn is None:
            return [tensor] if getattr(tensor, "requires_grad", False) else []
        leaves, seen, stack, visited = [], set(), [fn], 0
        while stack and visited < MAX_GRAPH_NODES:
            node = stack.pop()
            visited += 1
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            variable = getattr(node, "variable", None)
            if variable is not None:
                leaves.append(variable)
                continue
            stack.extend(next_fn for next_fn, _ in getattr(node, "next_functions", ()) if next_fn is not None)
        return leaves

    def before_backward(self, tensor: Any) -> None:
        self.backward_calls += 1
        if self.backward_calls > MAX_BACKWARD_MEASURED:
            return
        try:
            self._check_stale(tensor)
        except Exception:  # noqa: BLE001
            pass

    def _check_stale(self, tensor: Any) -> None:
        import torch

        leaves = [leaf for leaf in self._leaves(tensor) if not getattr(leaf, "is_sparse", False)]
        if not leaves:
            return
        with torch.no_grad():
            held = sum(1 for leaf in leaves if leaf.grad is not None and not leaf.grad.is_sparse
                       and bool(leaf.grad.abs().sum() > 0))
            moved, total, changed = 0.0, 0.0, False
            for leaf in leaves:
                old = self._leaf_before.get(id(leaf))
                if old is not None and old.shape == leaf.shape:
                    delta = float((leaf.detach() - old).float().norm())
                    moved += delta * delta
                    total += float(old.float().norm()) ** 2
                    changed = changed or delta > 0.0
                elif id(leaf) in self._leaf_versions:
                    changed = changed or leaf._version != self._leaf_versions[id(leaf)]
        if not self._leaf_before and not self._leaf_versions:
            return  # the first backward pass: nothing to compare with
        self.stale_checks += 1
        stale = held >= 0.5 * len(leaves)
        self.stale_any += int(stale)
        if changed:
            self.updates_seen += 1
            self.stale_after_update += int(stale)
            if total > 0 and len(self.bw_update_ratios) < MAX_STEPS_KEPT:
                self.bw_update_ratios.append(math.sqrt(moved) / (math.sqrt(total) + 1e-12))
                if self._last_grad_norm:
                    self.bw_step_over_grad.append(math.sqrt(moved) / self._last_grad_norm)

    def after_backward(self, tensor: Any) -> None:
        if self.backward_calls > MAX_BACKWARD_MEASURED:
            return
        try:
            self._after_backward(tensor)
        except Exception:  # noqa: BLE001
            pass

    def _after_backward(self, tensor: Any) -> None:
        import torch

        leaves = [leaf for leaf in self._leaves(tensor) if not getattr(leaf, "is_sparse", False)]
        if not leaves:
            return
        if not self._optimizers and not self._models and self.backward_calls <= 2:
            self._find_models(leaves, torch)  # no optimizer: find the modules these weights belong to
        order = {}
        for model in self._models.values():
            for index, p in enumerate(model.parameters()):
                order.setdefault(id(p), index)
        if all(id(leaf) in order for leaf in leaves):
            leaves.sort(key=lambda leaf: order[id(leaf)])
        else:
            leaves.reverse()  # the graph is walked from the loss back, so the first layer comes last
        with torch.no_grad():
            norms = [float(leaf.grad.detach().float().norm()) if leaf.grad is not None and not leaf.grad.is_sparse
                     else 0.0 for leaf in leaves]
            total = math.sqrt(sum(n * n for n in norms))
            if len(self.bw_grad_norms) < MAX_STEPS_KEPT:
                self.bw_grad_norms.append(total)
            self._last_grad_norm = total
            weights = [n for leaf, n in zip(leaves, norms) if leaf.dim() >= 2]
            if len(weights) >= 2 and weights[0] > 0 and weights[-1] > 0:
                self.layer_grad_ratios.append(math.log10(weights[0] / weights[-1]))
            if sum(leaf.numel() for leaf in leaves) <= MAX_TRACKED_VALUES:
                self._leaf_before = {id(leaf): leaf.detach().clone() for leaf in leaves}
            else:
                self._leaf_before = {}
                self._leaf_versions = {id(leaf): leaf._version for leaf in leaves}

    def _find_models(self, leaves: List[Any], torch: Any) -> None:
        import gc

        wanted = {id(leaf) for leaf in leaves}
        modules = []
        for candidate in gc.get_objects():
            try:
                if isinstance(candidate, torch.nn.Module):
                    modules.append(candidate)
            except Exception:  # noqa: BLE001 - some proxy objects raise on isinstance
                continue
        children = {id(child) for module in modules for child in module.children()}
        for module in modules:
            if id(module) not in children and any(id(p) in wanted for p in module.parameters()):
                self._watch(module, torch)

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
            self.optimizer_kinds.add(type(optimizer).__name__)
            rates = [float(g["lr"]) for g in optimizer.param_groups if isinstance(g.get("lr"), (int, float))]
            self.lr = max(rates) if rates else None
        for key, model in models.items():
            if key not in self._models:
                self._watch(model, torch)
        params = [p for g in optimizer.param_groups for p in g["params"]]
        self._optimizer_params.update(id(p) for p in params)
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
        self._model_params.update(id(p) for p in model.parameters() if p.requires_grad)
        mode_dependent = (torch.nn.modules.dropout._DropoutNd, torch.nn.modules.batchnorm._NormBase)
        self.mode_layers = self.mode_layers or any(isinstance(m, mode_dependent) for m in model.modules())
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
            "loss_calls": self.loss_calls, "mode_layers": self.mode_layers,
            "outside_optimizer_fraction": (len(self._model_params - self._optimizer_params) / len(self._model_params)
                                           if self._model_params and self._optimizers else None),
            "input_abs_mean": sum(self.input_abs_means) / len(self.input_abs_means) if self.input_abs_means else None,
            "input_std": sum(self.input_stds) / len(self.input_stds) if self.input_stds else None,
            "relu_zero_fraction": sum(self.relu_zero) / len(self.relu_zero) if self.relu_zero else None,
            "bias_mean": sum(biases) / len(biases) if biases else None,
            # version 4
            "backward_calls": self.backward_calls, "stale_checks": self.stale_checks,
            "stale_after_update": self.stale_after_update, "stale_any": self.stale_any,
            "updates_seen": self.updates_seen, "bw_grad_norms": self.bw_grad_norms,
            "bw_update_ratios": self.bw_update_ratios, "bw_step_over_grad": self.bw_step_over_grad,
            "layer_grad_ratios": self.layer_grad_ratios,
            "saturated_fraction": sum(self.saturated) / len(self.saturated) if self.saturated else None,
            "relu_zero_all": sum(self.relu_zero_all) / len(self.relu_zero_all) if self.relu_zero_all else None,
            "optimizer_kinds": sorted(self.optimizer_kinds),
            "adaptive_optimizer": any(k in ADAPTIVE_OPTIMIZERS for k in self.optimizer_kinds),
        }
