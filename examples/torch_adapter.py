"""Stopgap adapter: feed a live PyTorch training loop into the Phase 0 invariants.

Phase 1 will replace this with proper hooks. Until then, this lets you run the
invariant library against your own training code.

NOT TESTED AGAINST REAL TORCH when written (the build sandbox had no torch).
Treat the first run as a smoke test and use the "plant a bug" checks in the
README/guide to confirm each invariant actually fires on your setup.

Usage (call snapshot() AFTER optimizer.step() and BEFORE zero_grad(), because
gradient norms must still exist and weight deltas need the post-step weights):

    collector = TorchStateCollector(model)

    for step, (x, y) in enumerate(train_loader):
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        for failure in collector.snapshot(step, loss.item()):
            print(failure)
        optimizer.zero_grad()

    # validation pass: call AFTER model.eval() and a forward pass
    model.eval()
    with torch.no_grad():
        model(x_val)
    for failure in collector.snapshot(step, is_eval_phase=True):
        print(failure)
    model.train()

Costs: keeps one extra copy of every parameter (for weight deltas) and calls
.item() several times per step, which synchronizes the GPU. Snapshot every N
steps instead of every step if that matters for your run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from silent_bug_agent import TrainingState, get_default_registry, run_all  # noqa: E402


class TorchStateCollector:
    def __init__(self, model, invariants=None, capture_activations=True):
        self.model = model
        # Invariants are stateful (rolling windows), so build them once and reuse.
        self.invariants = invariants if invariants is not None else get_default_registry()
        self.capture_activations = capture_activations
        self.loss_history = []
        self._prev_weights = {n: p.detach().clone() for n, p in model.named_parameters()}
        self._act_stats = {}
        self._hooks = []
        for name, module in model.named_modules():
            is_leaf = next(module.children(), None) is None
            if is_leaf:
                self._hooks.append(module.register_forward_hook(self._make_hook(name, module)))

    @staticmethod
    def _label(name, module):
        # The mode invariant matches on substrings like "batchnorm"/"dropout".
        # Attribute names such as "bn1" don't contain those, so include the
        # class name in the key.
        return f"{name}[{type(module).__name__}]"

    def _make_hook(self, name, module):
        label = self._label(name, module)

        def hook(_module, _inputs, output):
            if not self.capture_activations:
                return
            if not torch.is_tensor(output) or not output.is_floating_point():
                return
            with torch.no_grad():
                o = output.detach()
                self._act_stats[label] = {
                    "mean": o.mean().item(),
                    "std": o.std().item() if o.numel() > 1 else 0.0,
                    "nan_count": int(torch.isnan(o).sum().item()),
                    "inf_count": int(torch.isinf(o).sum().item()),
                    "zero_frac": float((o == 0).float().mean().item()),
                }

        return hook

    def snapshot(self, step, loss=None, is_eval_phase=False):
        """Build a TrainingState, run every invariant, return the failures."""
        if loss is not None and not is_eval_phase:
            self.loss_history.append(float(loss))

        grad_norms, weight_deltas, requires_grad = {}, {}, {}
        for name, p in self.model.named_parameters():
            requires_grad[name] = p.requires_grad
            if is_eval_phase:
                continue
            if p.grad is not None:
                grad_norms[name] = p.grad.detach().norm().item()
            weight_deltas[name] = (p.detach() - self._prev_weights[name]).norm().item()
            self._prev_weights[name].copy_(p.detach())

        module_modes = {
            self._label(n, m): ("train" if m.training else "eval")
            for n, m in self.model.named_modules()
        }

        state = TrainingState(
            step=step,
            loss_history=list(self.loss_history),
            grad_norms=grad_norms,
            weight_deltas=weight_deltas,
            requires_grad=requires_grad,
            activation_stats=dict(self._act_stats),
            module_modes=module_modes,
            is_eval_phase=is_eval_phase,
        )
        return [r for r in run_all(self.invariants, state) if not r.passed]

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
