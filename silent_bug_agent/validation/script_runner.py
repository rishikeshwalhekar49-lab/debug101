"""Runs a training script under measurement, in its own process.

    python -m silent_bug_agent.validation.script_runner train.py --mode overfit --result out.json

It works on any script, because nothing in the script has to cooperate:

* every seed the script sets is shifted by the run's seed, so a hard-coded
  `torch.manual_seed(42)` still gives different runs across seeds;
* losses are read from `backward()` and steps from a global optimizer hook;
* `--max-steps` stops training cleanly, which keeps CPU-only runs short.

In `overfit` mode each step sees the same batch: a DataLoader repeats its first
batch, and for data drawn inside the loop the random state is rewound after each
step, so step 2 onwards trains on one fixed batch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from typing import Any, Dict, List, Optional

MAX_LOSSES = 5000


class Recorder:
    def __init__(self, seed: int, mode: str, max_steps: Optional[int]) -> None:
        self.seed, self.mode, self.max_steps = seed, mode, max_steps
        self.losses: List[float] = []
        self.steps = 0
        self.params: Dict[int, bool] = {}  # id(trainable parameter) -> ever had a non-zero gradient
        self._rng: Optional[tuple] = None
        self._loaders: Dict[int, Any] = {}
        self._undo: List[Any] = []

    def install(self) -> None:
        import torch
        from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

        from ..runtime.instrument import StopTraining

        recorder = self
        self._stop = StopTraining
        original_seed, original_backward = torch.manual_seed, torch.Tensor.backward

        def manual_seed(value: int) -> Any:
            return original_seed(int(value) + recorder.seed)

        def backward(tensor: Any, *args: Any, **kwargs: Any) -> Any:
            if tensor.numel() == 1 and tensor.is_floating_point() and len(recorder.losses) < MAX_LOSSES:
                recorder.losses.append(float(tensor.detach()))
            return original_backward(tensor, *args, **kwargs)

        torch.manual_seed = manual_seed
        torch.Tensor.backward = backward
        self._undo += [lambda: setattr(torch, "manual_seed", original_seed),
                       lambda: setattr(torch.Tensor, "backward", original_backward)]
        handles = [register_optimizer_step_pre_hook(self._before_step), register_optimizer_step_post_hook(self._after_step)]
        self._undo += [h.remove for h in handles]
        if self.mode == "overfit":
            self._repeat_first_batch(torch)

    def _repeat_first_batch(self, torch: Any) -> None:
        original_iter = torch.utils.data.DataLoader.__iter__
        recorder = self

        def repeating(loader: Any) -> Any:
            key = id(loader)
            if key not in recorder._loaders:
                source = original_iter(loader)
                try:
                    recorder._loaders[key] = next(source)
                except StopIteration:
                    return iter(())
                del source
            try:
                count = len(loader)
            except TypeError:
                count = 1000
            return iter([recorder._loaders[key]] * count)

        torch.utils.data.DataLoader.__iter__ = repeating
        self._undo.append(lambda: setattr(torch.utils.data.DataLoader, "__iter__", original_iter))

    def _before_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                grad = parameter.grad
                seen = grad is not None and bool((grad._values() if grad.is_sparse else grad).abs().sum() > 0)
                self.params[id(parameter)] = self.params.get(id(parameter), False) or seen

    def _after_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        self.steps += 1
        if self.mode == "overfit":
            self._rewind()
        if self.max_steps is not None and self.steps >= self.max_steps:
            raise self._stop(f"step budget of {self.max_steps} reached")

    def _rewind(self) -> None:
        import torch

        if self._rng is None:
            numpy_state = None
            if "numpy" in sys.modules:
                numpy_state = sys.modules["numpy"].random.get_state()
            self._rng = (torch.get_rng_state(), random.getstate(), numpy_state)
            return
        torch.set_rng_state(self._rng[0])
        random.setstate(self._rng[1])
        if self._rng[2] is not None:
            sys.modules["numpy"].random.set_state(self._rng[2])

    def uninstall(self) -> None:
        for undo in reversed(self._undo):
            undo()
        self._undo.clear()

    def report(self) -> Dict[str, Any]:
        # in overfit mode step 1 saw a different batch from every later step, so it is left out
        values = self.losses[1:] if self.mode == "overfit" and len(self.losses) > 3 else self.losses
        finite = [v for v in values if math.isfinite(v)]
        head = sum(finite[:3]) / max(1, len(finite[:3])) if finite else None
        tail = sum(finite[-3:]) / max(1, len(finite[-3:])) if finite else None
        return {"steps": self.steps, "losses": self.losses, "nonfinite_losses": len(values) - len(finite),
                "initial_loss": head, "final_loss": tail,
                "loss_ratio": (tail / head) if finite and head else None,
                "params_trainable": len(self.params), "params_with_gradient": sum(self.params.values())}


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(seed % (2 ** 32))


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    script_args: List[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, script_args = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(prog="script_runner")
    parser.add_argument("script")
    parser.add_argument("--mode", choices=["run", "overfit"], default="run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--result", required=True)
    parser.add_argument("--as-file", default=None, help="path the script should believe it was loaded from")
    options = parser.parse_args(argv)

    from ..runtime.instrument import exec_script

    seed_everything(options.seed)
    recorder = Recorder(options.seed, options.mode, options.max_steps)
    recorder.install()
    try:
        outcome = exec_script(options.script, script_args, display_path=options.as_file)
    finally:
        recorder.uninstall()
    report = recorder.report()
    report.update({"status": outcome.status, "error": outcome.error, "seconds": round(outcome.duration, 3)})
    with open(options.result, "w", encoding="utf-8") as handle:
        json.dump(report, handle)
    return 1 if outcome.status == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
