"""Standalone validation of the Module 1 runtime hooks.

    python -m silent_bug_agent.runtime.validate                 # CPU, or CUDA if available
    python -m silent_bug_agent.runtime.validate --device cuda --steps 8 --json report.json

1. **Architectures** -- MLP, ResNet (BatchNorm, in-place ReLU, skip connections),
   LSTM, GRU, Transformer encoder, GPT-style causal decoder (nn.MultiheadAttention
   with a mask) and a tanh/sigmoid autoencoder. For each one:
   training is bit-identical with and without hooks; one TrainingState per step;
   the loss, every per-parameter gradient norm and every per-parameter weight
   delta match an independent measurement; activation statistics cover the
   selected layers and gradients flow back through them; no invariant reports a
   critical failure on the clean run; detach() removes every hook.
2. **Planted bugs** -- the scenarios of examples/planted_bugs_torch.py, run
   through RuntimeHooks with the default invariant registry, must give the same
   verdicts as the stopgap adapter (eval_mode, frozen, high_lr, dead_relu,
   random_labels detected; no critical failure on the clean baseline).
3. **Extra signals** -- NaN input, saturated tanh, a loss spike, a huge update,
   histograms, a consumer thread on the stream, and a torch.compile-d model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..invariants.base import Severity, TrainingState
from ..invariants.registry import get_default_registry
from .hooks import HookConfig, RuntimeHooks

ORIGINAL_BACKWARD = torch.Tensor.backward


# ------------------------------------------------------------ architectures
class SeqClassifier(nn.Module):
    def __init__(self, cell: str = "lstm", vocab: int = 50, d: int = 32, classes: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(d, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(64, classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.emb(tokens))
        return self.head(out[:, -1])


class BasicBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class TinyResNet(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(BasicBlock(16), BasicBlock(16))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.pool(self.layer1(self.stem(x))), 1))


class EncoderClassifier(nn.Module):
    def __init__(self, vocab: int = 50, d: int = 32, classes: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(d, classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.emb(tokens)).mean(1))


class CausalBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = 40, T: int = 16, d: int = 32) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([CausalBlock(d, 4) for _ in range(2)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


@dataclass
class Arch:
    name: str
    build: Callable[[], nn.Module]
    batch: Callable[[int, torch.device], Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optimizer: Callable[[Any], torch.optim.Optimizer]


def _gen(step: int) -> torch.Generator:
    return torch.Generator().manual_seed(1000 + step)


def _cls_batch(shape: Tuple[int, ...], classes: int, tokens: Optional[int] = None):
    def f(step: int, dev: torch.device):
        g = _gen(step)
        x = torch.randint(0, tokens, shape, generator=g) if tokens else torch.randn(*shape, generator=g)
        y = torch.randint(0, classes, (shape[0],), generator=g)
        return (x.to(dev),), y.to(dev)
    return f


def _lm_batch(step: int, dev: torch.device):
    seq = torch.randint(0, 40, (8, 17), generator=_gen(step)).to(dev)
    return (seq[:, :-1],), seq[:, 1:]


def _ae_batch(step: int, dev: torch.device):
    x = torch.rand(32, 32, generator=_gen(step)).to(dev)
    return (x,), x


ARCHS: Dict[str, Arch] = {
    "mlp": Arch("mlp", lambda: nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU(),
                                             nn.Linear(64, 5)),
                _cls_batch((32, 20), 5), F.cross_entropy, lambda p: torch.optim.Adam(p, lr=1e-3)),
    "resnet": Arch("resnet", TinyResNet, _cls_batch((16, 3, 16, 16), 10), F.cross_entropy,
                   lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    "lstm": Arch("lstm", lambda: SeqClassifier("lstm"), _cls_batch((16, 12), 4, tokens=50), F.cross_entropy,
                 lambda p: torch.optim.Adam(p, lr=1e-3)),
    "gru": Arch("gru", lambda: SeqClassifier("gru"), _cls_batch((16, 12), 4, tokens=50), F.cross_entropy,
                lambda p: torch.optim.AdamW(p, lr=1e-3)),
    "transformer": Arch("transformer", EncoderClassifier, _cls_batch((16, 12), 4, tokens=50), F.cross_entropy,
                        lambda p: torch.optim.AdamW(p, lr=1e-3)),
    "gpt": Arch("gpt", TinyGPT, _lm_batch, lambda o, y: F.cross_entropy(o.reshape(-1, o.size(-1)), y.reshape(-1)),
                lambda p: torch.optim.AdamW(p, lr=3e-4)),
    "autoencoder": Arch("autoencoder", lambda: nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 32), nn.Sigmoid()),
                        _ae_batch, F.mse_loss, lambda p: torch.optim.SGD(p, lr=0.1)),
}


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------- training
def _train(arch: Arch, steps: int, dev: torch.device, *, config: Optional[HookConfig], with_hooks: bool,
           mutate: Optional[Callable[[nn.Module], None]] = None,
           batch_fault: Optional[Callable[[int, tuple], tuple]] = None,
           optimizer: Optional[Callable[[Any], torch.optim.Optimizer]] = None,
           compile_backend: Optional[str] = None) -> Dict[str, Any]:
    torch.manual_seed(0)
    model = arch.build().to(dev)
    if mutate is not None:
        mutate(model)
    opt = (optimizer or arch.optimizer)(model.parameters())
    run_model = torch.compile(model, backend=compile_backend) if compile_backend else model
    hooks = RuntimeHooks(run_model, opt, config, invariants=get_default_registry()).attach() if with_hooks else None
    named = [(n, p) for n, p in model.named_parameters()]
    manual: List[Dict[str, Any]] = []
    step_times: List[float] = []
    for step in range(steps):
        t0 = time.perf_counter()
        run_model.train()
        x, y = arch.batch(step, dev)
        if batch_fault is not None:
            x = batch_fault(step, x)
        opt.zero_grad(set_to_none=True)
        loss = arch.loss(run_model(*x), y)
        loss.backward()
        grads = {n: float(p.grad.detach().double().norm()) for n, p in named if p.grad is not None}
        before = {n: p.detach().clone() for n, p in named}
        opt.step()
        deltas = {n: float((p.detach().double() - before[n].double()).norm()) for n, p in named}
        manual.append({"loss": float(loss.detach()), "grads": grads, "deltas": deltas})
        if dev.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    timed = sorted(step_times[1:] or step_times)
    states: List[TrainingState] = []
    failures: List[Any] = []
    detach_clean, n_layers = True, 0
    if hooks is not None:
        hooks.flush()
        states = hooks.stream.drain()
        failures = list(hooks.failures)
        n_layers = len(hooks.layers)
        hooks.detach()
        detach_clean = (
            all(len(m._forward_hooks) == 0 for m in run_model.modules())
            and len(opt._optimizer_step_pre_hooks) == 0 and len(opt._optimizer_step_post_hooks) == 0
            and torch.Tensor.backward is ORIGINAL_BACKWARD
        )
    return {"states": states, "failures": failures, "manual": manual, "seconds": timed[len(timed) // 2],
            "detach_clean": detach_clean, "n_layers": n_layers, "params": [p.detach().clone() for p in model.parameters()]}


def _close(a: Optional[float], b: float, rtol: float) -> bool:
    return a is not None and math.isfinite(a) and abs(a - b) <= rtol * max(1.0, abs(b))


def _dicts_close(got: Dict[str, float], want: Dict[str, float], rtol: float) -> bool:
    return set(got) == set(want) and all(_close(got[k], v, rtol) for k, v in want.items())


def validate_architecture(name: str, steps: int = 5, device: Optional[torch.device] = None,
                          config: Optional[HookConfig] = None) -> Dict[str, Any]:
    arch = ARCHS[name]
    dev = device or default_device()
    cfg = config or HookConfig(histogram_bins=8)
    base = _train(arch, steps, dev, config=None, with_hooks=False)
    hooked = _train(arch, steps, dev, config=cfg, with_hooks=True)
    states, manual = hooked["states"], hooked["manual"]
    non_intrusive = ([m["loss"] for m in base["manual"]] == [m["loss"] for m in manual]
                     and all(torch.equal(a, b) for a, b in zip(base["params"], hooked["params"])))
    one_per_step = len(states) == steps and [s.step for s in states] == list(range(1, steps + 1))
    loss_ok = one_per_step and all(_close(s.loss_history[-1] if s.loss_history else None, m["loss"], 1e-5)
                                   for s, m in zip(states, manual))
    grads_ok = one_per_step and all(_dicts_close(s.grad_norms, m["grads"], 1e-3) for s, m in zip(states, manual))
    deltas_ok = one_per_step and all(_dicts_close(s.weight_deltas, m["deltas"], 1e-3) for s, m in zip(states, manual))
    act_cov = min((sum("mean" in v for v in s.activation_stats.values()) / max(1, hooked["n_layers"]) for s in states),
                  default=0.0)
    grad_cov = min((sum("grad_norm" in v for v in s.activation_stats.values()) for s in states), default=0)
    hist_ok = all(s.extra.get("histograms") for s in states) if cfg.histogram_bins else True
    critical = sorted({r.invariant_id for _, r in hooked["failures"] if r.severity == Severity.CRITICAL})
    result = {
        "arch": name, "device": str(dev), "steps": steps, "layers": hooked["n_layers"], "snapshots": len(states),
        "non_intrusive": non_intrusive, "loss_matches": loss_ok, "grad_norms_match": grads_ok,
        "weight_deltas_match": deltas_ok, "activation_coverage": round(act_cov, 3), "layers_with_grad_flow": grad_cov,
        "histograms": hist_ok, "critical_failures": critical, "detach_clean": hooked["detach_clean"],
        "overhead": round(hooked["seconds"] / max(base["seconds"], 1e-9), 2),
    }
    result["passed"] = all([non_intrusive, one_per_step, loss_ok, grads_ok, deltas_ok, act_cov >= 0.8, grad_cov >= 1,
                            hist_ok, not critical, hooked["detach_clean"]])
    return result


# ----------------------------------------------------------- planted bugs
EXPECTED = {
    "none": set(),
    "eval_mode": {"train_eval_mode_mismatch"},
    "frozen": {"frozen_weights"},
    "high_lr": {"gradient_norm_bounds", "activation_health"},
    "dead_relu": {"activation_health"},
    "random_labels": {"loss_trend"},
}


class PlantedNet(nn.Module):
    """The model of examples/planted_bugs_torch.py."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.act1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.out = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.bn(self.act1(self.fc1(x))))
        return self.out(self.act2(self.fc2(x)))


def run_planted(bug: str, steps: int = 200, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Same scenario as examples/planted_bugs_torch.py::run_one, monitored by RuntimeHooks."""
    dev = device or torch.device("cpu")
    torch.manual_seed(0)
    w_true = torch.randn(10, 3).to(dev)
    model = PlantedNet().to(dev)
    if bug == "dead_relu":
        with torch.no_grad():
            model.fc1.bias.fill_(-100.0)
    lr = 500.0 if bug == "high_lr" else 0.05
    params = [p for n, p in model.named_parameters() if not (bug == "frozen" and n.startswith("fc2"))]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    hooks = RuntimeHooks(model, opt, invariants=get_default_registry()).attach()
    for step in range(1, steps + 1):
        x = torch.randn(64, 10).to(dev)
        y = (x @ w_true).argmax(dim=1)
        if bug == "random_labels":
            y = torch.randint(0, 3, (64,)).to(dev)
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if not math.isfinite(loss.item()):
            break
        if step % 50 == 0:
            if bug != "eval_mode":
                model.eval()
            with torch.no_grad():
                model(torch.randn(64, 10).to(dev))
            if bug != "eval_mode":
                model.train()
    hooks.detach()
    fired = Counter(r.invariant_id for _, r in hooks.failures)
    critical = {r.invariant_id for _, r in hooks.failures if r.severity == Severity.CRITICAL}
    ok = (not critical) if bug == "none" else bool(set(fired) & EXPECTED[bug])
    return {"bug": bug, "ok": ok, "fired": dict(fired), "snapshots": hooks.stream.published}


def validate_planted_bugs(device: Optional[torch.device] = None) -> Dict[str, Dict[str, Any]]:
    return {bug: run_planted(bug, device=device) for bug in EXPECTED}


# ------------------------------------------------------------ extra signals
def validate_extra_signals(device: Optional[torch.device] = None) -> Dict[str, bool]:
    dev = device or default_device()
    mlp, ae = ARCHS["mlp"], ARCHS["autoencoder"]
    out: Dict[str, bool] = {}

    def nan_input(step: int, x: tuple) -> tuple:
        if step == 2:
            x = (x[0].clone(),)
            x[0][0, 0] = float("nan")
        return x

    r = _train(mlp, 4, dev, config=HookConfig(), with_hooks=True, batch_fault=nan_input)
    s = r["states"][2]
    out["NaN input -> nan_count + activation_health CRITICAL"] = (
        s.activation_stats["0[Linear]"]["nan_count"] > 0
        and any(f.invariant_id == "activation_health" and f.severity == Severity.CRITICAL and st == 3
                for st, f in r["failures"])
    )

    def saturate(model: nn.Module) -> None:
        with torch.no_grad():
            model[0].weight.mul_(1000.0)

    r = _train(ae, 3, dev, config=HookConfig(), with_hooks=True, mutate=saturate)
    out["huge weights -> tanh saturated_frac >= 0.95"] = r["states"][0].activation_stats["1[Tanh]"]["saturated_frac"] >= 0.95

    def spike(step: int, x: tuple) -> tuple:
        return (x[0] * 1e3,) if step == 25 else x

    r = _train(mlp, 28, dev, config=HookConfig(), with_hooks=True, batch_fault=spike)
    flags = [s.extra["loss"]["spike"] for s in r["states"]]
    out["scaled batch -> loss spike flagged only at that step"] = flags[25] and sum(flags) == 1

    r = _train(mlp, 3, dev, config=HookConfig(), with_hooks=True, optimizer=lambda p: torch.optim.SGD(p, lr=1e4))
    out["lr x1e4 -> update_ratio > 1"] = (r["states"][0].extra["update_ratio"] or 0) > 1

    r = _train(mlp, 2, dev, config=HookConfig(histogram_bins=16), with_hooks=True)
    h = r["states"][0].extra.get("histograms", {})
    out["histograms are normalised"] = bool(h) and all(abs(sum(v["density"]) - 1) < 1e-4 for v in h.values())
    return out


def validate_stream_consumer(device: Optional[torch.device] = None, steps: int = 12) -> bool:
    """A separate agent thread receives every snapshot while training runs."""
    dev = device or default_device()
    arch = ARCHS["mlp"]
    torch.manual_seed(0)
    model = arch.build().to(dev)
    opt = arch.optimizer(model.parameters())
    hooks = RuntimeHooks(model, opt, HookConfig(max_pending=1)).attach()
    got: List[int] = []

    def agent() -> None:
        for state in hooks.stream:
            got.append(state.step)

    t = threading.Thread(target=agent, daemon=True)
    t.start()
    for step in range(steps):
        x, y = arch.batch(step, dev)
        opt.zero_grad()
        arch.loss(model(*x), y).backward()
        opt.step()
    hooks.detach()
    hooks.stream.close()
    t.join(timeout=10)
    return got == list(range(1, steps + 1))


def validate_compiled(device: Optional[torch.device] = None, backend: str = "aot_eager") -> bool:
    """A torch.compile-d model trains unchanged and still yields loss/gradient/weight snapshots."""
    dev = device or default_device()
    try:
        r = _train(ARCHS["mlp"], 4, dev, config=HookConfig(), with_hooks=True, compile_backend=backend)
    except Exception:
        return False
    return len(r["states"]) == 4 and all(
        _dicts_close(s.grad_norms, m["grads"], 1e-3) and _close(s.loss_history[-1], m["loss"], 1e-5)
        for s, m in zip(r["states"], r["manual"]))


# ---------------------------------------------------------------------- cli
def run_all(device: Optional[torch.device] = None, steps: int = 5, archs: Optional[List[str]] = None) -> Dict[str, Any]:
    dev = device or default_device()
    arch_results = [validate_architecture(a, steps, dev) for a in (archs or list(ARCHS))]
    planted = validate_planted_bugs(dev)
    extra = validate_extra_signals(dev)
    extra["stream consumer thread receives every snapshot"] = validate_stream_consumer(dev)
    extra["torch.compile (aot_eager) still produces snapshots"] = validate_compiled(dev)
    passed = (all(r["passed"] for r in arch_results) and all(p["ok"] for p in planted.values())
              and all(extra.values()))
    return {"device": str(dev), "torch": torch.__version__, "architectures": arch_results, "planted_bugs": planted,
            "extra": extra, "passed": passed}


def _print(report: Dict[str, Any]) -> None:
    cols = ["arch", "layers", "snapshots", "non_intrusive", "loss_matches", "grad_norms_match", "weight_deltas_match",
            "activation_coverage", "layers_with_grad_flow", "detach_clean", "overhead", "passed"]
    print(f"device={report['device']} torch={report['torch']}\n")
    print(" | ".join(cols))
    for r in report["architectures"]:
        print(" | ".join(str(r[c]) for c in cols))
    print("\nplanted bugs (examples/planted_bugs_torch.py scenarios, default invariants):")
    for bug, p in report["planted_bugs"].items():
        print(f"  {'PASS' if p['ok'] else 'FAIL'}  {bug:14s} fired={p['fired'] or 'nothing'}")
    print("\nextra signals:")
    for k, v in report["extra"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nOVERALL: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the Module 1 runtime hooks across standard architectures.")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:1 (default: cuda if available)")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--arch", nargs="*", choices=list(ARCHS))
    ap.add_argument("--json", help="write the full report to this file")
    a = ap.parse_args(argv)
    report = run_all(torch.device(a.device) if a.device else None, a.steps, a.arch)
    _print(report)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
