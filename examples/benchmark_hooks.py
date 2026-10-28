"""Overhead and memory benchmark of the Runtime Monitor Agent on heavy workloads.

    python examples/benchmark_hooks.py                         # ResNet-50, Transformer, MLP on the best device
    python examples/benchmark_hooks.py --device cuda --json bench.json --markdown bench.md
    python examples/benchmark_hooks.py --quick                 # tiny models, a few seconds (smoke test, no budget gate)

For every workload it trains the same model, in interleaved blocks, under four modes:

    baseline           no monitoring
    hooks_every_step   RuntimeHooks only, every step monitored
    agent_every_step   full agent (hooks + invariant thread + alerts), every step monitored
    agent_governed     full agent with the overhead governor (default 5% budget)

Overhead = mean step time of the mode / mean step time of the baseline blocks of the same
repetition - 1 (median over repetitions). Each step is bracketed by a device sync. It also
checks GPU memory (CUDA / MPS: exact allocated bytes) for growth during a monitored run and
for anything left behind after detach, and uses the PyTorch profiler to list the operators
monitoring adds. Exit code 1 if the governed overhead exceeds the budget or memory leaks.

Methodology and results: docs/overhead_benchmark.md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from silent_bug_agent.agents.runtime_monitor_agent import MonitorConfig, RuntimeMonitorAgent
from silent_bug_agent.runtime import HookConfig, RuntimeHooks
from silent_bug_agent.runtime.profiling import (MemoryLeakDetector, device_allocated, profile_diff, profile_steps,
                                                synchronize, time_steps)

MODES = ("baseline", "hooks_every_step", "agent_every_step", "agent_governed")


# ---------------------------------------------------------------- workloads
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin: int, width: int, stride: int = 1) -> None:
        super().__init__()
        cout = width * self.expansion
        self.conv1 = nn.Conv2d(cin, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return self.relu(self.bn3(self.conv3(out)) + idt)


class ResNet50(nn.Module):
    """Standard ResNet-50 (25.6M parameters, identical layout to torchvision.models.resnet50)."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        layers, cin = [], 64
        for width, blocks, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)):
            seq = []
            for b in range(blocks):
                seq.append(Bottleneck(cin, width, stride if b == 0 else 1))
                cin = width * 4
            layers.append(nn.Sequential(*seq))
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


class TransformerLM(nn.Module):
    """BERT-base-sized pre-norm Transformer encoder with an LM head (6 layers, d_model 512 by default)."""

    def __init__(self, vocab: int, d: int, heads: int, layers: int, ff: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, tokens):
        return self.head(self.norm(self.encoder(self.emb(tokens))))


def mlp(sizes: List[int]) -> nn.Module:
    mods: List[nn.Module] = []
    for a, b in zip(sizes, sizes[1:]):
        mods += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*mods[:-1])


@dataclass
class Workload:
    name: str
    describe: str
    build: Callable[[torch.device], Tuple[nn.Module, torch.optim.Optimizer, Callable[[], Any]]]


def _classifier_step(model, opt, x, y):
    crit = nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        return loss

    return step


def workloads(device: torch.device, quick: bool) -> Dict[str, Workload]:
    big = device.type != "cpu"
    if quick:
        def build_resnet(dev):
            m = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(), nn.Linear(16, 10)).to(dev)
            o = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
            return m, o, _classifier_step(m, o, torch.randn(8, 3, 32, 32, device=dev), torch.randint(0, 10, (8,), device=dev))

        def build_tf(dev):
            m = TransformerLM(100, 32, 4, 2, 64).to(dev)
            o = torch.optim.AdamW(m.parameters(), lr=1e-3)
            tok = torch.randint(0, 100, (4, 16), device=dev)
            crit = nn.CrossEntropyLoss()

            def step():
                o.zero_grad(set_to_none=True)
                crit(m(tok).flatten(0, 1), tok.flatten()).backward()
                o.step()
            return m, o, step

        def build_mlp(dev):
            m = mlp([64, 128, 128, 10]).to(dev)
            o = torch.optim.AdamW(m.parameters(), lr=1e-3)
            return m, o, _classifier_step(m, o, torch.randn(32, 64, device=dev), torch.randint(0, 10, (32,), device=dev))

        return {"resnet50": Workload("resnet50", "tiny CNN (quick mode)", build_resnet),
                "transformer": Workload("transformer", "tiny Transformer (quick mode)", build_tf),
                "mlp": Workload("mlp", "tiny MLP (quick mode)", build_mlp)}

    bs_img = 32 if device.type == "cuda" else (16 if big else 8)
    bs_tok = 16 if device.type == "cuda" else (8 if big else 4)

    def build_resnet(dev):
        m = ResNet50().to(dev)
        o = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        x = torch.randn(bs_img, 3, 224, 224, device=dev)
        return m, o, _classifier_step(m, o, x, torch.randint(0, 1000, (bs_img,), device=dev))

    def build_tf(dev):
        m = TransformerLM(32000, 512, 8, 6, 2048).to(dev)
        o = torch.optim.AdamW(m.parameters(), lr=1e-4)
        tok = torch.randint(0, 32000, (bs_tok, 256), device=dev)
        crit = nn.CrossEntropyLoss()

        def step():
            o.zero_grad(set_to_none=True)
            loss = crit(m(tok[:, :-1]).flatten(0, 1), tok[:, 1:].flatten())
            loss.backward()
            o.step()
        return m, o, step

    def build_mlp(dev):
        m = mlp([1024, 4096, 4096, 4096, 1000]).to(dev)
        o = torch.optim.AdamW(m.parameters(), lr=1e-4)
        return m, o, _classifier_step(m, o, torch.randn(512, 1024, device=dev), torch.randint(0, 1000, (512,), device=dev))

    return {
        "resnet50": Workload("resnet50", f"ResNet-50, {bs_img}x3x224x224, SGD+momentum", build_resnet),
        "transformer": Workload("transformer", f"Transformer LM (6 layers, d=512, 8 heads, vocab 32k), "
                                               f"{bs_tok}x255 tokens, AdamW", build_tf),
        "mlp": Workload("mlp", "MLP 1024-4096-4096-4096-1000, batch 512, AdamW", build_mlp),
    }


# ---------------------------------------------------------------- the runner
class _Mode:
    """Context manager that turns one monitoring mode on and off."""

    def __init__(self, mode: str, model: nn.Module, opt: torch.optim.Optimizer, budget: float) -> None:
        self.mode, self.model, self.opt, self.budget = mode, model, opt, budget
        self.hooks: Optional[RuntimeHooks] = None
        self.session: Any = None

    def __enter__(self) -> "_Mode":
        if self.mode == "hooks_every_step":
            self.hooks = RuntimeHooks(self.model, self.opt, HookConfig(stream_capacity=256, max_loss_history=2048)).attach()
        elif self.mode.startswith("agent"):
            agent = RuntimeMonitorAgent(MonitorConfig())
            budget = None if self.mode == "agent_every_step" else self.budget
            self.session = agent.watch(self.model, self.opt, overhead_budget=budget)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.hooks is not None:
            self.hooks.detach()
        if self.session is not None:
            self.session.close()

    def governor(self) -> Optional[Dict[str, Any]]:
        if self.session is None:
            return None
        ch = next(iter(self.session.channels.values()))
        return ch.governor.stats.to_dict() if ch.governor is not None else None

    def findings(self) -> List[str]:
        return [f.bug_id for f in self.session.findings()] if self.session is not None else []


def run_workload(w: Workload, device: torch.device, *, steps: int, reps: int, budget: float, warmup: int,
                 governed_warmup: int, profile: bool) -> Dict[str, Any]:
    torch.manual_seed(0)
    model, opt, step = w.build(device)
    n_params = sum(p.numel() for p in model.parameters())
    time_steps(step, warmup, device)
    times: Dict[str, List[List[float]]] = {m: [] for m in MODES}
    governor: Optional[Dict[str, Any]] = None
    findings: List[str] = []
    for rep in range(reps):
        order = MODES[rep % len(MODES):] + MODES[:rep % len(MODES)]  # rotate the order to cancel drift
        for mode in order:
            with _Mode(mode, model, opt, budget) as ctx:
                n_timed = steps
                if mode != "baseline":
                    time_steps(step, governed_warmup if mode == "agent_governed" else 2, device)  # governor: 8 calibrate + 6 probe
                if mode == "agent_governed":
                    governor = ctx.governor()
                    n = (governor or {}).get("every_n_steps") or 1
                    n_timed = n * max(1, round(steps / n))  # whole sampling cycles: one monitored step per N
                times[mode].append(time_steps(step, n_timed, device))
                if mode.startswith("agent"):
                    findings += ctx.findings()

    def overhead(mode: str) -> float:
        per_rep = [statistics.fmean(t) / statistics.fmean(b) - 1 for t, b in zip(times[mode], times["baseline"])]
        return statistics.median(per_rep)

    result: Dict[str, Any] = {
        "workload": w.name, "description": w.describe, "parameters": n_params,
        "baseline_step_ms": statistics.median([statistics.fmean(t) for t in times["baseline"]]) * 1e3,
        "overhead": {m: overhead(m) for m in MODES if m != "baseline"},
        "governor": governor,
        "false_alarms_on_healthy_training": sorted(set(findings)),
    }
    result["memory"] = memory_check(model, opt, step, device, steps=max(steps, 8))
    if profile:
        base = profile_steps(step, 2, device)
        with _Mode("hooks_every_step", model, opt, budget):
            step()
            mon = profile_steps(step, 2, device)
        result["profiler_added_ops"] = profile_diff(base, mon, top=8)
    del model, opt, step
    return result


def memory_check(model: nn.Module, opt: torch.optim.Optimizer, step: Callable[[], Any], device: torch.device,
                 steps: int, cycles: int = 2) -> Dict[str, Any]:
    """Device memory: growth during a monitored run, constant overhead, and residue after detach.

    An unmonitored control block shows how much the training step itself leaves behind (allocator or
    library caches), and the monitored run is repeated ``cycles`` times: a leak in the monitor shows in
    every cycle, a one-off allocation in only one, so the residue reported is the smallest over cycles."""
    if device_allocated(device) is None:
        return {"device": str(device), "checked": False, "note": "CPU tensors have no exact allocation counter"}
    for _ in range(3):
        step()
    control = MemoryLeakDetector(device, warmup=3)
    control.mark_before_attach()
    for _ in range(steps):
        step()
        control.sample()
    control.mark_after_detach()
    ctl = control.report()
    reports, overhead = [], 0
    for _ in range(cycles):
        det = MemoryLeakDetector(device, warmup=3)
        det.mark_before_attach()
        base = device_allocated(device) or 0
        with _Mode("agent_every_step", model, opt, 0.05):
            for _ in range(steps):
                step()
                det.sample()
            overhead = (device_allocated(device) or 0) - base
        det.mark_after_detach()
        reports.append(det.report())
    growth = max(r.device_growth_bytes or 0 for r in reports)
    residual = min(r.detach_residual_bytes or 0 for r in reports)
    return {"device": str(device), "checked": True, "samples": sum(r.samples for r in reports),
            "device_growth_bytes": growth, "detach_residual_bytes": residual,
            "detach_residual_per_cycle": [r.detach_residual_bytes for r in reports],
            "control_growth_bytes": ctl.device_growth_bytes, "control_residual_bytes": ctl.detach_residual_bytes,
            "monitoring_overhead_bytes": overhead, "leak": growth > 0 or residual > 0}


def environment(device: torch.device) -> Dict[str, Any]:
    name = str(device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        name = f"Apple GPU (MPS), {platform.machine()}"
    return {"date": _dt.date.today().isoformat(), "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "device": name, "cpu_threads": torch.get_num_threads()}


def to_markdown(env: Dict[str, Any], results: List[Dict[str, Any]], budget: float) -> str:
    lines = [f"Environment: {env['device']}; torch {env['torch']}; Python {env['python']}; {env['platform']}; "
             f"{env['cpu_threads']} CPU threads; {env['date']}.", "",
             "| workload | params | baseline step | hooks every step | agent every step | agent governed | governor N | GPU memory growth | left after detach |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        o, g, m = r["overhead"], r["governor"] or {}, r["memory"]
        mem = (f"{m['device_growth_bytes']} B", f"{m['detach_residual_bytes']} B") if m.get("checked") else ("n/a", "n/a")
        lines.append(f"| {r['description']} | {r['parameters'] / 1e6:.1f}M | {r['baseline_step_ms']:.0f} ms | "
                     f"{o['hooks_every_step']:+.1%} | {o['agent_every_step']:+.1%} | **{o['agent_governed']:+.1%}** | "
                     f"{g.get('every_n_steps', '-')} | {mem[0]} | {mem[1]} |")
    lines.append("")
    lines.append(f"Budget: {budget:.0%}. Overhead = mean step time vs. interleaved no-monitoring blocks (median of repetitions).")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--workloads", default="resnet50,transformer,mlp")
    ap.add_argument("--steps", type=int, default=10, help="timed steps per block")
    ap.add_argument("--reps", type=int, default=3, help="repetitions of the interleaved blocks")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny models (smoke test)")
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    a = ap.parse_args(argv)
    if a.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = a.device
    device = torch.device(dev)
    env = environment(device)
    print(f"# Runtime Monitor Agent overhead benchmark on {env['device']} (torch {env['torch']})", flush=True)
    ws = workloads(device, a.quick)
    results = []
    for name in [n.strip() for n in a.workloads.split(",") if n.strip()]:
        w = ws[name]
        print(f"- {w.describe} ...", flush=True)
        r = run_workload(w, device, steps=a.steps, reps=a.reps, budget=a.budget, warmup=3,
                         governed_warmup=16, profile=not a.no_profile)
        o = r["overhead"]
        print(f"  baseline {r['baseline_step_ms']:.1f} ms/step | hooks every step {o['hooks_every_step']:+.1%} | "
              f"agent every step {o['agent_every_step']:+.1%} | agent governed {o['agent_governed']:+.1%} "
              f"(N={(r['governor'] or {}).get('every_n_steps')})", flush=True)
        results.append(r)
    md = to_markdown(env, results, a.budget)
    print()
    print(md)
    # tiny quick-mode models have sub-millisecond steps: a smoke test, not held to the budget
    failures = [] if a.quick else [
        f"{r['workload']}: governed overhead {r['overhead']['agent_governed']:.1%} > {a.budget:.0%}"
        for r in results if r["overhead"]["agent_governed"] > a.budget]
    failures += [f"{r['workload']}: memory leak {r['memory']}" for r in results if r["memory"].get("leak")]
    out = {"environment": env, "budget": a.budget, "steps": a.steps, "reps": a.reps, "quick": a.quick,
           "results": results, "failures": failures}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    for line in failures:
        print("FAIL", line)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
