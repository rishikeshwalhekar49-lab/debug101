"""Runtime Monitor Agent: contract, tools and integration tests with synthetic training loops.

The first group needs no torch (contract, alert logic, trends). The rest run real
training loops and are skipped when torch is not installed.
"""

import importlib.util
import io
import contextlib
import math
import sys
import textwrap
from pathlib import Path

import pytest

from silent_bug_agent.agents import runtime_monitor_agent as rma
from silent_bug_agent.agents.runtime_monitor_agent import MonitorConfig, RuntimeMonitorAgent
from silent_bug_agent.findings import Finding, Report
from silent_bug_agent.invariants.base import Severity, TrainingState
from silent_bug_agent.schemas import AgentState

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

REPO = Path(__file__).resolve().parents[1]
PLANTED = REPO / "examples" / "sample_scripts" / "planted_runtime_bugs.py"


# =========================================================== contract (no torch)
def test_contract_call_populates_detection_report(monkeypatch):
    seen = {}

    def fake_monitor(self, path, args=None):
        seen["path"], seen["args"] = path, args
        return Report(findings=[Finding("frozen_layer_no_grad", "runtime", "m", Severity.WARNING, 0.9, file=str(path))],
                      files_scanned=[str(path)])

    monkeypatch.setattr(RuntimeMonitorAgent, "monitor_script", fake_monitor)
    state = AgentState(run_id="r1", target_file="train.py", extra={"script_args": ["--epochs", "1"]})
    out = RuntimeMonitorAgent()(state)
    assert out is state
    assert isinstance(state.detection_report, Report)
    assert [f.bug_id for f in state.detection_report.findings] == ["frozen_layer_no_grad"]
    assert seen == {"path": Path("train.py"), "args": ["--epochs", "1"]}
    assert "runtime_monitor" in state.extra


def test_contract_call_keeps_existing_findings(monkeypatch):
    static = Finding("missing_zero_grad", "static", "s", Severity.WARNING, 0.85, file="train.py", line=10)
    runtime_dup = Finding("missing_zero_grad", "runtime", "r", Severity.CRITICAL, 0.9, file="train.py", line=11)
    runtime_new = Finding("dead_relu", "runtime", "d", Severity.WARNING, 0.7, file="train.py", line=5)
    monkeypatch.setattr(RuntimeMonitorAgent, "monitor_script",
                        lambda self, p, a=None: Report(findings=[runtime_dup, runtime_new], files_scanned=["train.py"]))
    state = AgentState(target_file="train.py", detection_report=Report(findings=[static], files_scanned=["train.py"]))
    RuntimeMonitorAgent()(state)
    assert [f.detector for f in state.detection_report.findings] == ["static", "runtime"]


def test_contract_empty_target_and_missing_script():
    state = RuntimeMonitorAgent()(AgentState())
    assert state.detection_report.findings == [] and "target_file" in state.detection_report.diagnostics[0]
    if HAS_TORCH:
        r = RuntimeMonitorAgent().monitor_script(Path("does/not/exist.py"))
        assert r.findings == [] and "not found" in r.diagnostics[0]


def test_without_torch_returns_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(rma, "_torch_missing", lambda: "runtime monitor skipped: PyTorch is not installed")
    script = tmp_path / "t.py"
    script.write_text("print('hi')\n")
    r = RuntimeMonitorAgent().monitor_script(script)
    assert r.findings == [] and "PyTorch is not installed" in r.diagnostics[0]


def test_alternative_bug_ids_exist_in_taxonomy():
    from silent_bug_agent.runtime.anomaly import ALTERNATIVE_BUG_IDS
    from silent_bug_agent.taxonomy.loader import load_taxonomy

    ids = {p.id for p in load_taxonomy()}
    for bug, alts in ALTERNATIVE_BUG_IDS.items():
        assert bug in ids, bug
        assert set(alts) <= ids, (bug, set(alts) - ids)


# ============================================================ tools (no torch)
def _signal(rule, bug, sev=Severity.WARNING, step=1, conf=0.6, **kw):
    from silent_bug_agent.runtime.trends import Signal

    return Signal(rule, bug, sev, conf, f"{rule} at {step}", step, **kw)


def test_dispatcher_confirms_noisy_rules_after_repeats_and_critical_at_once():
    from silent_bug_agent.runtime.alerts import AlertDispatcher

    d = AlertDispatcher()
    events = []
    d.subscribe(lambda a, kind: events.append((kind, a.bug_id, a.severity)))
    d.ingest("m", [_signal("invariant:gradient_norm_bounds", "vanishing_gradients", step=1)])
    assert events == [] and d.suppressed == 1  # one transient reading is not a finding
    d.ingest("m", [_signal("invariant:gradient_norm_bounds", "vanishing_gradients", step=2)])
    assert events == [("new", "vanishing_gradients", Severity.WARNING)]
    d.ingest("m", [_signal("runtime:nonfinite_loss", "exploding_gradients", Severity.CRITICAL, step=3, conf=0.95)])
    assert events[-1] == ("new", "exploding_gradients", Severity.CRITICAL)


def test_dispatcher_aggregates_escalates_and_isolates_callback_errors():
    from silent_bug_agent.runtime.alerts import AlertDispatcher

    d = AlertDispatcher()
    kinds = []
    d.subscribe(lambda a, kind: 1 / 0)  # a broken consumer
    d.subscribe(lambda a, kind: kinds.append(kind))
    for step in range(1, 6):  # two layers fire in the same snapshot: counted once per step
        d.ingest("m", [_signal("trend:dead_units", "dead_relu", step=step, offenders={"a": 0.6}),
                       _signal("trend:dead_units", "dead_relu", step=step, offenders={"b": 0.7})])
    d.ingest("m", [_signal("invariant:activation_health", "dead_relu", Severity.CRITICAL, step=6)])
    (alert,) = d.confirmed()
    assert alert.rules["trend:dead_units"].count == 5
    assert set(alert.offenders) == {"a", "b"}
    assert kinds == ["new", "escalated"] and d.callback_errors == 2
    f = alert.to_finding()
    assert f.detector == "runtime" and f.severity == Severity.CRITICAL and 0 < f.confidence <= 0.99
    assert f.extra["first_step"] == 1 and f.extra["last_step"] == 6
    assert "learning_rate_misconfigured" in f.extra["alternative_bug_ids"]


def _state(step, loss, grad=1.0, update=0.01, history=None, acts=None):
    return TrainingState(step=step, loss_history=history or [], activation_stats=acts or {},
                         extra={"loss": {"value": loss, "spike": False}, "global_grad_norm": grad,
                                "update_ratio": update, "lrs": [0.1]})


def test_trend_analyzer_grad_growth_and_noise_robust_divergence():
    from silent_bug_agent.runtime.trends import TrendAnalyzer

    t = TrendAnalyzer()
    fired = []
    for s in range(1, 13):  # gradient norm doubles every step
        fired += [x.rule for x in t.observe(_state(s, 1.0, grad=0.5 * 2 ** s))]
    assert "trend:grad_norm_growth" in fired

    t = TrendAnalyzer()
    noisy = [0.2 + (0.15 if i % 3 == 0 else -0.1) for i in range(60)]  # flat but noisy small loss
    assert not [x for i, v in enumerate(noisy) for x in t.observe(_state(i + 1, v)) if x.rule == "trend:loss_divergence"]

    t = TrendAnalyzer()
    rising = [1.0 * 1.12 ** i for i in range(30)]
    assert any(x.rule == "trend:loss_divergence" for i, v in enumerate(rising) for x in t.observe(_state(i + 1, v)))


def test_trend_analyzer_dead_units_need_patience():
    from silent_bug_agent.runtime.trends import TrendAnalyzer

    t = TrendAnalyzer()
    out = []
    for s in range(1, 7):
        out.append([x.rule for x in t.observe(_state(s, 1.0, acts={"act[ReLU]": {"dead_unit_frac": 0.6, "zero_frac": 0.7}}))])
    assert out[:4] == [[], [], [], []] and "trend:dead_units" in out[4]


def test_converged_loss_plateau_is_not_reported():
    from silent_bug_agent.runtime.anomaly import InvariantEvaluator

    ev = InvariantEvaluator()
    hist = [2.0 - 1.8 * min(1, i / 30) for i in range(120)]  # fell from 2.0 to 0.2, then flat
    rules = []
    for i in range(1, 121):
        rules += [s.rule for s in ev.evaluate(_state(i, hist[i - 1], history=hist[:i]))]
    assert "invariant:loss_trend" not in rules

    ev = InvariantEvaluator()
    flat = [1.1] * 120  # never learned anything
    rules = []
    for i in range(1, 121):
        rules += [s.rule for s in ev.evaluate(_state(i, 1.1, history=flat[:i]))]
    assert "invariant:loss_trend" in rules


# ================================================ integration: synthetic scripts
EXPECTED = {
    "none": set(),
    "eval_mode": {"train_eval_mode_mismatch"},
    "frozen": {"frozen_layer_no_grad"},
    "high_lr": {"exploding_gradients"},
    "dead_relu": {"dead_relu"},
    "partial_dead_relu": {"dead_relu"},  # Module 0 alone misses it (zero_frac < 90%)
    "random_labels": {"learning_rate_misconfigured"},  # loss never improves
    "missing_zero_grad": {"missing_zero_grad"},
    "nan_input": {"exploding_gradients"},
    "exploding": {"exploding_gradients"},
}


def _monitor(bug, **cfg):
    agent = RuntimeMonitorAgent(MonitorConfig(isolate=False, **cfg))
    with contextlib.redirect_stdout(io.StringIO()):
        return agent, agent.monitor_script(PLANTED, ["--bug", bug])


@needs_torch
@pytest.mark.parametrize("bug", sorted(EXPECTED))
def test_planted_bug_scripts(bug):
    agent, report = _monitor(bug)
    found = {f.bug_id for f in report.findings}
    assert EXPECTED[bug] <= found, (bug, found)
    if bug == "none":
        assert found == set(), report.format_text()
    assert all(f.detector == "runtime" and f.file == str(PLANTED) and f.line > 0 for f in report.findings)
    assert agent.last_summary["steps"] == 200 and agent.last_summary["script_status"] == "ok"


@needs_torch
def test_findings_point_at_the_responsible_source_line():
    _, report = _monitor("dead_relu")
    dead = next(f for f in report.findings if f.bug_id == "dead_relu")
    assert dead.evidence == "self.act1 = nn.ReLU()"
    _, report = _monitor("frozen")
    frozen = next(f for f in report.findings if f.bug_id == "frozen_layer_no_grad")
    assert frozen.evidence.startswith("optimizer = torch.optim.SGD(")
    assert frozen.extra["first_step"] == 1  # optimizer coverage check: before the 20-step frozen_weights patience
    _, report = _monitor("eval_mode")
    mode = report.findings[0]
    assert "model(torch.randn(64, 10))" in mode.evidence and mode.extra["phase"] == "eval"
    assert mode.extra["context"]["phase"] == "eval"


@needs_torch
def test_budget_stops_the_script_early():
    agent, report = _monitor("none", max_steps=40)
    assert agent.last_summary["steps"] == 40 and agent.last_summary["stopped_early"]
    assert "step budget" in report.diagnostics[0]


@needs_torch
def test_call_protocol_end_to_end():
    agent = RuntimeMonitorAgent(MonitorConfig(isolate=False))
    state = AgentState(run_id="x", target_file=str(PLANTED), extra={"script_args": ["--bug", "frozen"]})
    with contextlib.redirect_stdout(io.StringIO()):
        agent(state)
    assert "frozen_layer_no_grad" in {f.bug_id for f in state.detection_report.findings}
    assert state.extra["runtime_monitor"]["channels"]["Net"]["optimizer"] == "SGD"


@needs_torch
def test_subprocess_mode_streams_early_alerts():
    alerts = []
    agent = RuntimeMonitorAgent(MonitorConfig(max_steps=120), on_alert=alerts.append)
    report = agent.monitor_script(PLANTED, ["--bug", "missing_zero_grad"])
    assert alerts and alerts[0].bug_id == "missing_zero_grad"  # streamed while the child was running
    assert alerts[0].extra["step"] <= 5
    assert "missing_zero_grad" in {f.bug_id for f in report.findings}
    assert agent.last_summary["stopped_early"] is True


@needs_torch
def test_script_errors_and_scripts_without_training(tmp_path):
    crash = tmp_path / "crash.py"
    crash.write_text(textwrap.dedent("""
        import torch, torch.nn as nn
        m = nn.Linear(4, 1); opt = torch.optim.SGD(m.parameters(), lr=0.1)
        for step in range(10):
            opt.zero_grad(); m(torch.randn(8, 4)).pow(2).mean().backward(); opt.step()
        raise ValueError("boom")
    """))
    r = RuntimeMonitorAgent(MonitorConfig(isolate=False)).monitor_script(crash)
    assert "raised an error after 10 step(s)" in r.diagnostics[0] and "boom" in r.diagnostics[0]
    nothing = tmp_path / "nothing.py"
    nothing.write_text("x = 1 + 1\n")
    r = RuntimeMonitorAgent(MonitorConfig(isolate=False)).monitor_script(nothing)
    assert any("no PyTorch training loop" in d for d in r.diagnostics)


@needs_torch
def test_two_models_two_optimizers(tmp_path):
    gan = tmp_path / "gan.py"
    gan.write_text(textwrap.dedent("""
        import torch, torch.nn as nn
        class Generator(nn.Module):
            def __init__(self):
                super().__init__(); self.a = nn.Linear(4, 8); self.b = nn.Linear(8, 4)
            def forward(self, z): return self.b(torch.relu(self.a(z)))
        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__(); self.a = nn.Linear(4, 8); self.b = nn.Linear(8, 1)
            def forward(self, x): return self.b(torch.relu(self.a(x)))
        G, D = Generator(), Discriminator()
        opt_g = torch.optim.Adam(G.b.parameters(), lr=1e-3)   # bug: G.a never trained
        opt_d = torch.optim.Adam(D.parameters(), lr=1e-3)
        bce = nn.BCEWithLogitsLoss()
        for step in range(60):
            real, z = torch.randn(16, 4) + 2, torch.randn(16, 4)
            opt_d.zero_grad()
            ld = bce(D(real), torch.ones(16, 1)) + bce(D(G(z).detach()), torch.zeros(16, 1))
            ld.backward(); opt_d.step()
            opt_g.zero_grad()
            lg = bce(D(G(z)), torch.ones(16, 1))
            lg.backward(); opt_g.step()
    """))
    agent = RuntimeMonitorAgent(MonitorConfig(isolate=False))
    r = agent.monitor_script(gan)
    assert set(agent.last_summary["channels"]) == {"Discriminator", "Generator"}
    frozen = [f for f in r.findings if f.bug_id == "frozen_layer_no_grad"]
    assert len(frozen) == 1 and frozen[0].extra["channel"] == "Generator"
    assert frozen[0].message.startswith("(Generator)") and "a.weight" in frozen[0].message


# ====================================================== integration: live loops
def _tiny(seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


@needs_torch
def test_watch_live_loop_alerts_while_training():
    import torch
    import torch.nn as nn

    model = _tiny()
    opt = torch.optim.SGD(model[2].parameters(), lr=0.1)  # bug: first layer never updates
    seen_at = []
    agent = RuntimeMonitorAgent(on_alert=lambda f: seen_at.append((f.bug_id, current["step"])))
    current = {"step": 0}
    with agent.watch(model, opt, overhead_budget=None) as session:
        for step in range(1, 81):
            current["step"] = step
            opt.zero_grad()
            nn.functional.cross_entropy(model(torch.randn(32, 16)), torch.randint(0, 4, (32,))).backward()
            opt.step()
    assert seen_at and seen_at[0][0] == "frozen_layer_no_grad" and seen_at[0][1] < 80  # early, not at the end
    report = session.report()
    f = next(f for f in report.findings if f.bug_id == "frozen_layer_no_grad")
    assert f.file.endswith("test_runtime_monitor_agent.py") and f.line > 0  # user code of the live loop


@needs_torch
def test_stop_on_critical_ends_the_run():
    import torch
    import torch.nn as nn

    model = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    agent = RuntimeMonitorAgent(MonitorConfig(stop_on_critical=True))
    steps = 0
    with agent.watch(model, opt, overhead_budget=None) as session:
        for steps in range(1, 201):
            opt.zero_grad()
            x = torch.randn(32, 16)
            if steps >= 10:
                x[0, 0] = float("nan")
            nn.functional.cross_entropy(model(x), torch.randint(0, 4, (32,))).backward()
            opt.step()
    assert steps < 200 and session.stop_reason and "exploding_gradients" in session.stop_reason


@needs_torch
def test_governor_samples_tiny_models_and_keeps_every_step_when_cheap():
    import torch
    import torch.nn as nn

    def run(budget):
        model = _tiny()
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        with RuntimeMonitorAgent().watch(model, opt, overhead_budget=budget) as s:
            for _ in range(60):
                opt.zero_grad()
                nn.functional.cross_entropy(model(torch.randn(32, 16)), torch.randint(0, 4, (32,))).backward()
                opt.step()
        return next(iter(s.channels.values())).governor.stats

    tight = run(0.05)  # a tiny model costs far more than 5% to monitor every step
    assert tight.probes >= 1 and tight.every_n_steps > 1
    # the budget is met, or monitoring is already at its sparsest (one step in max_every_n = 100)
    assert tight.estimated_overhead is not None
    assert tight.estimated_overhead <= 0.05 + 1e-9 or tight.every_n_steps == 100
    loose = run(100.0)  # any cost fits: every step stays monitored
    assert loose.every_n_steps == 1


@needs_torch
def test_detach_leaves_nothing_behind():
    import torch
    import torch.nn as nn
    from torch.nn.modules import module as module_mod

    orig_backward = torch.Tensor.backward
    orig_zero = (torch.optim.Optimizer.zero_grad, nn.Module.zero_grad)
    orig_init = torch.optim.Optimizer.__init__
    model = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(3):  # repeated attach/detach cycles
        with RuntimeMonitorAgent().watch(model, opt) as s:
            for _ in range(5):
                opt.zero_grad()
                model(torch.randn(8, 16)).sum().backward()
                opt.step()
        ch = next(iter(s.channels.values()))
        assert all(b is None for b in ch.hooks.param_coll._shadow)  # device buffers released
    assert torch.Tensor.backward is orig_backward and torch.optim.Optimizer.__init__ is orig_init
    assert (torch.optim.Optimizer.zero_grad, nn.Module.zero_grad) == orig_zero
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    assert not opt._optimizer_step_pre_hooks and not opt._optimizer_step_post_hooks
    assert not module_mod._global_forward_hooks and not module_mod._global_forward_pre_hooks


@needs_torch
def test_overlapping_monitors_restore_patches_in_any_order():
    """Two models monitored at once (a GAN), detached in install order, not reverse: nothing stays patched."""
    import torch
    import torch.nn as nn
    from silent_bug_agent.runtime import RuntimeHooks

    orig = torch.Tensor.backward
    a, b = _tiny(0), _tiny(1)
    oa, ob = torch.optim.SGD(a.parameters(), lr=0.1), torch.optim.SGD(b.parameters(), lr=0.1)
    ha, hb = RuntimeHooks(a, oa).attach(), RuntimeHooks(b, ob).attach()
    for m, o in ((a, oa), (b, ob)):
        o.zero_grad()
        m(torch.randn(4, 16)).sum().backward()
        o.step()
    ha.detach()
    hb.detach()
    assert torch.Tensor.backward is orig
    a(torch.randn(4, 16)).sum().backward()  # a later backward() reaches no stale collector
    assert ha.losses.values == [] and hb.losses.values == []


@needs_torch
def test_no_device_memory_leak_on_gpu():
    import torch
    import torch.nn as nn
    from silent_bug_agent.runtime.profiling import MemoryLeakDetector

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
    if dev is None:
        pytest.skip("no GPU (CUDA or MPS)")
    model = _tiny().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(torch.randn(32, 16, device=dev)), torch.randint(0, 4, (32,), device=dev)).backward()
        opt.step()

    for _ in range(3):
        step()
    det = MemoryLeakDetector(dev, warmup=3)
    det.mark_before_attach()
    with RuntimeMonitorAgent().watch(model, opt, overhead_budget=None):
        for _ in range(40):
            step()
            det.sample()
    det.mark_after_detach()
    rep = det.report()
    assert rep.device_growth_bytes == 0 and rep.detach_residual_bytes == 0 and not rep.leak, rep


@needs_torch
def test_host_memory_stays_bounded():
    import tracemalloc

    import torch
    import torch.nn as nn

    model = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    cfg = MonitorConfig(hook_overrides={"max_loss_history": 50, "stream_capacity": 16})
    pkg = str(REPO / "silent_bug_agent")

    def ours(snap):
        return sum(s.size for s in snap.statistics("filename") if s.traceback[0].filename.startswith(pkg))

    tracemalloc.start()
    try:
        with RuntimeMonitorAgent(cfg).watch(model, opt, overhead_budget=None):
            sizes = []
            for step in range(1, 401):
                opt.zero_grad()
                nn.functional.cross_entropy(model(torch.randn(16, 16)), torch.randint(0, 4, (16,))).backward()
                opt.step()
                if step in (200, 400):
                    sizes.append(ours(tracemalloc.take_snapshot()))
    finally:
        tracemalloc.stop()
    assert sizes[1] - sizes[0] < 256 * 1024, sizes  # no per-step growth once the bounded buffers are full


@needs_torch
def test_profiler_shows_what_monitoring_adds():
    import torch
    import torch.nn as nn
    from silent_bug_agent.runtime import RuntimeHooks
    from silent_bug_agent.runtime.profiling import profile_diff, profile_steps

    model = _tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)

    def step():
        opt.zero_grad()
        model(torch.randn(8, 16)).sum().backward()
        opt.step()

    base = profile_steps(step, 2)
    with RuntimeHooks(model, opt):
        step()
        mon = profile_steps(step, 2)
    added = profile_diff(base, mon, top=20)
    assert added and any(r["added_calls"] > 0 for r in added)


@needs_torch
def test_benchmark_runner_quick_smoke(tmp_path):
    sys.path.insert(0, str(REPO / "examples"))
    try:
        import benchmark_hooks
    finally:
        sys.path.pop(0)
    out = tmp_path / "bench.json"
    with contextlib.redirect_stdout(io.StringIO()):
        code = benchmark_hooks.main(["--quick", "--device", "cpu", "--steps", "3", "--reps", "1",
                                     "--workloads", "mlp", "--no-profile", "--json", str(out)])
    import json

    data = json.loads(out.read_text())
    assert code == 0 and data["results"][0]["workload"] == "mlp"
    assert set(data["results"][0]["overhead"]) == {"hooks_every_step", "agent_every_step", "agent_governed"}
