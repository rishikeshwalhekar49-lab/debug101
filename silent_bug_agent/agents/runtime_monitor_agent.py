"""Runtime Monitor Agent: watches training live and reports silent bugs as Findings.

Supervisor protocol (Person 1 integration contract)::

    from silent_bug_agent.agents.runtime_monitor_agent import RuntimeMonitorAgent
    supervisor = SupervisorAgent(detection_agent=RuntimeMonitorAgent())
    state = RuntimeMonitorAgent()(state)          # fills state.detection_report

What the agent does with a training script (``monitor_script``):

1. runs it unmodified, in a subprocess by default, for a bounded budget
   (``max_steps`` optimizer steps / ``max_seconds``);
2. finds the model and optimizer automatically and attaches the Module 1
   runtime hooks (``silent_bug_agent/runtime/``) -- the telemetry events;
3. continuously checks every snapshot with the Module 0 invariant suite, the
   runtime checks ported from package101 and a rolling-window trend analyzer;
4. aggregates anomalies (dead units, exploding/vanishing gradients, frozen
   parameters, missing zero_grad, train/eval mode, divergence, ...) into one
   contextual :class:`~silent_bug_agent.findings.Finding` per bug, each pointing
   at a file and line of the script, and calls ``on_alert`` the moment one is
   confirmed (early alerts, streamed out of the subprocess while it runs);
5. returns a :class:`~silent_bug_agent.findings.Report`.

``watch(model, optimizer)`` monitors a live run in-process instead, keeping the
monitoring overhead under ``overhead_budget`` (5% by default).

Command line (the ``monitor`` CLI command lazy-imports this class)::

    python -m silent_bug_agent.agents.runtime_monitor_agent train.py [--max-steps 300] [--json] [-- script args]

This module imports torch lazily, so the Supervisor and CLI can import it
without torch installed (``monitor_script`` then returns a diagnostic).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..findings import SEVERITY_RANK, Finding, Report, merge_reports, parse_severity
from ..invariants.base import Severity
from ..schemas import AgentState

ALERT_MARK = "@@silent_bug_agent:alert "  # early-alert lines the monitored subprocess writes to its stdout
_PKG_PARENT = str(Path(__file__).resolve().parents[2])


@dataclass
class MonitorConfig:
    # budget of a monitored script run
    max_steps: Optional[int] = 300  # optimizer steps; None = run to completion
    max_seconds: Optional[float] = 900.0
    isolate: bool = True  # run the script in a subprocess (crash/timeout isolation, clean GPU memory)
    cwd: Optional[str] = None  # working directory of the script (default: the current one, like `python script.py`)
    timeout_grace: float = 120.0  # extra seconds before a subprocess that ignores the budget is killed
    echo_output: bool = False  # forward the script's own output
    # overhead control
    overhead_budget: Optional[float] = 0.05  # live sessions (watch): amortised monitoring cost cap; None = every step
    script_overhead_budget: Optional[float] = None  # monitor_script: diagnostic runs are short, monitor every step
    # alerting
    min_severity: str = "warning"  # findings below this are dropped
    confirm_after: int = 2  # occurrences needed before a noisy per-step check becomes a finding
    stop_on_critical: bool = False  # stop the run (StopTraining) once a critical alert is confirmed
    grad_lifecycle: bool = True  # watch zero_grad() / gradient tensors for a missing zero_grad()
    hook_overrides: Dict[str, Any] = field(default_factory=dict)  # HookConfig fields, e.g. {"histogram_bins": 16}

    def session_options(self, budget: Optional[float]) -> Any:
        from ..runtime.session import SessionOptions

        return SessionOptions(overhead_budget=budget, min_severity=parse_severity(self.min_severity),
                              confirm_after=self.confirm_after, stop_on_critical=self.stop_on_critical,
                              grad_lifecycle=self.grad_lifecycle, hook_overrides=dict(self.hook_overrides))


def _torch_missing() -> Optional[str]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return "runtime monitor skipped: PyTorch is not installed (pip install torch)"
    return None


class RuntimeMonitorAgent:
    """Detection agent that observes training at runtime (Person 2, Module 1)."""

    name = "runtime_monitor"

    def __init__(self, config: Optional[MonitorConfig] = None, *,
                 on_alert: Optional[Callable[[Finding], None]] = None) -> None:
        self.config = config or MonitorConfig()
        self.on_alert = on_alert
        self.last_summary: Dict[str, Any] = {}

    # ------------------------------------------------------------ contract
    def monitor_script(self, path: Path, args: Optional[Sequence[str]] = None) -> Report:
        """Attach hooks to training script, return Report of anomaly Findings."""
        path = Path(path)
        shown = str(path)
        missing = _torch_missing()
        if missing:
            return Report(files_scanned=[shown], diagnostics=[missing])
        if not path.is_file():
            return Report(diagnostics=[f"runtime monitor: script not found: {shown}"])
        args = list(args or [])
        if self.config.isolate:
            report, summary = self._monitor_subprocess(path, args)
        else:
            report, summary = self._monitor_in_process(path, args)
        self.last_summary = summary
        return report

    def __call__(self, state: AgentState) -> AgentState:
        """Supervisor protocol: populate ``state.detection_report``."""
        if not state.target_file:
            report = Report(diagnostics=["runtime monitor: AgentState.target_file is empty"])
        else:
            report = self.monitor_script(Path(state.target_file), state.extra.get("script_args"))
        # keep findings another detector (static/semantic) already put there; runtime duplicates are dropped
        state.detection_report = report if state.detection_report is None else merge_reports(state.detection_report, report)
        state.extra["runtime_monitor"] = self.last_summary
        return state

    # ---------------------------------------------------------- live runs
    def watch(self, model: Any, optimizer: Any = None, *, overhead_budget: Any = "config",
              file: Optional[str] = None) -> Any:
        """Monitor a live in-process run; use as a context manager::

            with agent.watch(model, optimizer) as session:
                train()                     # unchanged loop
            report = session.report()

        ``session.alerts`` lists confirmed alerts while training runs."""
        missing = _torch_missing()
        if missing:
            raise RuntimeError(missing)
        from ..runtime.instrument import user_frame
        from ..runtime.session import MonitorSession

        budget = self.config.overhead_budget if overhead_budget == "config" else overhead_budget
        session = MonitorSession(self.config.session_options(budget), file=file, on_alert=self._dispatch)
        here = user_frame()
        session.add(model, optimizer, optimizer_location=None, fallback=here)
        return session.start()

    def _dispatch(self, finding: Finding, kind: str = "new") -> None:
        if self.on_alert is not None:
            self.on_alert(finding)

    # ---------------------------------------------------- script execution
    def _monitor_in_process(self, path: Path, args: List[str],
                            emit: Optional[Callable[[Finding, str], None]] = None) -> Tuple[Report, Dict[str, Any]]:
        from ..runtime.instrument import AutoInstrumenter, exec_script
        from ..runtime.session import MonitorSession

        cfg = self.config
        session = MonitorSession(cfg.session_options(cfg.script_overhead_budget), file=str(path),
                                 on_alert=emit or self._dispatch)
        inst = AutoInstrumenter(lambda m, o, loc: session.add(m, o, optimizer_location=loc),
                                max_steps=cfg.max_steps, max_seconds=cfg.max_seconds)
        session.optimizer_registry = lambda: list(inst.optimizers.values())
        session.start()
        inst.install()
        try:
            result = exec_script(str(path), args, cfg.cwd)
        finally:
            inst.uninstall()
            session.close()
        report = session.report()
        steps = inst.status.steps
        if result.status == "stopped" or inst.status.stopped_early:
            report.diagnostics.insert(0, f"runtime: stopped after {steps} step(s): "
                                         f"{inst.status.stop_reason or session.stop_reason or 'stop requested'}")
        elif result.status == "error":
            report.diagnostics.insert(0, f"runtime: the script raised an error after {steps} step(s) "
                                         f"(not a silent bug): {result.error}")
        else:
            report.diagnostics.insert(0, f"runtime: the script finished after {steps} step(s) in {result.duration:.1f}s")
        if not session.channels:
            report.diagnostics.append("runtime: no PyTorch training loop was observed (no model was called whose "
                                      "parameters an optimizer updates), so nothing was monitored")
        summary = session.summary()
        summary.update({"script_status": result.status, "script_error": result.error, "steps": steps,
                        "stopped_early": inst.status.stopped_early, "stop_reason": inst.status.stop_reason or None,
                        "duration_s": round(result.duration, 3)})
        return report, summary

    def _monitor_subprocess(self, path: Path, args: List[str]) -> Tuple[Report, Dict[str, Any]]:
        cfg = self.config
        streamed: List[Finding] = []
        tail: deque = deque(maxlen=40)
        with tempfile.TemporaryDirectory(prefix="sba_monitor_") as tmp:
            cfg_path, result_path = os.path.join(tmp, "config.json"), os.path.join(tmp, "result.json")
            child_cfg = asdict(cfg)
            child_cfg["isolate"] = False
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"script": str(path.resolve()), "args": args, "config": child_cfg,
                           "result_path": result_path}, f)
            env = dict(os.environ)
            env["PYTHONPATH"] = _PKG_PARENT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen([sys.executable, "-m", "silent_bug_agent.agents.runtime_monitor_agent",
                                     "--child", cfg_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env, cwd=cfg.cwd or os.getcwd())

            def pump() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if line.startswith(ALERT_MARK):
                        try:
                            finding = Finding.from_dict(json.loads(line[len(ALERT_MARK):]))
                        except Exception:
                            continue
                        finding.file = str(path) if finding.file == str(path.resolve()) else finding.file
                        streamed.append(finding)
                        self._dispatch(finding)
                        continue
                    tail.append(line.rstrip("\n"))
                    if cfg.echo_output:
                        print(line, end="")

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            timeout = None if cfg.max_seconds is None else cfg.max_seconds + cfg.timeout_grace
            killed = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                killed = True
            reader.join(timeout=10)
            payload = None
            if os.path.exists(result_path):
                with open(result_path, encoding="utf-8") as f:
                    payload = json.load(f)
        if payload is not None:
            report = Report.from_json(json.dumps(payload["report"]))
            report.files_scanned = [str(path)]
            for f in report.findings:
                if f.file == str(path.resolve()):
                    f.file = str(path)
            return report, payload.get("summary", {})
        # the child died before writing its report: keep the alerts it streamed
        reason = f"killed after {timeout:g}s (budget exceeded)" if killed else f"exited with code {proc.returncode}"
        diag = [f"runtime: the monitored process {reason} before it could report; "
                f"{len(streamed)} early alert(s) were received"]
        if tail:
            diag.append("runtime: last output: " + " | ".join(list(tail)[-5:]))
        return Report(findings=streamed, files_scanned=[str(path)], diagnostics=diag), {"killed": killed,
                                                                                       "returncode": proc.returncode}


# ------------------------------------------------------------------ entry
def _child_main(cfg_path: str) -> int:
    """Subprocess side of monitor_script: run the script in-process and write the report."""
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    out = sys.__stdout__
    lock = threading.Lock()

    def emit(finding: Finding, kind: str) -> None:
        with lock:
            out.write(ALERT_MARK + json.dumps(finding.to_dict(), default=str) + "\n")
            out.flush()

    agent = RuntimeMonitorAgent(MonitorConfig(**cfg["config"]))
    report, summary = agent._monitor_in_process(Path(cfg["script"]), list(cfg.get("args", [])), emit=emit)
    with open(cfg["result_path"], "w", encoding="utf-8") as f:
        json.dump({"report": report.to_dict(), "summary": summary}, f, default=str)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--child":
        return _child_main(argv[1])
    script_args: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, script_args = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(prog="python -m silent_bug_agent.agents.runtime_monitor_agent",
                                description="Run a training script under the Runtime Monitor Agent.")
    p.add_argument("script")
    p.add_argument("--max-steps", type=int, default=300, help="optimizer-step budget (0 = no limit)")
    p.add_argument("--max-seconds", type=float, default=900.0, help="wall-clock budget (0 = no limit)")
    p.add_argument("--in-process", action="store_true", help="run the script in this process")
    p.add_argument("--stop-on-critical", action="store_true")
    p.add_argument("--echo", action="store_true", help="show the script's own output")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    path = Path(a.script)
    if not path.is_file():
        print(f"Error: script '{a.script}' does not exist.", file=sys.stderr)
        return 2

    def live(f: Finding) -> None:
        if not a.json:
            print(f"[early alert] {f.severity.value} {f.bug_id} at {f.file}:{f.line} (step {f.extra.get('step')})",
                  file=sys.stderr, flush=True)

    agent = RuntimeMonitorAgent(MonitorConfig(max_steps=a.max_steps or None, max_seconds=a.max_seconds or None,
                                              isolate=not a.in_process, stop_on_critical=a.stop_on_critical,
                                              echo_output=a.echo), on_alert=live)
    report = agent.monitor_script(path, script_args)
    print(report.to_json() if a.json else report.format_text())
    worst = max((SEVERITY_RANK[f.severity] for f in report.findings), default=-1)
    return 1 if worst >= SEVERITY_RANK[Severity.WARNING] else 0


if __name__ == "__main__":
    sys.exit(main())
