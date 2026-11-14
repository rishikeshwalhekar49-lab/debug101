"""Command line for silent_bug_agent.

    python -m silent_bug_agent scan <path> [--semantic] [--json] [--events]
    python -m silent_bug_agent monitor <script> [--max-steps N] [--json] [--events] [-- script args]
    python -m silent_bug_agent diagnose <script> [--no-llm] [--json] [--events] [-- script args]
    python -m silent_bug_agent fix <script> [--max-iter N] [--dry-run] [--no-llm] [--json] [--events] [-- script args]
    python -m silent_bug_agent doctor [--live] [--events]
    python -m silent_bug_agent ui

Heavy libraries are imported inside each command, so `scan` works without PyTorch.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .events import EventEmitter
from .findings import Report, merge_reports
from .invariants.base import Severity
from .semantic.analyzer import SemanticAnalyzer
from .semantic.router import ModelRouter
from .static import StaticAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="silent-bug-agent",
        description="Detect, diagnose and fix silent bugs in deep learning training code.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    scan_parser = subparsers.add_parser("scan", help="Scan a file or directory for silent bugs")
    scan_parser.add_argument("path", type=str, help="Path to a Python file or directory containing training code")
    scan_parser.add_argument("--semantic", action="store_true", default=False,
                             help="Add the LLM semantic review (needs GROQ_API_KEY or CEREBRAS_API_KEY)")
    scan_parser.add_argument("--json", action="store_true", default=False, help="Output the final report as JSON")
    scan_parser.add_argument("--model", type=str, default=None, help="Model override, as provider/model")
    scan_parser.add_argument("--events", action="store_true", default=False, help="Emit NDJSON events for the Ink TUI")

    monitor_parser = subparsers.add_parser("monitor", help="Run a training script under the runtime hooks")
    monitor_parser.add_argument("script", type=str)
    monitor_parser.add_argument("--max-steps", type=int, default=300, help="Optimizer steps to watch (0 = no limit)")
    monitor_parser.add_argument("--max-seconds", type=float, default=900.0, help="Time limit (0 = no limit)")
    monitor_parser.add_argument("--stop-on-critical", action="store_true")
    monitor_parser.add_argument("--json", action="store_true")
    monitor_parser.add_argument("--events", action="store_true", default=False, help="Emit NDJSON events for the Ink TUI")

    diagnose_parser = subparsers.add_parser("diagnose", help="Detect bugs and rank their root causes")
    diagnose_parser.add_argument("script", type=str)
    diagnose_parser.add_argument("--max-steps", type=int, default=150)
    diagnose_parser.add_argument("--no-llm", action="store_true", help="Use rules and statistics only")
    diagnose_parser.add_argument("--json", action="store_true")
    diagnose_parser.add_argument("--events", action="store_true", default=False, help="Emit NDJSON events for the Ink TUI")

    fix_parser = subparsers.add_parser("fix", help="Run the full supervisor loop and apply a validated patch")
    fix_parser.add_argument("script", type=str)
    fix_parser.add_argument("--max-iter", type=int, default=3, help="Repair attempts per bug round")
    fix_parser.add_argument("--max-rounds", type=int, default=3, help="Bugs to fix in one command")
    fix_parser.add_argument("--dry-run", action="store_true", help="Show the patch without changing the file")
    fix_parser.add_argument("--no-llm", action="store_true", help="Use rules and statistics only")
    fix_parser.add_argument("--seeds", type=int, default=5, help="Seeds for the statistical gate (3 to 5)")
    fix_parser.add_argument("--json", action="store_true")
    fix_parser.add_argument("--events", action="store_true", default=False, help="Emit NDJSON events for the Ink TUI")

    doctor_parser = subparsers.add_parser("doctor", help="Check the installation, keys and models")
    doctor_parser.add_argument("--live", action="store_true", help="Also send one test request to each model")
    doctor_parser.add_argument("--events", action="store_true", default=False, help="Emit NDJSON events for the Ink TUI")

    wsp_parser = subparsers.add_parser("wsp", help="Simultaneous Code-Space and Weight-Space diagnosis & repair")
    wsp_parser.add_argument("script", type=str, help="Path to training script")
    wsp_parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt/.pth/.safetensors)")
    wsp_parser.add_argument("--max-iter", type=int, default=3, help="Max repair & validation iterations")
    wsp_parser.add_argument("--dry-run", action="store_true", help="Show proposed changes without modifying files")
    wsp_parser.add_argument("--yes", "-y", action="store_true", help="Non-interactive auto-approval of solutions")
    wsp_parser.add_argument("--events", action="store_true", default=False, help="Emit NDJSON events for the Ink TUI")

    subparsers.add_parser("ui", help="Launch the Pastel + Ink interactive TUI (requires Node.js)")
    return parser


def _use_utf8_output() -> None:
    """A Windows console can reject characters in a report, so replace them instead of crashing."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def run_scan(path_str: str, semantic: bool = False, json_output: bool = False,
             model: Optional[str] = None, events: bool = False) -> int:
    emitter = EventEmitter(enabled=events)
    target = Path(path_str).resolve()
    if not target.exists():
        emitter.error(f"Target path '{path_str}' does not exist.")
        emitter.done(2)
        print(f"Error: Target path '{path_str}' does not exist.", file=sys.stderr)
        return 2

    emitter.phase("static_analysis", "start")
    static_analyzer = StaticAnalyzer()
    report = static_analyzer.analyze_file(target) if target.is_file() else static_analyzer.analyze_directory(target)
    emitter.phase("static_analysis", "done", findings=len(report.findings))

    if semantic:
        import os

        emitter.phase("semantic_review", "start")
        router = ModelRouter()
        if model:
            os.environ["SBA_MODEL_SEMANTIC_REVIEW"] = model
        client = router.task("semantic_review")
        if client is None:
            report.diagnostics.append("Semantic review skipped: set GROQ_API_KEY or CEREBRAS_API_KEY in .env.")
            emitter.phase("semantic_review", "done", skipped=True)
        else:
            analyzer = SemanticAnalyzer(client=client)
            sem_report = analyzer.analyze_file(target) if target.is_file() else analyzer.analyze_directory(target)
            report = merge_reports(report, sem_report)
            used = router.summary()["models_used"]
            report.diagnostics.append(f"Semantic review used: {', '.join(used) or 'no model answered'}.")
            emitter.phase("semantic_review", "done", findings=len(sem_report.findings))

    emitter.findings_batch(
        [f.to_dict() for f in report.sorted_findings()],
        diagnostics=report.diagnostics,
    )
    emitter.result(report.to_dict())
    has_issues = any(f.severity in (Severity.WARNING, Severity.CRITICAL) for f in report.findings)
    emitter.done(1 if has_issues else 0)
    if not events:
        print(report.to_json(indent=2) if json_output else report.format_text())
    return 1 if has_issues else 0


def _need_file(path_str: str) -> Optional[Path]:
    target = Path(path_str).resolve()
    if not target.is_file():
        print(f"Error: script '{path_str}' does not exist.", file=sys.stderr)
        return None
    return target


def run_monitor(script: str, script_args: List[str], *, max_steps: int = 300, max_seconds: float = 900.0,
                stop_on_critical: bool = False, json_output: bool = False, events: bool = False) -> int:
    emitter = EventEmitter(enabled=events)
    target = _need_file(script)
    if target is None:
        emitter.error(f"Script '{script}' does not exist.")
        emitter.done(2)
        return 2
    from .agents.runtime_monitor_agent import MonitorConfig, RuntimeMonitorAgent

    emitter.phase("runtime_monitor", "start", max_steps=max_steps)

    def early(finding: Any) -> None:
        emitter.runtime_alert(
            severity=finding.severity.value,
            bug_id=finding.bug_id,
            file=finding.file,
            line=finding.line,
            message=finding.message,
        )
        if not json_output and not events:
            print(f"[early alert] {finding.severity.value} {finding.bug_id} at {finding.file}:{finding.line}",
                  file=sys.stderr, flush=True)

    config = MonitorConfig(max_steps=max_steps or None, max_seconds=max_seconds or None,
                           stop_on_critical=stop_on_critical, cwd=str(target.parent))
    report = RuntimeMonitorAgent(config, on_alert=early).monitor_script(target, script_args)
    emitter.phase("runtime_monitor", "done", findings=len(report.findings))
    emitter.findings_batch([f.to_dict() for f in report.sorted_findings()], diagnostics=report.diagnostics)
    emitter.result(report.to_dict())
    has_issues = bool(report.at_least(Severity.WARNING))
    emitter.done(1 if has_issues else 0)
    if not events:
        print(report.to_json() if json_output else report.format_text())
    return 1 if has_issues else 0


def run_diagnose(script: str, script_args: List[str], *, max_steps: int = 150, no_llm: bool = False,
                 json_output: bool = False, events: bool = False) -> int:
    emitter = EventEmitter(enabled=events)
    target = _need_file(script)
    if target is None:
        emitter.error(f"Script '{script}' does not exist.")
        emitter.done(2)
        return 2
    from .agents.diagnosis_agent import DiagnosisAgent
    from .agents.runtime_monitor_agent import MonitorConfig, RuntimeMonitorAgent
    from .schemas import AgentState

    emitter.phase("perception", "start")
    router = ModelRouter(offline=no_llm)
    report = StaticAnalyzer().analyze_file(target)
    client = router.task("semantic_review")
    if client is not None:
        report = merge_reports(report, SemanticAnalyzer(client).analyze_file(target))
    state = AgentState(target_file=str(target), detection_report=report, extra={"script_args": script_args})
    state = RuntimeMonitorAgent(MonitorConfig(max_steps=max_steps, cwd=str(target.parent)))(state)
    emitter.phase("perception", "done", findings=len(state.detection_report.findings))

    emitter.phase("diagnosis", "start")
    state = DiagnosisAgent(client=router.task("diagnosis"))(state)
    emitter.phase("diagnosis", "done", hypotheses=len(state.hypotheses))

    rows = [{"rank": i + 1, "bug_id": h.bug_id, "confidence": h.confidence, "explanation": h.description,
             "diagnosed_by": h.extra.get("diagnosed_by")} for i, h in enumerate(state.hypotheses)]
    for row in rows:
        emitter.hypothesis(
            rank=row["rank"], bug_id=row["bug_id"], confidence=row["confidence"],
            description=row["explanation"], diagnosed_by=row["diagnosed_by"],
        )
    emitter.result({"hypotheses": rows, "llm": router.summary()})
    emitter.done(1 if rows else 0)
    if not events:
        if json_output:
            print(json.dumps({"hypotheses": rows, "llm": router.summary()}, indent=2))
        else:
            print(f"LLM: {router.describe()}")
            for row in rows:
                print(f"{row['rank']}. {row['bug_id']}  (confidence {row['confidence']:.2f}, {row['diagnosed_by']})")
                print(f"   {row['explanation']}")
            if not rows:
                print("No bugs were detected.")
            for note in state.detection_report.diagnostics:
                print(f"note: {note}")
    return 1 if rows else 0


def _print_run(info: Dict[str, Any], dry_run: bool) -> None:
    print(f"\nStatus: {info['status'].upper()}  ({info['outcome']})")
    if info.get("summary"):
        print(f"Perception: {info['summary']}")
    for hypothesis in info["hypotheses"][:5]:
        print(f"  cause: {hypothesis['bug_id']}  (confidence {hypothesis['confidence']:.2f}, {hypothesis['diagnosed_by']})")
    for attempt in info["attempts"]:
        word = "accepted" if attempt["accepted"] else "rejected"
        print(f"  attempt: {attempt['bug_id']} from {attempt['source']}: {word}. {attempt['notes']}")
    if info["accepted_diff"]:
        print("\nValidated patch:\n" + info["accepted_diff"])
        if info["patch_applied"]:
            print(f"Patch applied. Original saved as {info['backup']}")
        elif dry_run:
            print("Dry run: the file was not changed.")
    if info["reason"]:
        print(f"Reason: {info['reason']}")
    for note in info["llm_notes"] + info["memory_notes"]:
        print(f"note: {note}")
    llm = info.get("llm") or {}
    print(f"Models used: {', '.join(llm.get('models_used', [])) or 'none (offline rules and statistics)'}"
          f"  |  LLM calls: {llm.get('calls', 0)}, failed: {llm.get('failed_calls', 0)}, tokens: {llm.get('tokens', 0)}")


def run_fix(script: str, script_args: List[str], *, max_iter: int = 3, max_rounds: int = 3, dry_run: bool = False,
            no_llm: bool = False, seeds: int = 5, json_output: bool = False, events: bool = False) -> int:
    emitter = EventEmitter(enabled=events)
    target = _need_file(script)
    if target is None:
        emitter.error(f"Script '{script}' does not exist.")
        emitter.done(2)
        return 2
    from .agents.supervisor import PipelineConfig, SupervisorAgent, describe_state

    config = PipelineConfig(max_iterations=max_iter, script_args=script_args, dry_run=dry_run, offline=no_llm,
                            seeds=(42, 1337, 2024, 7, 99)[:max(3, min(5, seeds))])
    try:
        supervisor = SupervisorAgent(config, emitter=emitter)
    except Exception as exc:
        emitter.error(f"Failed to initialise supervisor: {exc}")
        emitter.done(2)
        raise
    if not json_output and not events:
        print(f"LLM: {supervisor.router.describe()}")

    emitter.phase("pipeline", "start", max_rounds=max_rounds, max_iter=max_iter, dry_run=dry_run)
    try:
        states = supervisor.run_until_clean(str(target), max_rounds=1 if dry_run else max_rounds)
    except Exception as exc:
        emitter.error(f"Pipeline crashed: {exc}")
        emitter.done(2)
        raise
    runs = [describe_state(state) for state in states]

    for info in runs:
        for hyp in info.get("hypotheses", []):
            emitter.hypothesis(
                rank=info["hypotheses"].index(hyp) + 1,
                bug_id=hyp["bug_id"],
                confidence=hyp["confidence"],
                description=hyp.get("explanation", ""),
                diagnosed_by=hyp.get("diagnosed_by"),
            )
        for attempt in info.get("attempts", []):
            emitter.verdict(
                bug_id=attempt.get("bug_id", ""),
                accepted=attempt.get("accepted", False),
                validator=attempt.get("validator", ""),
                p_value=attempt.get("p_value"),
                effect_size=attempt.get("effect_size"),
                notes=attempt.get("notes", ""),
            )
        if info.get("accepted_diff"):
            emitter.patch(
                bug_id=info.get("hypotheses", [{}])[0].get("bug_id", "") if info.get("hypotheses") else "",
                strategy="direct_edit",
                source="pipeline",
                diff=info["accepted_diff"],
                confidence=1.0,
            )

    emitter.result({"runs": runs})
    failed = runs[-1]["status"] == "failed"
    emitter.done(1 if failed else 0)
    if not events:
        if json_output:
            print(json.dumps(runs, indent=2, default=str))
        else:
            for info in runs:
                _print_run(info, dry_run)
    return 1 if failed else 0


def run_doctor(live: bool = False, events: bool = False) -> int:
    """Checks every requirement and says exactly what to do about each failure."""
    emitter = EventEmitter(enabled=events)
    from .config import find_env_file, load_env
    from .semantic.router import PROVIDERS, TASK_MODELS

    emitter.phase("doctor", "start")
    problems = 0

    def report(ok: bool, label: str, advice: str = "") -> None:
        nonlocal problems
        problems += 0 if ok else 1
        emitter.doctor_check(ok=ok, label=label, advice=advice if not ok else "")
        if not events:
            print(f"[{'ok' if ok else 'PROBLEM'}] {label}" + (f": {advice}" if advice and not ok else ""))

    if not events:
        print(f"Python {sys.version.split()[0]} on {sys.platform}")
    report(sys.version_info >= (3, 10), "Python 3.10 or newer", "install a newer Python")
    for module, install in (("torch", "pip install torch"), ("yaml", "pip install pyyaml"), ("httpx", "pip install httpx"),
                            ("langgraph", "pip install langgraph"), ("scipy", "pip install scipy"),
                            ("pydantic", "pip install pydantic"), ("chromadb", 'pip install -e ".[semantic]"')):
        try:
            __import__(module)
            report(True, f"library {module}")
        except ImportError:
            report(False, f"library {module}", install)
    env_file = find_env_file()
    load_env()
    if not events:
        print(f"[{'ok' if env_file else 'note'}] .env file: {env_file or 'not found, copy .env.example to .env'}")
    router = ModelRouter()
    for name, provider in PROVIDERS.items():
        if not events:
            print(f"[{'ok' if router.key_for(name) else 'note'}] {provider.key_env}: {'set' if router.key_for(name) else 'not set'}")
    report(router.available(), "at least one LLM key", "without a key only rules and statistics run; see .env.example")
    if router.available() and not events:
        for task in TASK_MODELS:
            print(f"      {task}: " + " then ".join(f"{c.provider}/{c.model}" for c in router.choices(task)))
    if live:
        seen = set()
        for task in TASK_MODELS:
            for choice in router.choices(task):
                if (choice.provider, choice.model) in seen:
                    continue
                seen.add((choice.provider, choice.model))
                try:
                    reply = router.client_for(choice).complete_json(
                        'Reply with the JSON object {"status": "ok"} and nothing else.', None,
                        schema={"type": "object", "properties": {"status": {"type": "string"}},
                                "required": ["status"], "additionalProperties": False}, schema_name="ping")
                    report(isinstance(reply, dict) and reply.get("status") == "ok", f"live call {choice.provider}/{choice.model}",
                           f"unexpected reply: {str(reply)[:120]}")
                except Exception as exc:
                    report(False, f"live call {choice.provider}/{choice.model}", str(exc)[:200])

    emitter.phase("doctor", "done", problems=problems)
    emitter.done(0 if problems == 0 else 1)
    if not events:
        print("All checks passed." if problems == 0 else f"{problems} problem(s) found.")
    return 0 if problems == 0 else 1


def run_wsp(script: str, checkpoint: Optional[str] = None, *, max_iter: int = 3, dry_run: bool = False,
            interactive: bool = True, events: bool = False) -> int:
    """Run simultaneous Code-Space and Weight-Space debugging with interactive approval."""
    emitter = EventEmitter(enabled=events)
    target = _need_file(script)
    if target is None:
        emitter.error(f"Script '{script}' does not exist.")
        emitter.done(2)
        return 2

    from .wsp.dual_supervisor import DualConfig, DualSupervisorAgent

    config = DualConfig(max_iterations=max_iter, interactive=interactive, dry_run=dry_run)
    supervisor = DualSupervisorAgent(config, emitter=emitter)
    return supervisor.run(str(target), checkpoint_path=checkpoint)


def _launch_ui() -> int:
    """Launch the Pastel + Ink TUI from the tui/ subdirectory."""
    pkg_root = Path(__file__).resolve().parent.parent
    tui_dir = pkg_root / "tui"

    if not tui_dir.is_dir():
        print("Error: tui/ directory not found. Run `cd tui && npm install` first.", file=sys.stderr)
        return 1

    node = shutil.which("node")
    if node is None:
        print("Error: Node.js not found. Install it from https://nodejs.org", file=sys.stderr)
        return 1

    npm = shutil.which("npm")
    dist = tui_dir / "dist" / "cli.js"
    if not dist.exists():
        if npm:
            print("Building TUI...", file=sys.stderr)
            result = subprocess.run([npm, "run", "build"], cwd=str(tui_dir), capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Build failed:\n{result.stderr}", file=sys.stderr)
                return 1
        else:
            print("Error: TUI not built and npm not found. Run `cd tui && npm install && npm run build`.",
                  file=sys.stderr)
            return 1

    try:
        proc = subprocess.run([node, str(dist)] + sys.argv[2:], cwd=str(pkg_root))
        return proc.returncode
    except KeyboardInterrupt:
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    _use_utf8_output()
    argv = list(sys.argv[1:] if argv is None else argv)
    script_args: List[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, script_args = argv[:split], argv[split + 1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        return run_scan(path_str=args.path, semantic=args.semantic, json_output=args.json,
                        model=args.model, events=args.events)
    if args.command == "monitor":
        return run_monitor(args.script, script_args, max_steps=args.max_steps, max_seconds=args.max_seconds,
                           stop_on_critical=args.stop_on_critical, json_output=args.json, events=args.events)
    if args.command == "diagnose":
        return run_diagnose(args.script, script_args, max_steps=args.max_steps, no_llm=args.no_llm,
                            json_output=args.json, events=args.events)
    if args.command == "fix":
        return run_fix(args.script, script_args, max_iter=args.max_iter, max_rounds=args.max_rounds,
                       dry_run=args.dry_run, no_llm=args.no_llm, seeds=args.seeds,
                       json_output=args.json, events=args.events)
    if args.command == "doctor":
        return run_doctor(live=args.live, events=args.events)
    if args.command == "wsp":
        return run_wsp(args.script, checkpoint=args.checkpoint, max_iter=args.max_iter,
                       dry_run=args.dry_run, interactive=not args.yes, events=args.events)
    if args.command == "ui":
        return _launch_ui()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
