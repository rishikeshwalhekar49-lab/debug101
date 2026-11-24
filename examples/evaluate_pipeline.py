"""End-to-end evaluation: repair rate on buggy scripts and damage rate on healthy ones.

Each benchmark is run twice through the full pipeline on a private copy:

* buggy: the pipeline should find the planted bug and produce a validated patch;
* healthy (the same script as if run with `--fix`): it should leave the code alone.

By default the pipeline sees a blind copy, saved as `train.py`: no docstring, no
comments, no `--fix` switch and no benchmark name, so nothing in the file gives
the bug away (see `silent_bug_agent/mutation/blind.py`). `--raw` runs the
original files instead, as earlier versions of this script did.

    python examples/evaluate_pipeline.py --offline
    python examples/evaluate_pipeline.py --offline --new       # also the 10 newer benchmarks in examples/new
    python examples/evaluate_pipeline.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silent_bug_agent.agents.supervisor import PipelineConfig, SupervisorAgent, describe_state  # noqa: E402
from silent_bug_agent.learning.experience_store import ExperienceStore  # noqa: E402
from silent_bug_agent.mutation.blind import blind_copy, expected_bug  # noqa: E402

SAMPLES = REPO_ROOT / "examples" / "sample_scripts"
NEW = REPO_ROOT / "examples" / "new"

BENCHMARKS = {
    "benchmark_01_missing_zero_grad.py": "missing_zero_grad",
    "benchmark_02_train_eval_mode.py": "train_eval_mode_mismatch",
    "benchmark_03_frozen_layer.py": "frozen_layer_no_grad",
    "benchmark_04_softmax_before_crossentropy.py": "softmax_before_cross_entropy",
    "benchmark_05_learning_rate_too_high.py": "learning_rate_misconfigured",
    "benchmark_06_dead_relu.py": "dead_relu",
    "benchmark_07_wrong_loss_reduction.py": "wrong_loss_reduction",
    "benchmark_08_unnormalized_inputs.py": "non_normalized_inputs",
}


def evaluate(source: Path, expected: str, healthy: bool, offline: bool, seeds: int, raw: bool,
             learned: str = "advisory") -> Dict[str, Any]:
    started = time.perf_counter()
    name = source.name
    with tempfile.TemporaryDirectory(prefix="sba_eval_", ignore_cleanup_errors=True) as folder:
        if raw:
            script, args = Path(folder) / name, ["--fix"] if healthy else []
            shutil.copy(source, script)
        else:
            script, args = Path(folder) / "train.py", []
            script.write_text(blind_copy(source.read_text(encoding="utf-8"), fixed=healthy), encoding="utf-8")
        config = PipelineConfig(script_args=args, offline=offline, dry_run=True,
                                seeds=(42, 1337, 2024, 7, 99)[:seeds], use_vectors=False, learned=learned)
        store = ExperienceStore(Path(folder) / "memory", use_vectors=False)
        info = describe_state(SupervisorAgent(config, memory=store).run(str(script)))
    return {"script": name, "healthy": healthy, "expected": expected, "status": info["status"], "outcome": info["outcome"],
            "top_cause": info["hypotheses"][0]["bug_id"] if info["hypotheses"] else None,
            "fixed_bug": next((a["bug_id"] for a in info["attempts"] if a["accepted"]), None),
            "patched": info["accepted_diff"] is not None, "route": info["attempts"][-1]["route"] if info["attempts"] else None,
            "p_value": info["attempts"][-1]["p_value"] if info["attempts"] else None, "attempts": len(info["attempts"]),
            "reason": info["reason"], "seconds": round(time.perf_counter() - started, 1), "llm": info["llm"],
            "timings": info.get("timings", {})}


def benchmarks(include_new: bool) -> List[Tuple[Path, str]]:
    jobs = [(SAMPLES / name, bug) for name, bug in BENCHMARKS.items()]
    if include_new:
        for path in sorted(NEW.glob("benchmark_*.py")):
            bug = expected_bug(path.read_text(encoding="utf-8"))
            if bug:
                jobs.append((path, bug))
    return jobs


def no_healthy_twin(path: Path) -> bool:
    """A benchmark whose `--fix` changes nothing has no healthy version to test."""
    source = path.read_text(encoding="utf-8")
    return blind_copy(source) == blind_copy(source, fixed=True)


def time_summary(rows: List[Dict[str, Any]]) -> str:
    totals: Dict[str, float] = {}
    for row in rows:
        for stage, seconds in row["timings"].items():
            totals[stage] = totals.get(stage, 0.0) + seconds
    whole = totals.pop("total", 0.0)
    if not whole:
        return ""
    stages = sorted(((k, v) for k, v in totals.items() if "." not in k and k != "llm"), key=lambda kv: -kv[1])
    parts = ", ".join(f"{k} {100 * v / whole:.0f}%" for k, v in stages if v >= 0.005 * whole)
    return f"Where the time went ({whole:.0f}s over {len(rows)} runs): {parts}; LLM calls {totals.get('llm', 0.0):.0f}s"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Rules and statistics only")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2, help="Scripts to process at once")
    parser.add_argument("--json", default=None, help="Save the results to this file")
    parser.add_argument("--new", action="store_true", help="Also run the 10 newer benchmarks in examples/new")
    parser.add_argument("--learned", choices=["off", "advisory", "active"], default="advisory",
                        help="Trace classifier mode (active lets it add findings and re-rank causes)")
    parser.add_argument("--raw", action="store_true",
                        help="Run the original files, hints and --fix switch included, instead of blind copies")
    options = parser.parse_args()

    jobs = [(path, bug, healthy) for path, bug in benchmarks(options.new) for healthy in (False, True)
            if not (healthy and no_healthy_twin(path))]
    with ThreadPoolExecutor(options.workers) as pool:
        rows: List[Dict[str, Any]] = list(pool.map(
            lambda job: evaluate(job[0], job[1], job[2], options.offline, options.seeds, options.raw,
                                 options.learned), jobs))

    for row in rows:
        kind = "healthy" if row["healthy"] else "buggy  "
        p_value = f"{row['p_value']:.2g}" if isinstance(row["p_value"], float) else "-"
        print(f"{kind} {row['script'][:42]:42} {row['outcome']:13} patched={str(row['patched']):5} "
              f"top={row['top_cause']} fixed={row['fixed_bug']} route={row['route']} p={p_value} {row['seconds']}s")
        if row["reason"]:
            print(f"        reason: {str(row['reason'])[:160]}")
    buggy = [r for r in rows if not r["healthy"]]
    healthy = [r for r in rows if r["healthy"]]
    print(f"\nRepaired: {sum(r['patched'] for r in buggy)} of {len(buggy)} buggy scripts")
    print(f"Correct cause ranked first: {sum(r['top_cause'] == r['expected'] for r in buggy)} of {len(buggy)}")
    print(f"Accepted patch fixed the planted bug itself: {sum(r['fixed_bug'] == r['expected'] for r in buggy)} of {len(buggy)}")
    print(f"Healthy scripts changed (damage): {sum(r['patched'] for r in healthy)} of {len(healthy)}")
    skipped = [path.name for path, _ in benchmarks(options.new) if no_healthy_twin(path)]
    if skipped:
        print(f"No healthy twin (their --fix changes nothing): {', '.join(skipped)}")
    print(("Raw files" if options.raw else "Blind copies (no docstrings, comments, --fix switch or names)")
          + " were evaluated.")
    summary = time_summary(rows)
    if summary:
        print(summary)
    if options.json:
        Path(options.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
