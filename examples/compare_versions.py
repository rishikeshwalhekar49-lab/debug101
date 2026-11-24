"""Compare two versions of the debugger on the same scripts, offline.

    git worktree add ../old 0619721
    python examples/compare_versions.py --old ../old --new . --json versions.json

Each version's own `fix --dry-run --no-llm --json` runs on every script, as a separate
process with that version's package first on the path and a fresh memory folder, so
neither version learns from the scripts before. For each script the two versions run
one after the other, so both see the same load on the machine.

Scripts: blind copies of the 18 project benchmarks and their healthy twins, and the
test split of the hand-written scripts in examples/holdout/ (never used for tuning).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silent_bug_agent.mutation.blind import blind_copy, expected_bug  # noqa: E402
from silent_bug_agent.mutation.dataset import holdout_jobs  # noqa: E402


def scripts() -> List[Dict[str, str]]:
    out = []
    paths = sorted((REPO_ROOT / "examples" / "sample_scripts").glob("benchmark_[0-9]*.py")) \
        + sorted((REPO_ROOT / "examples" / "new").glob("benchmark_[0-9]*.py"))
    for path in paths:
        source = path.read_text(encoding="utf-8")
        buggy, healthy = blind_copy(source), blind_copy(source, fixed=True)
        out.append({"name": path.stem, "label": expected_bug(source) or "", "code": buggy, "set": "benchmarks"})
        if healthy != buggy:
            out.append({"name": path.stem + " (healthy)", "label": "clean", "code": healthy, "set": "benchmarks"})
    for job in holdout_jobs():
        if job["split"] == "holdout_test":
            out.append({"name": f"{job['group']} ({job['label']})", "label": job["label"], "code": job["code"],
                        "set": "hand-written test"})
    return out


def run_version(repo: Path, code: str, timeout: float) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sba_versions_") as folder:
        script = Path(folder) / "train.py"
        script.write_text(code, encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(repo), SBA_HOME=str(Path(folder) / "memory"))
        env.pop("GROQ_API_KEY", None)
        env.pop("CEREBRAS_API_KEY", None)
        started = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-m", "silent_bug_agent", "fix", str(script), "--dry-run",
                                   "--no-llm", "--json", "--seeds", "5"], cwd=folder, env=env, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"error": "timeout", "seconds": timeout}
        seconds = time.perf_counter() - started
    try:
        runs = json.loads(proc.stdout[proc.stdout.index("["):])
    except ValueError:
        return {"error": (proc.stderr or proc.stdout)[-300:], "seconds": seconds}
    last = runs[-1]
    accepted = next((a for a in last.get("attempts", []) if a.get("accepted")), None)
    return {"seconds": round(seconds, 2), "outcome": last.get("outcome"),
            "top_cause": last["hypotheses"][0]["bug_id"] if last.get("hypotheses") else None,
            "patched": bool(last.get("accepted_diff")), "fixed_bug": accepted["bug_id"] if accepted else None}


def summarise(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    buggy = [r for r in rows if r["label"] != "clean"]
    healthy = [r for r in rows if r["label"] == "clean"]
    times = sorted(r[key]["seconds"] for r in rows)
    return {"buggy": len(buggy), "healthy": len(healthy),
            "repaired": sum(r[key].get("patched", False) for r in buggy),
            "right_cause_first": sum(r[key].get("top_cause") == r["label"] for r in buggy),
            "planted_bug_fixed": sum(r[key].get("fixed_bug") == r["label"] for r in buggy),
            "healthy_changed": sum(r[key].get("patched", False) for r in healthy),
            "errors": sum("error" in r[key] for r in rows),
            "total_seconds": round(sum(times), 1),
            "median_seconds": times[len(times) // 2] if times else None,
            "median_seconds_buggy": sorted(r[key]["seconds"] for r in buggy)[len(buggy) // 2] if buggy else None}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, help="checkout of the older version")
    parser.add_argument("--new", default=str(REPO_ROOT), help="checkout of the newer version")
    parser.add_argument("--workers", type=int, default=2, help="scripts at once (each runs both versions in turn)")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--json", default=None)
    options = parser.parse_args(argv)
    old, new = Path(options.old).resolve(), Path(options.new).resolve()

    def both(item: Dict[str, str]) -> Dict[str, Any]:
        row = dict(item, old=run_version(old, item["code"], options.timeout),
                   new=run_version(new, item["code"], options.timeout))
        row.pop("code")
        print(f"{row['name'][:60]:60} old {row['old'].get('seconds')}s {row['old'].get('outcome')} | "
              f"new {row['new'].get('seconds')}s {row['new'].get('outcome')}", flush=True)
        return row

    with ThreadPoolExecutor(options.workers) as pool:
        rows = list(pool.map(both, scripts()))
    result = {"old": str(old), "new": str(new), "rows": rows, "summary": {}}
    for subset in ("benchmarks", "hand-written test", "all"):
        chosen = rows if subset == "all" else [r for r in rows if r["set"] == subset]
        result["summary"][subset] = {"old": summarise(chosen, "old"), "new": summarise(chosen, "new")}
        print(subset, json.dumps(result["summary"][subset]))
    if options.json:
        Path(options.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
