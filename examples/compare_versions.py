"""Compare versions of the debugger on the same scripts, offline.

    git worktree add ../shreyas 0619721
    git worktree add ../v1 0728d71
    git worktree add ../v3 b7179ad
    python examples/compare_versions.py --version shreyas=../shreyas --version v1=../v1 \
        --version v3=../v3 --version v4=. --scripts old colab --json versions.json

Each version's own `fix --dry-run --no-llm --json` runs on every script, as a separate
process with that version's package first on the path and a fresh memory folder, so
no version learns from the scripts before. For each script the versions run one after
the other, so all see the same load on the machine. `name=path` may add arguments for
that version's `fix` after a colon: `v4=.:--learned active`.

Script sets (--scripts):
* old: blind copies of the 18 project benchmarks and their healthy twins, and the test
  split of examples/holdout/. Seen before: v3 was compared with Shreyas's version on them.
* colab: the test split of the Colab-notebook scripts (examples/colab/), on Colab's sample
  data, only the versions whose bug was shown to make training worse (--data, the v4
  dataset, says which). Never used to choose anything.
* dev: the development splits (examples/holdout/ 01 to 04 and the Colab dev scripts),
  for choosing pipeline settings.

The older `--old A --new B` form still works and names them "old" and "new".
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silent_bug_agent.mutation.blind import blind_copy, expected_bug  # noqa: E402
from silent_bug_agent.mutation.dataset import holdout_jobs, read  # noqa: E402


def scripts(sets: Sequence[str] = ("old",), data: Optional[Path] = None) -> List[Dict[str, str]]:
    out = []
    if "old" in sets:
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
    wanted = {"colab": ("colab_test",), "dev": ("holdout_dev", "colab_dev")}
    splits = [split for name in sets if name in wanted for split in wanted[name]]
    if splits:
        if data is None:
            raise SystemExit("--scripts colab/dev needs --data (the v4 dataset), which says which versions to keep")
        names = {"colab_test": "Colab test", "colab_dev": "Colab dev", "holdout_dev": "hand-written dev"}
        seen = set()
        for row in read(data):
            if row["split"] in splits and (row["group"], row["label"], row["code"]) not in seen:
                seen.add((row["group"], row["label"], row["code"]))
                index = sum(1 for o in out if o["name"].startswith(f"{row['group']} ({row['label']})"))
                suffix = f" #{index + 1}" if index else ""
                out.append({"name": f"{row['group']} ({row['label']}){suffix}", "label": row["label"],
                            "code": row["code"], "set": names[row["split"]]})
    return out


def run_version(repo: Path, code: str, timeout: float, extra: Sequence[str] = ()) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sba_versions_") as folder:
        script = Path(folder) / "train.py"
        script.write_text(code, encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(repo), SBA_HOME=str(Path(folder) / "memory"))
        env.pop("GROQ_API_KEY", None)
        env.pop("CEREBRAS_API_KEY", None)
        started = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-m", "silent_bug_agent", "fix", str(script), "--dry-run",
                                   "--no-llm", "--json", "--seeds", "5", *extra], cwd=folder, env=env,
                                  capture_output=True, text=True, timeout=timeout)
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
            "causes": [h["bug_id"] for h in last.get("hypotheses") or []][:5],
            "patched": bool(last.get("accepted_diff")), "fixed_bug": accepted["bug_id"] if accepted else None,
            "learned": (last.get("learned") or {}).get("top") if isinstance(last.get("learned"), dict) else None}


def summarise(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    buggy = [r for r in rows if r["label"] != "clean"]
    healthy = [r for r in rows if r["label"] == "clean"]
    times = sorted(r[key]["seconds"] for r in rows)
    return {"buggy": len(buggy), "healthy": len(healthy),
            "repaired": sum(r[key].get("patched", False) for r in buggy),
            "right_cause_first": sum(r[key].get("top_cause") == r["label"] for r in buggy),
            "right_cause_in_top3": sum(r["label"] in (r[key].get("causes") or [])[:3] for r in buggy),
            # "named correctly": the top cause for a buggy script, nothing (or no patch) for a healthy one
            "named_correctly": sum(r[key].get("top_cause") == r["label"] for r in buggy)
            + sum(not r[key].get("patched", False) and not r[key].get("top_cause") for r in healthy),
            "planted_bug_fixed": sum(r[key].get("fixed_bug") == r["label"] for r in buggy),
            "healthy_changed": sum(r[key].get("patched", False) for r in healthy),
            "errors": sum("error" in r[key] for r in rows),
            "total_seconds": round(sum(times), 1),
            "median_seconds": times[len(times) // 2] if times else None,
            "median_seconds_buggy": sorted(r[key]["seconds"] for r in buggy)[len(buggy) // 2] if buggy else None}


def parse_version(text: str) -> Tuple[str, Path, List[str]]:
    name, _, rest = text.partition("=")
    path, _, extra = rest.partition(":")
    return name, Path(path).resolve(), extra.split()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="append", default=[], help="name=checkout[:extra fix arguments]")
    parser.add_argument("--old", default=None, help="checkout of the older version (two-version form)")
    parser.add_argument("--new", default=str(REPO_ROOT), help="checkout of the newer version (two-version form)")
    parser.add_argument("--scripts", nargs="+", default=["old"], choices=["old", "colab", "dev"])
    parser.add_argument("--data", default=None, help="the v4 dataset (for --scripts colab or dev)")
    parser.add_argument("--workers", type=int, default=2, help="scripts at once (each runs every version in turn)")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--json", default=None)
    options = parser.parse_args(argv)
    versions = [parse_version(v) for v in options.version]
    if options.old:
        versions = [("old", Path(options.old).resolve(), []), ("new", Path(options.new).resolve(), [])] + versions
    if not versions:
        parser.error("give --version name=path (one or more), or --old")

    def each(item: Dict[str, str]) -> Dict[str, Any]:
        row = dict(item)
        for name, repo, extra in versions:
            row[name] = run_version(repo, item["code"], options.timeout, extra)
        row.pop("code")
        print(f"{row['name'][:52]:52} " + " | ".join(f"{name} {row[name].get('seconds')}s {row[name].get('outcome')}"
                                                     for name, _, _ in versions), flush=True)
        return row

    with ThreadPoolExecutor(options.workers) as pool:
        rows = list(pool.map(each, scripts(options.scripts, Path(options.data) if options.data else None)))
    result = {"versions": {name: {"path": str(repo), "arguments": extra} for name, repo, extra in versions},
              "rows": rows, "summary": {}}
    if options.old:
        result.update({"old": str(versions[0][1]), "new": str(versions[1][1])})
    for subset in sorted({r["set"] for r in rows}) + ["all"]:
        chosen = rows if subset == "all" else [r for r in rows if r["set"] == subset]
        result["summary"][subset] = {name: summarise(chosen, name) for name, _, _ in versions}
        print(subset, json.dumps(result["summary"][subset]))
    if options.json:
        Path(options.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
