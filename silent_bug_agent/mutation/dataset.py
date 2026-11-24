"""Build the training set for the bug classifier: generated scripts, mutated, traced.

    python -m silent_bug_agent.mutation.dataset --templates 60 --out traces.jsonl

For each generated healthy script, every mutation operator that applies plants
one bug, and every version (the healthy one too) is run for 60 optimizer steps
under `script_runner --trace`. Each line of the output is one run: its label (a
taxonomy bug id, or "clean"), the template it came from (so an evaluation can
hold out whole templates), and the raw trace, from which `learned/features.py`
computes the features. Runs that crash are left out: a crash is not a silent bug.

With `--benchmarks` the blind copies of the project's own benchmarks are traced
too, as a separate test set: hand-written scripts the classifier never trains on.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .blind import blind_copy, expected_bug
from .operators import OPERATORS, mutants
from .templates import random_spec, render

REPO = Path(__file__).resolve().parent.parent.parent
CLEAN = "clean"


def template_jobs(count: int, seed: int = 0) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    jobs: List[Dict[str, Any]] = []
    for index in range(count):
        spec = random_spec(rng, data_seed=seed * 10_000 + index)
        healthy = render(spec)
        group = f"template_{seed}_{index}"
        jobs.append({"group": group, "label": CLEAN, "code": healthy, "spec": spec.to_dict(), "split": "train"})
        for bug, code in mutants(healthy).items():
            jobs.append({"group": group, "label": bug, "code": code, "spec": spec.to_dict(), "split": "train"})
    return jobs


def benchmark_jobs() -> List[Dict[str, Any]]:
    """Blind benchmark copies whose bug the operators also plant, plus every healthy twin (labelled clean)."""
    jobs: List[Dict[str, Any]] = []
    paths = sorted((REPO / "examples" / "sample_scripts").glob("benchmark_[0-9]*.py")) \
        + sorted((REPO / "examples" / "new").glob("benchmark_[0-9]*.py"))
    for path in paths:
        source = path.read_text(encoding="utf-8")
        bug, buggy, healthy = expected_bug(source), blind_copy(source), blind_copy(source, fixed=True)
        if bug in OPERATORS:
            jobs.append({"group": path.stem, "label": bug, "code": buggy, "split": "benchmark"})
        if healthy != buggy:
            jobs.append({"group": path.stem, "label": CLEAN, "code": healthy, "split": "benchmark"})
    return jobs


def trace_run(code: str, *, sandbox: Any = None, max_steps: int = 60, timeout: float = 120.0,
              seed: int = 0, cwd: Optional[str] = None, as_file: Optional[str] = None,
              script_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run code once under `script_runner --trace`; returns the run report (with `status`)."""
    from ..validation.sandbox import SandboxConfig, SubprocessSandbox

    sandbox = sandbox or SubprocessSandbox(SandboxConfig(max_cpu_cores=2))
    result = sandbox.run_measured(code, seed=seed, max_steps=max_steps, timeout=timeout, trace=True, cwd=cwd,
                                  as_file=as_file, script_args=script_args)
    report = dict(result.run_report or {})
    report.setdefault("status", "ok" if result.success else "error")
    if result.timed_out:
        report["status"] = "timeout"
    return report


def usable(report: Dict[str, Any]) -> bool:
    """A run that finished, or that the step budget stopped on purpose, with a trace recorded."""
    return report.get("status") in ("ok", "stopped") and bool(report.get("trace"))


def build(out: Path, *, templates: int = 60, seed: int = 0, benchmarks: bool = True, workers: Optional[int] = None,
          max_steps: int = 60, log: Any = print) -> Dict[str, int]:
    jobs = template_jobs(templates, seed) + (benchmark_jobs() if benchmarks else [])
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    started = time.perf_counter()
    done = {"kept": 0, "crashed": 0}

    def one(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        report = trace_run(job["code"], max_steps=max_steps)
        if not usable(report):
            return None
        return {"group": job["group"], "label": job["label"], "split": job["split"], "spec": job.get("spec"),
                "report": {"losses": report.get("losses", []), "trace": report["trace"]}}

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as pool:
        for index, row in enumerate(pool.map(one, jobs), start=1):
            if row is None:
                done["crashed"] += 1
            else:
                done["kept"] += 1
                handle.write(json.dumps(row) + "\n")
            if index % 25 == 0 or index == len(jobs):
                log(f"{index}/{len(jobs)} runs traced ({time.perf_counter() - started:.0f}s)")
    return done


def read(path: Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m silent_bug_agent.mutation.dataset", description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="JSON-lines file to write")
    parser.add_argument("--templates", type=int, default=60, help="Healthy scripts to generate and mutate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-benchmarks", action="store_true", help="Leave out the benchmark test set")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=60)
    options = parser.parse_args(argv)
    counts = build(Path(options.out), templates=options.templates, seed=options.seed,
                   benchmarks=not options.no_benchmarks, workers=options.workers, max_steps=options.max_steps)
    print(f"wrote {counts['kept']} runs to {options.out} ({counts['crashed']} crashed and were left out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
