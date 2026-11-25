"""Build the training set for the bug classifier: generated scripts, mutated, traced.

    python -m silent_bug_agent.mutation.dataset --templates 60 --out traces.jsonl

For each generated healthy script, every mutation operator that applies plants
one bug, and every version (the healthy one too) is run for 60 optimizer steps
under `script_runner --trace`. Each line of the output is one run: its label (a
taxonomy bug id, or "clean"), the template it came from (so an evaluation can
hold out whole templates), and the raw trace, from which `learned/features.py`
computes the features. Runs that crash are left out: a crash is not a silent bug.

Each generated script also gets a second healthy version that learns more slowly (a
4x larger batch and half the learning rate), because a slow healthy run is what the
classifier most often mistook for a bug.

Two hand-written test sets are traced too, unless `--no-benchmarks`: blind copies of
the project's benchmarks (split "benchmark") and the separately written scripts in
`examples/holdout/` (splits "holdout_dev", which models may be tuned on, and
"holdout_test", which is only ever used to report results).

Version 4 adds training scripts from Google Colab notebooks (`examples/colab/`), on
Colab's own sample data: some are training data ("colab_train", bugs planted by the
operators, each traced with three seeds), the rest are split into "colab_dev" and
"colab_test" with bugs planted by hand. It also checks labels: a planted bug must
make the run worse than the healthy version of the same script, with the same seed
(`has_effect`). Buggy versions that change nothing are dropped, and so is every
version of a script whose healthy run diverges. The older test sets (benchmarks and
examples/holdout) are kept exactly as they were, so all versions are compared on
the same runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
        jobs.append({"group": group, "label": CLEAN, "code": healthy, "spec": spec.to_dict(), "split": "train",
                     "reference": True})
        slow = replace(spec, batch_size=min(256, spec.batch_size * 4), lr=round(spec.lr / 2, 6))
        jobs.append({"group": group, "label": CLEAN, "code": render(slow), "spec": slow.to_dict(), "split": "train"})
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


def holdout_jobs() -> List[Dict[str, Any]]:
    """The hand-written scripts in examples/holdout/, every version (see variants.py there)."""
    import importlib.util

    path = REPO / "examples" / "holdout" / "variants.py"
    if not path.is_file():
        return []
    spec = importlib.util.spec_from_file_location("holdout_variants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return [{"group": v["script"].removesuffix(".py"), "label": v["label"], "code": v["code"],
             "split": f"holdout_{v['split']}"} for v in module.versions()]


def colab_versions() -> List[Dict[str, Any]]:
    import importlib.util

    path = REPO / "examples" / "colab" / "variants.py"
    if not path.is_file():
        return []
    spec = importlib.util.spec_from_file_location("colab_variants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.versions()


COLAB_TRAIN_SEEDS = (0, 1, 2)


def colab_jobs(evaluation: bool = True) -> List[Dict[str, Any]]:
    """Colab-notebook scripts: training scripts with operator-planted bugs, and the dev/test splits as written."""
    jobs: List[Dict[str, Any]] = []
    for version in colab_versions():
        group = version["script"].removesuffix(".py")
        if version["split"] == "train":
            healthy = version["code"]
            first = not any(j["group"] == group for j in jobs)
            for seed in COLAB_TRAIN_SEEDS:
                jobs.append({"group": group, "label": CLEAN, "code": healthy, "split": "colab_train", "seed": seed,
                             "reference": first})
                if first:
                    for bug, code in mutants(healthy).items():
                        jobs.append({"group": group, "label": bug, "code": code, "split": "colab_train", "seed": seed})
        elif evaluation:
            first = version["label"] == CLEAN and not any(j["group"] == group for j in jobs)
            jobs.append({"group": group, "label": version["label"], "code": version["code"],
                         "split": f"colab_{version['split']}", "seed": 0, "reference": first})
    return jobs


def _finite_tail(values: List[float], n: int = 10) -> Optional[float]:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite[-n:]) / len(finite[-n:]) if finite else None


def _validation(report: Dict[str, Any]) -> Optional[float]:
    calls = [c["value"] for c in (report.get("trace") or {}).get("loss_calls") or []
             if not c.get("train") and c.get("value") is not None and math.isfinite(c["value"])]
    return sum(calls[-3:]) / len(calls[-3:]) if calls else None


def diverged(report: Dict[str, Any]) -> bool:
    losses = report.get("losses") or []
    return bool(losses) and (not all(math.isfinite(v) for v in losses) or abs(losses[-1]) > 1e6)


def has_effect(label: str, bug: Dict[str, Any], healthy: Dict[str, Any], margin: float = 0.05,
               definitional: bool = False) -> bool:
    """Does the planted bug make this run worse than the healthy one (same script, same seed)?

    Losses are compared as they are, except for `wrong_loss_reduction`, whose summed loss
    is on another scale: there the share of the loss that training removed is compared.
    `train_eval_mode_mismatch` only shows in validation, so any change there counts, and
    with no validation in the run it is kept (a model with dropout or batch norm left in
    training mode for validation is wrong by definition). With `definitional`, a summed
    loss planted by the operator is kept whatever it did: the taxonomy defines that bug
    by the code (a sum where the step size was set for a mean), and Adam hides its effect.
    """
    if diverged(bug):
        return True
    train_bug, train_ok = _finite_tail(bug.get("losses") or []), _finite_tail(healthy.get("losses") or [])
    val_bug, val_ok = _validation(bug), _validation(healthy)
    if label == "train_eval_mode_mismatch":
        if val_bug is None or val_ok is None:
            return True
        return abs(math.log(max(val_bug, 1e-12) / max(val_ok, 1e-12))) > 0.02
    if label == "wrong_loss_reduction" and definitional:
        return True  # defined by the code: a summed loss with a step size meant for a mean
    if label == "wrong_loss_reduction":
        def removed(report: Dict[str, Any]) -> Optional[float]:
            finite = [v for v in report.get("losses") or [] if math.isfinite(v) and v > 0]
            return (sum(finite[-10:]) / len(finite[-10:])) / (sum(finite[:3]) / len(finite[:3])) if finite else None
        r_bug, r_ok = removed(bug), removed(healthy)
        return r_bug is not None and r_ok is not None and r_bug > r_ok * (1 + 2 * margin)
    worse_train = train_bug is not None and train_ok is not None and train_bug > train_ok * (1 + margin) + 1e-6
    worse_val = val_bug is not None and val_ok is not None and val_bug > val_ok * (1 + margin) + 1e-6
    return worse_train or worse_val


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


CHECKED_SPLITS = ("train", "colab_train", "colab_dev", "colab_test")  # the older test sets stay as they were
CHECK_STEPS = 300  # dev and test labels are checked on a longer run: some scripts barely move in 60 steps
LONG_CHECK = ("colab_dev", "colab_test")


def _check_report(report: Dict[str, Any]) -> Dict[str, Any]:
    calls = [c for c in (report.get("trace") or {}).get("loss_calls") or [] if not c.get("train")]
    return {"losses": report.get("losses", []), "trace": {"loss_calls": calls}}


def check_labels(rows: List[Dict[str, Any]], log: Any = print) -> List[Dict[str, Any]]:
    """Drop buggy versions without an effect, and scripts whose healthy run diverges (see `has_effect`)."""
    reference: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        if row["label"] == CLEAN and row.get("reference"):
            reference.setdefault((row["group"], row.get("seed", 0)), row)
    kept, dropped = [], []
    for row in rows:
        if row["split"] not in CHECKED_SPLITS:
            kept.append(row)
            continue
        healthy = reference.get((row["group"], row.get("seed", 0)))
        if healthy is None or diverged(healthy.get("check") or healthy["report"]):
            dropped.append((row, "the healthy version diverged or crashed"))
        elif row["label"] != CLEAN and not has_effect(row["label"], row.get("check") or row["report"],
                                                      healthy.get("check") or healthy["report"],
                                                      definitional=row["split"] in ("train", "colab_train")):
            dropped.append((row, "the bug made no difference"))
        else:
            kept.append(row)
    for row, why in dropped:
        if row["split"] in ("colab_dev", "colab_test"):
            log(f"  left out of {row['split']}: {row['group']} ({row['label']}): {why}")
    log(f"label check: kept {len(kept)}, left out {len(dropped)} "
        f"({sum(1 for r, _ in dropped if r['split'] == 'train')} generated, "
        f"{sum(1 for r, _ in dropped if r['split'] == 'colab_train')} Colab training, "
        f"{sum(1 for r, _ in dropped if r['split'] in ('colab_dev', 'colab_test'))} Colab dev/test)")
    return kept


def build(out: Path, *, templates: int = 60, seed: int = 0, benchmarks: bool = True, workers: Optional[int] = None,
          max_steps: int = 60, log: Any = print) -> Dict[str, int]:
    jobs = template_jobs(templates, seed) + colab_jobs(evaluation=benchmarks) \
        + (benchmark_jobs() + holdout_jobs() if benchmarks else [])
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    started = time.perf_counter()
    done = {"kept": 0, "crashed": 0}

    def one(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        report = trace_run(job["code"], max_steps=max_steps, seed=job.get("seed", 0))
        if not usable(report):
            return None
        row = {"group": job["group"], "label": job["label"], "split": job["split"], "spec": job.get("spec"),
               "seed": job.get("seed", 0), "reference": bool(job.get("reference")), "code": job["code"],
               "report": {"losses": report.get("losses", []), "trace": report["trace"]}}
        if job["split"] in LONG_CHECK:
            longer = trace_run(job["code"], max_steps=CHECK_STEPS, seed=job.get("seed", 0), timeout=300.0)
            if usable(longer):
                row["check"] = _check_report(longer)
        return row

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(workers) as pool:
        for index, row in enumerate(pool.map(one, jobs), start=1):
            if row is None:
                done["crashed"] += 1
            else:
                rows.append(row)
            if index % 25 == 0 or index == len(jobs):
                log(f"{index}/{len(jobs)} runs traced ({time.perf_counter() - started:.0f}s)")
    rows = check_labels(rows, log)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    done["kept"] = len(rows)
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
