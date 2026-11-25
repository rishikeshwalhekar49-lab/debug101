"""Score every version of the trace classifier on the same test runs.

    python examples/compare_classifier_versions.py --data traces_v4.jsonl --v2-data traces_v2.jsonl \
        --v4-data traces_v4_150.jsonl --json docs/images/classifier_versions.json

Each version keeps its own features and weights, taken from git, and reads the same
scripts' traced runs (the version 4 tracer records everything the older ones read,
measured the same way). v1 to v3 were trained on 60-step traces and are scored on
60-step traces (--data); v4 traces 150 steps, as long as the monitor's run it runs
beside, and is scored on 150-step traces of the same scripts (--v4-data).

* v1: commit 0728d71 (24 trace features, one hidden layer);
* v2: never committed. It is rebuilt exactly: v1's 24 features plus the 8 loss
  features, one hidden layer of 16 units, trained on the v2 dataset. The rebuilt model
  reproduces v2's recorded scores (76.9% on the tuning scripts, 65.5% on the
  hand-written test, 84.6% on the benchmarks);
* v3: commit b7179ad (34 features, no hidden layer);
* v4: the working tree (trace features, backward-pass features and code features).

Test sets: the "old test" (examples/holdout 05 to 10 and the blind benchmarks, 84
runs: v3 was tuned on mistakes seen there, and v4's grad-clearing fix was seen there
first) and the "Colab test" (examples/colab, never used to choose anything).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silent_bug_agent.learned.features import _loss_features, vectorize  # noqa: E402
from silent_bug_agent.learned.model import TraceClassifier, load_default  # noqa: E402
from silent_bug_agent.mutation.dataset import read  # noqa: E402

CLEAN = "clean"
SETS = {"old test": ("holdout_test", "benchmark"), "Colab test": ("colab_test",),
        "tuning (dev)": ("holdout_dev", "colab_dev")}
LOSS_NAMES = ["sum_reduction", "start_vs_chance", "end_vs_chance", "has_chance", "val_slope", "val_vs_train_end",
              "val_end_vs_chance", "has_validation_loss"]


def _from_git(commit: str, folder: Path, name: str) -> types.ModuleType:
    """Import silent_bug_agent/learned/{features,model}.py of `commit` as a package `name`."""
    package = folder / name
    package.mkdir(parents=True, exist_ok=True)
    for file in ("__init__.py", "features.py", "model.py", "weights.json"):
        text = subprocess.run(["git", "show", f"{commit}:silent_bug_agent/learned/{file}"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout
        (package / file).write_text(text, encoding="utf-8")
    module = types.ModuleType(name)
    module.__path__ = [str(package)]
    sys.modules[name] = module
    for part in ("features", "model"):
        spec = importlib.util.spec_from_file_location(f"{name}.{part}", package / f"{part}.py")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.{part}"] = loaded
        spec.loader.exec_module(loaded)  # type: ignore[union-attr]
        setattr(module, part, loaded)
    return module


def predictor_from_git(commit: str, folder: Path, name: str) -> Callable[[Dict[str, Any]], str]:
    module = _from_git(commit, folder, name)
    model = module.model.TraceClassifier.load(folder / name / "weights.json")
    return lambda row: model.classes[int(model.probabilities(np.asarray([module.features.vectorize(row["report"])]))
                                         .argmax())]


def rebuilt_v2(v1: types.ModuleType, v2_rows: List[Dict[str, Any]]) -> Callable[[Dict[str, Any]], str]:
    def vector(report: Dict[str, Any]) -> List[float]:
        loss = _loss_features((report.get("trace") or {}).get("loss_calls") or [])
        return list(v1.features.vectorize(report)) + [loss[k] for k in LOSS_NAMES]

    train = [r for r in v2_rows if r["split"] == "train"]
    model = TraceClassifier.fit(np.asarray([vector(r["report"]) for r in train]), [r["label"] for r in train],
                                hidden=16, l2=1e-3, seed=0)
    checks = {}
    for split in ("holdout_dev", "holdout_test", "benchmark"):
        chosen = [r for r in v2_rows if r["split"] == split]
        predicted = model.probabilities(np.asarray([vector(r["report"]) for r in chosen])).argmax(axis=1)
        checks[split] = round(float(np.mean([model.classes[p] == r["label"] for p, r in zip(predicted, chosen)])), 3)
    print(f"rebuilt v2 on its own data: {checks} (recorded: holdout_dev 0.769, holdout_test 0.655, benchmark 0.846)")
    return lambda row: model.classes[int(model.probabilities(np.asarray([vector(row["report"])])).argmax())]


def metrics(truth: Sequence[str], predicted: Sequence[str]) -> Dict[str, Any]:
    healthy = [p for t, p in zip(truth, predicted) if t == CLEAN]
    return {"n": len(truth), "correct": int(sum(t == p for t, p in zip(truth, predicted))),
            "accuracy": float(np.mean([t == p for t, p in zip(truth, predicted)])) if truth else None,
            "healthy": len(healthy), "healthy_flagged": int(sum(p != CLEAN for p in healthy))}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="the v4 dataset")
    parser.add_argument("--v2-data", required=True, help="the v2 dataset, to rebuild v2")
    parser.add_argument("--v4-data", default=None, help="v4's own traces of the same scripts (default: --data)")
    parser.add_argument("--json", default=None)
    options = parser.parse_args(argv)

    rows = list(read(Path(options.data)))
    v4_rows = list(read(Path(options.v4_data))) if options.v4_data else rows
    with tempfile.TemporaryDirectory(prefix="sba_versions_") as folder:
        v1 = _from_git("0728d71", Path(folder), "sba_v1")
        predictors: Dict[str, Callable[[Dict[str, Any]], str]] = {
            "v1": predictor_from_git("0728d71", Path(folder), "sba_v1"),
            "v2": rebuilt_v2(v1, list(read(Path(options.v2_data)))),
            "v3": predictor_from_git("b7179ad", Path(folder), "sba_v3"),
        }
        current = load_default()
        if current is None:
            raise SystemExit("the working tree has no trained v4 classifier (silent_bug_agent/learned/weights.json)")
        predictors["v4"] = lambda row: current.classes[int(current.probabilities(
            np.asarray([vectorize(row["report"], row.get("code"))])).argmax())]
        result: Dict[str, Any] = {"sets": {name: list(splits) for name, splits in SETS.items()}, "versions": {}}
        for version, predict in predictors.items():
            result["versions"][version] = {}
            for name, splits in SETS.items():
                chosen = [r for r in (v4_rows if version == "v4" else rows) if r["split"] in splits]
                result["versions"][version][name] = metrics([r["label"] for r in chosen], [predict(r) for r in chosen])
            print(version, {k: f"{v['correct']}/{v['n']}" for k, v in result["versions"][version].items()})
    if options.json:
        Path(options.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
