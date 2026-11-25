"""Train the trace classifier and say how good it is, honestly.

    python -m silent_bug_agent.mutation.dataset --templates 80 --out traces.jsonl
    python -m silent_bug_agent.learned.train --data traces.jsonl --out silent_bug_agent/learned/weights.json

The data has three jobs, and each split has one:

* learn: "train" (generated scripts) and "colab_train" (Colab-notebook scripts with
  operator-planted bugs). Cross-validation over them, holding out whole scripts, says
  how well the model does on scripts written the same way.
* choose: "holdout_dev" (examples/holdout/ 01 to 04) and "colab_dev" (four Colab
  scripts). The model's settings (hidden layer size, regularisation, whether some
  training copies hide the code features) are chosen on them, and so is the
  calibration temperature; only their mistakes are printed.
* report: "holdout_test", "benchmark" and "colab_test". Never used to choose anything.

The settings with the best accuracy on the development scripts (averaged over three
seeds; on a tie the simpler model wins) are kept, and the model is refitted on all the
training runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..mutation.dataset import read
from .code import CODE_FEATURES
from .features import FEATURES, vectorize
from .model import WEIGHTS, TraceClassifier

# simplest first; "mask" adds a copy of every training run with the code features hidden
CANDIDATES = tuple({"hidden": h, "l2": l2, "mask": mask} for mask in (False, True) for h in (0, 16, 32)
                   for l2 in (1e-3, 1e-2))
TRAIN_SPLITS = ("train", "colab_train")
DEV_SPLITS = ("holdout_dev", "colab_dev")
TEST_SPLITS = ("holdout_test", "benchmark", "colab_test")
REPORTS = (("dev", DEV_SPLITS), ("holdout_test", ("holdout_test",)), ("benchmarks", ("benchmark",)),
           ("old_test", ("holdout_test", "benchmark")), ("colab_test", ("colab_test",)), ("test", TEST_SPLITS))
CODE_COLUMNS = [FEATURES.index(name) for name in CODE_FEATURES]


def _matrix(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[str]]:
    return (np.asarray([vectorize(r["report"], r.get("code")) for r in rows], dtype=float), [r["label"] for r in rows],
            [r["group"] for r in rows])


def masked(X: np.ndarray, y: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Every run twice: as it is, and with the code features hidden (as if the code said nothing)."""
    hidden = X.copy()
    hidden[:, CODE_COLUMNS] = 0.0
    return np.vstack([X, hidden]), list(y) + list(y)


def fit(X: np.ndarray, y: List[str], params: Dict[str, Any], seed: int = 0,
        classes: Optional[Sequence[str]] = None) -> TraceClassifier:
    if params.get("mask"):
        X, y = masked(X, y)
    return TraceClassifier.fit(X, y, hidden=params["hidden"], l2=params["l2"], seed=seed, classes=classes)


def temperature(model: TraceClassifier, X: np.ndarray, y: Sequence[str]) -> float:
    """The logit temperature with the lowest log loss on (X, y)."""
    index = {c: i for i, c in enumerate(model.classes)}
    rows = [(i, index[label]) for i, label in enumerate(y) if label in index]
    if not rows:
        return 1.0
    logits = model.logits(X)
    best = (float("inf"), 1.0)
    for t in np.round(np.arange(0.5, 5.01, 0.1), 2):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -float(np.mean([logp[i, j] for i, j in rows]))
        best = min(best, (nll, float(t)))
    return best[1]


def _folds(groups: Sequence[str], k: int, seed: int) -> List[np.ndarray]:
    unique = sorted(set(groups))
    np.random.default_rng(seed).shuffle(unique)
    fold_of = {g: i % k for i, g in enumerate(unique)}
    return [np.asarray([fold_of[g] == f for g in groups]) for f in range(k)]


def score(truth: Sequence[str], predicted: Sequence[str]) -> Dict[str, Any]:
    per_class: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for t, p in zip(truth, predicted):
        per_class[t][1] += 1
        per_class[t][0] += int(t == p)
    recall = {c: round(hit / total, 3) for c, (hit, total) in sorted(per_class.items())}
    return {"accuracy": round(float(np.mean([t == p for t, p in zip(truth, predicted)])), 3) if truth else None,
            "macro_recall": round(float(np.mean(list(recall.values()))), 3) if recall else None,
            "recall": recall, "n": len(truth)}


def cross_validate(X: np.ndarray, y: List[str], groups: List[str], params: Dict[str, Any], k: int = 5,
                   seed: int = 0) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    predicted: List[Optional[str]] = [None] * len(y)
    classes = sorted(set(y))
    for test in _folds(groups, k, seed):
        model = fit(X[~test], [l for l, t in zip(y, test) if not t], params, classes=classes, seed=seed)
        P = model.probabilities(X[test])
        for i, row in zip(np.flatnonzero(test), P):
            predicted[i] = classes[int(np.argmax(row))]
    pairs = [(t, p or "") for t, p in zip(y, predicted)]
    return score(y, [p for _, p in pairs]), pairs


def confusion(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    counts = Counter((t, p) for t, p in pairs if t != p)
    return [f"{t} -> {p}: {n}" for (t, p), n in counts.most_common(8)]


def _accuracy(model: TraceClassifier, X: np.ndarray, y: Sequence[str]) -> float:
    predicted = [model.classes[int(i)] for i in model.probabilities(X).argmax(axis=1)]
    return float(np.mean([p == t for p, t in zip(predicted, y)])) if len(y) else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m silent_bug_agent.learned.train")
    parser.add_argument("--data", required=True, help="JSON lines from silent_bug_agent.mutation.dataset")
    parser.add_argument("--out", default=str(WEIGHTS))
    parser.add_argument("--folds", type=int, default=5)
    options = parser.parse_args(argv)

    rows = list(read(Path(options.data)))
    train = [r for r in rows if r["split"] in TRAIN_SPLITS]
    dev = [r for r in rows if r["split"] in DEV_SPLITS]
    X, y, groups = _matrix(train)
    print(f"{len(train)} training runs from {len(set(groups))} templates; classes: {dict(Counter(y))}")
    if not dev:
        print("no development scripts in the data: choosing by cross-validation instead")
    Xd, yd, _ = _matrix(dev) if dev else (None, [], [])

    best: Tuple[float, Dict[str, Any]] = (-1.0, CANDIDATES[0])
    for params in CANDIDATES:
        if dev:
            score_ = float(np.mean([_accuracy(fit(X, y, params, seed=seed), Xd, yd) for seed in range(3)]))
        else:
            score_ = cross_validate(X, y, groups, params, k=options.folds)[0]["accuracy"]
        print(f"  hidden={params['hidden']:<3} l2={params['l2']:<6} mask={params['mask']!s:<5} "
              f"{'development' if dev else 'cross-validated'} accuracy {score_:.3f}")
        if score_ > best[0] + 1e-9:
            best = (score_, params)
    params = best[1]
    cv, pairs = cross_validate(X, y, groups, params, k=options.folds)
    model = fit(X, y, params)
    if dev:
        model.meta["temperature"] = temperature(model, Xd, yd)

    reports: Dict[str, Any] = {}
    mistakes: List[str] = []
    for name, splits in REPORTS:
        chosen = [r for r in rows if r["split"] in splits]
        if not chosen:
            reports[name] = {"n": 0}
            continue
        Xs, ys, gs = _matrix(chosen)
        predicted = [model.classes[int(i)] for i in model.probabilities(Xs).argmax(axis=1)]
        reports[name] = score(ys, predicted)
        if name == "dev":  # mistakes are studied on the development scripts only, never on the test scripts
            mistakes = [f"{g} ({t}) -> {p}" for g, t, p in zip(gs, ys, predicted) if t != p]

    started = time.perf_counter()
    for _ in range(1000):
        model.probabilities(X[:1])
    per_prediction_us = (time.perf_counter() - started) * 1000

    model.meta.update({"version": 4, "trained_on_runs": len(train), "templates": len(set(groups)), "chosen": params,
                       "cross_validated": cv, **reports, "dev_mistakes": mistakes,
                       "trained": time.strftime("%Y-%m-%d")})
    path = model.save(Path(options.out))
    print(f"kept hidden={params['hidden']} l2={params['l2']}; cross-validated {cv['accuracy']:.3f}; "
          f"most common mistakes there: {confusion(pairs) or 'none'}")
    for name, _ in REPORTS:
        if reports[name].get("n"):
            print(f"{name}: accuracy {reports[name]['accuracy']:.3f} on {reports[name]['n']} runs")
    for line in mistakes:
        print(f"  wrong on a development script: {line}")
    print(f"saved {path} ({path.stat().st_size / 1024:.1f} KB); one prediction takes {per_prediction_us:.0f} microseconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
