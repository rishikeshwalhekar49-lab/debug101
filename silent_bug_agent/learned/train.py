"""Train the trace classifier and say how good it is, honestly.

    python -m silent_bug_agent.mutation.dataset --templates 80 --out traces.jsonl
    python -m silent_bug_agent.learned.train --data traces.jsonl --out silent_bug_agent/learned/weights.json

Two numbers are reported, and they answer different questions:

* cross-validation over generated scripts, holding out whole templates: how well
  it recognises a planted bug in a script it has not seen, written the same way;
* the project's own benchmarks (blind copies): hand-written scripts it never
  trained on. This is the number to quote, because it is the closest to real use.

Both a softmax regression and a one-hidden-layer network are tried; the one with
the better cross-validated accuracy is kept, then refitted on every training run.
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
from .features import vectorize
from .model import WEIGHTS, TraceClassifier

CANDIDATES = ({"hidden": 0, "l2": 1e-3}, {"hidden": 32, "l2": 1e-3})


def _matrix(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[str]]:
    return (np.asarray([vectorize(r["report"]) for r in rows], dtype=float), [r["label"] for r in rows],
            [r["group"] for r in rows])


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
        model = TraceClassifier.fit(X[~test], [l for l, t in zip(y, test) if not t], classes=classes, **params)
        P = model.probabilities(X[test])
        for i, row in zip(np.flatnonzero(test), P):
            predicted[i] = classes[int(np.argmax(row))]
    pairs = [(t, p or "") for t, p in zip(y, predicted)]
    return score(y, [p for _, p in pairs]), pairs


def confusion(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    counts = Counter((t, p) for t, p in pairs if t != p)
    return [f"{t} -> {p}: {n}" for (t, p), n in counts.most_common(8)]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m silent_bug_agent.learned.train")
    parser.add_argument("--data", required=True, help="JSON lines from silent_bug_agent.mutation.dataset")
    parser.add_argument("--out", default=str(WEIGHTS))
    parser.add_argument("--folds", type=int, default=5)
    options = parser.parse_args(argv)

    rows = list(read(Path(options.data)))
    train = [r for r in rows if r["split"] == "train"]
    bench = [r for r in rows if r["split"] == "benchmark"]
    X, y, groups = _matrix(train)
    print(f"{len(train)} training runs from {len(set(groups))} templates; classes: {dict(Counter(y))}")

    results = []
    for params in CANDIDATES:
        started = time.perf_counter()
        cv, pairs = cross_validate(X, y, groups, params, k=options.folds)
        results.append((cv["accuracy"], params, cv, pairs))
        print(f"  hidden={params['hidden']:<3} held-out-template accuracy {cv['accuracy']:.3f}, "
              f"macro recall {cv['macro_recall']:.3f} ({time.perf_counter() - started:.0f}s)")
    _, params, cv, pairs = max(results, key=lambda r: r[0])
    model = TraceClassifier.fit(X, y, **params)

    bench_score: Dict[str, Any] = {"n": 0}
    mistakes: List[str] = []
    if bench:
        Xb, yb, gb = _matrix(bench)
        P = model.probabilities(Xb)
        predicted = [model.classes[int(np.argmax(p))] for p in P]
        bench_score = score(yb, predicted)
        mistakes = [f"{g} ({t}) -> {p}" for g, t, p in zip(gb, yb, predicted) if t != p]

    started = time.perf_counter()
    for _ in range(1000):
        model.probabilities(X[:1])
    per_prediction_us = (time.perf_counter() - started) * 1000

    model.meta.update({"trained_on_runs": len(train), "templates": len(set(groups)),
                       "cross_validated": cv, "benchmarks": bench_score, "benchmark_mistakes": mistakes,
                       "trained": time.strftime("%Y-%m-%d")})
    path = model.save(Path(options.out))
    print(f"kept hidden={params['hidden']}; most common mistakes: {confusion(pairs) or 'none'}")
    print(f"per-class recall (held-out templates): {cv['recall']}")
    if bench:
        print(f"benchmarks (hand-written, never trained on): accuracy {bench_score['accuracy']:.3f} "
              f"on {bench_score['n']} runs; recall {bench_score['recall']}")
        for line in mistakes:
            print(f"  wrong: {line}")
    print(f"saved {path} ({path.stat().st_size / 1024:.1f} KB); one prediction takes {per_prediction_us:.0f} microseconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
