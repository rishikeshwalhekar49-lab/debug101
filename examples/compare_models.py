"""Compare the trace classifier with other ways of naming the bug in a training run.

    python -m silent_bug_agent.mutation.dataset --templates 80 --out traces.jsonl
    python examples/compare_models.py --data traces.jsonl --seeds 5 --rules --json comparison.json

Every model answers the same question for the same runs: which of nine planted bugs,
or "clean". The data has three roles (see silent_bug_agent/learned/train.py):

* training scripts ("train", generated, and "colab_train", from Colab notebooks): models
  learn from them; 5-fold cross-validation over them, holding out whole scripts, is
  the "generated" score;
* development scripts ("holdout_dev" and "colab_dev"): each model family picks its
  settings here, from a small grid, the same way for every family;
* test scripts, never used to choose anything: the hand-written ones and the blind
  project benchmarks (the "old test", 84 runs, the same runs for every version) and
  the Colab-notebook test scripts ("colab_test"). "test" is all of them together.

Every model gets the same features: the run's trace and, from version 4, facts read
from the code (silent_bug_agent/learned/code.py).

With --seeds N every learned model is retrained N times (fold split and initialisation),
and scores are mean ± standard deviation.

Models: majority class; this project's hand-written detectors (--rules, slow: each
script is trained once; --reuse-rules takes their answers from an earlier --json);
the trace classifier with and without a hidden layer; random forest, gradient boosting,
k-NN and an RBF SVM on the same features (needs scikit-learn); and with --llm and a key,
gpt-oss-120b and gpt-oss-20b, given the trace statistics or the source code.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silent_bug_agent.learned.features import FEATURES, vectorize  # noqa: E402
from silent_bug_agent.learned.model import TraceClassifier  # noqa: E402
from silent_bug_agent.learned.train import DEV_SPLITS, TEST_SPLITS, TRAIN_SPLITS, _folds, masked  # noqa: E402
from silent_bug_agent.mutation.dataset import read  # noqa: E402

CLEAN = "clean"
REPORT_ONLY = {"unset_random_seed"}
SUBSETS = {"old_test": ("holdout_test", "benchmark"), "colab_test": ("colab_test",)}


# ----------------------------------------------------------------- metrics
def metrics(truth: Sequence[str], predicted: Sequence[str]) -> Dict[str, Any]:
    truth, predicted = list(truth), list(predicted)
    if not truth:
        return {"n": 0}
    healthy = [p for t, p in zip(truth, predicted) if t == CLEAN]
    buggy = [(t, p) for t, p in zip(truth, predicted) if t != CLEAN]
    classes = sorted(set(truth))
    recall = {c: float(np.mean([p == c for t, p in zip(truth, predicted) if t == c])) for c in classes}
    return {"n": len(truth), "accuracy": float(np.mean([t == p for t, p in zip(truth, predicted)])),
            "macro_recall": float(np.mean(list(recall.values()))),
            "bugs_named_correctly": float(np.mean([t == p for t, p in buggy])) if buggy else None,
            "false_alarms_on_healthy": float(np.mean([p != CLEAN for p in healthy])) if healthy else None,
            "healthy": len(healthy), "recall": recall}


def spread(values: Sequence[float]) -> Dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "runs": len(values)}


# ---------------------------------------------------------------- models
class Ours:
    def __init__(self, seed: int = 0, hidden: int = 32, l2: float = 1e-3, mask: bool = False) -> None:
        self.seed, self.hidden, self.l2, self.mask = seed, hidden, l2, mask

    def fit(self, X: np.ndarray, y: List[str]) -> "Ours":
        if self.mask:
            X, y = masked(X, y)
        self.model = TraceClassifier.fit(X, y, hidden=self.hidden, l2=self.l2, seed=self.seed)
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        return [self.model.classes[i] for i in self.model.probabilities(X).argmax(axis=1)]

    def size(self) -> Dict[str, Any]:
        blob = json.dumps(self.model.to_dict(), separators=(",", ":")).encode()
        return {"bytes": len(blob), "parameters": int(sum(W.size + b.size for W, b in self.model.layers))}


class Majority:
    def __init__(self, seed: int = 0) -> None:
        pass

    def fit(self, X: np.ndarray, y: List[str]) -> "Majority":
        labels, counts = np.unique(y, return_counts=True)
        self.label = str(labels[int(np.argmax(counts))])
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        return [self.label] * len(X)

    def size(self) -> Dict[str, Any]:
        return {"bytes": None, "parameters": 0}


def _sklearn_builder(name: str) -> Optional[Callable[..., Any]]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        return None
    build = {
        "Random forest (300 trees)": lambda seed, leaf=1, feats="sqrt": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=leaf, max_features=feats, random_state=seed, class_weight="balanced"),
        "Gradient boosting": lambda seed, lr=0.1, leaves=31: HistGradientBoostingClassifier(
            learning_rate=lr, max_leaf_nodes=leaves, random_state=seed, class_weight="balanced"),
        "k-nearest neighbours": lambda seed, k=5: make_pipeline(StandardScaler(), KNeighborsClassifier(k)),
        "SVM (RBF kernel)": lambda seed, C=1.0: make_pipeline(
            StandardScaler(), SVC(C=C, class_weight="balanced", random_state=seed)),
    }[name]

    class Wrapped:
        def __init__(self, seed: int = 0, **params: Any) -> None:
            self.seed, self.params = seed, params

        def fit(self, X: np.ndarray, y: List[str]) -> "Wrapped":
            self.model = build(self.seed, **self.params).fit(X, y)
            return self

        def predict(self, X: np.ndarray) -> List[str]:
            return [str(p) for p in self.model.predict(X)]

        def size(self) -> Dict[str, Any]:
            return {"bytes": len(pickle.dumps(self.model)), "parameters": None}

    return Wrapped


FAMILIES: Dict[str, List[Dict[str, Any]]] = {  # each grid lists the simplest settings first
    "Majority class": [{}],
    # the shipped settings, exactly as train.py chose them on the tuning scripts (averaged over three seeds)
    "Trace classifier (ours, shipped)": [{"hidden": 16, "l2": 1e-3, "mask": True}],
    "Trace classifier, no hidden layer (ours)": [{"hidden": 0, "l2": l2, "mask": m} for m in (False, True)
                                                 for l2 in (1e-3, 1e-2, 1e-1)],
    "Random forest (300 trees)": [{"leaf": leaf, "feats": f} for leaf in (1, 3) for f in ("sqrt", 0.5)],
    "Gradient boosting": [{"lr": lr, "leaves": n} for lr in (0.05, 0.1) for n in (15, 31)],
    "k-nearest neighbours": [{"k": k} for k in (3, 5, 9, 15)],
    "SVM (RBF kernel)": [{"C": c} for c in (0.3, 1.0, 3.0, 10.0)],
}


def factory_for(name: str) -> Optional[Callable[..., Any]]:
    if name == "Majority class":
        return Majority
    if "(ours" in name:
        return Ours
    return _sklearn_builder(name)


def evaluate_family(name: str, factory: Callable[..., Any], data: Dict[str, Any], seeds: Sequence[int]) -> Dict:
    X, y, groups = data["X"], data["y"], data["groups"]
    # 1. choose the settings on the development scripts
    choice, best = FAMILIES[name][0], -1.0
    if data["yd"]:
        for params in FAMILIES[name]:
            predicted = factory(0, **params).fit(X, y).predict(data["Xd"])
            accuracy = float(np.mean([p == t for p, t in zip(predicted, data["yd"])]))
            if accuracy > best + 1e-9:
                choice, best = params, accuracy
    # 2. score those settings over the seeds
    per_seed = []
    for seed in seeds:
        predicted = [""] * len(y)
        for test in _folds(groups, 5, seed):
            model = factory(seed, **choice).fit(X[~test], [label for label, t in zip(y, test) if not t])
            for index, label in zip(np.flatnonzero(test), model.predict(X[test])):
                predicted[index] = label
        final = factory(seed, **choice).fit(X, y)
        test_predicted = final.predict(data["Xt"])
        per_seed.append((metrics(y, predicted), metrics(data["yd"], final.predict(data["Xd"])) if data["yd"] else {},
                         metrics(data["yt"], test_predicted), final, subsets(data, test_predicted)))
    final = per_seed[0][3]
    started = time.perf_counter()
    for row in data["Xt"]:
        final.predict(row[None, :])
    per_prediction = (time.perf_counter() - started) / max(1, len(data["Xt"]))
    return {"model": name, "kind": "trained on trace features", "settings": choice,
            "generated": per_seed[0][0], "dev": per_seed[0][1], "test": per_seed[0][2],
            "test_predictions": final.predict(data["Xt"]), "seconds_per_prediction": per_prediction,
            "subsets": per_seed[0][4],
            "over_seeds": {"generated_accuracy": spread([r[0]["accuracy"] for r in per_seed]),
                           "test_accuracy": spread([r[2]["accuracy"] for r in per_seed]),
                           "test_false_alarms": spread([r[2]["false_alarms_on_healthy"] for r in per_seed]),
                           **{f"{name}_accuracy": spread([r[4][name]["accuracy"] for r in per_seed])
                              for name in SUBSETS}},
            **final.size()}


def subsets(data: Dict[str, Any], predicted: Sequence[str]) -> Dict[str, Any]:
    """Test metrics on the old test runs and on the Colab test runs separately."""
    out = {}
    for name, splits in SUBSETS.items():
        keep = [i for i, split in enumerate(data["test_splits"]) if split in splits]
        out[name] = metrics([data["yt"][i] for i in keep], [predicted[i] for i in keep])
    return out


# ------------------------------------------------------------- the rules
def rules_label(code: str, max_steps: int = 150) -> str:
    """Top cause from this project's detectors (static + runtime + causal engine), offline; "clean" if none."""
    from silent_bug_agent.agents.diagnosis_agent import DiagnosisAgent
    from silent_bug_agent.agents.runtime_monitor_agent import MonitorConfig, RuntimeMonitorAgent
    from silent_bug_agent.schemas import AgentState
    from silent_bug_agent.static.analyzer import StaticAnalyzer

    with tempfile.TemporaryDirectory(prefix="sba_compare_") as folder:
        script = Path(folder) / "train.py"
        script.write_text(code, encoding="utf-8")
        state = AgentState(target_file=str(script), detection_report=StaticAnalyzer().analyze_file(script),
                           extra={"script_args": []})
        state = RuntimeMonitorAgent(MonitorConfig(max_steps=max_steps, cwd=folder))(state)
        state = DiagnosisAgent()(state)
    ranked = [h.bug_id for h in state.hypotheses if h.bug_id not in REPORT_ONLY]
    return ranked[0] if ranked else CLEAN


def evaluate_rules(rows: List[Dict[str, Any]], data: Dict[str, Any], workers: int,
                   earlier: Optional[List[str]], only_evaluation: bool = True) -> Dict[str, Any]:
    started = time.perf_counter()
    if earlier is not None and len(earlier) == len(rows):
        predictions, seconds = earlier, None
    elif only_evaluation:  # the training runs are many; the detectors are scored on dev and test only
        rows = [r for r in rows if r["split"] in DEV_SPLITS + TEST_SPLITS]
        with ThreadPoolExecutor(workers) as pool:
            predictions = list(pool.map(rules_label, [r["code"] for r in rows]))
        seconds = (time.perf_counter() - started) * workers / max(1, len(rows))
    else:
        with ThreadPoolExecutor(workers) as pool:
            predictions = list(pool.map(rules_label, [r["code"] for r in rows]))
        seconds = (time.perf_counter() - started) * workers / max(1, len(rows))
    by_split: Dict[str, List[str]] = {"train": [], "dev": [], "test": []}
    for row, label in zip(rows, predictions):
        key = "test" if row["split"] in TEST_SPLITS else ("dev" if row["split"] in DEV_SPLITS else "train")
        by_split[key].append(label)
    return {"model": "Hand-written rules (this project's detectors)", "kind": "rules, no training",
            "generated": metrics(data["y"], by_split["train"]) if len(by_split["train"]) == len(data["y"]) else {"n": 0},
            "dev": metrics(data["yd"], by_split["dev"]),
            "test": metrics(data["yt"], by_split["test"]), "test_predictions": by_split["test"],
            "subsets": subsets(data, by_split["test"]),
            "all_predictions": predictions, "seconds_per_prediction": seconds, "bytes": None, "parameters": None,
            "note": "each prediction includes a 150-step training run"}


# --------------------------------------------------------------- the LLMs
LABEL_HELP = {
    "clean": "no bug",
    "missing_zero_grad": "optimizer.zero_grad() is never called, so gradients accumulate",
    "train_eval_mode_mismatch": "model.eval() is missing before validation (dropout/batch norm stay in training mode)",
    "softmax_before_cross_entropy": "softmax is applied to the outputs before CrossEntropyLoss",
    "wrong_loss_reduction": "the loss uses reduction='sum' where 'mean' is expected",
    "learning_rate_misconfigured": "the learning rate is far too high",
    "frozen_layer_no_grad": "a layer's parameters never update",
    "dead_relu": "ReLU units are dead (e.g. a large negative bias)",
    "non_normalized_inputs": "the inputs are not normalised (raw pixel-like scale)",
    "label_data_misalignment": "the labels were shuffled separately from the inputs",
}


def llm_label(client: Any, payload: str, kind: str) -> str:
    catalog = "\n".join(f"- {label}: {text}" for label, text in LABEL_HELP.items())
    what = ("Statistics recorded during 60 optimizer steps of a PyTorch training run (feature: value):"
            if kind == "trace" else "A PyTorch training script:")
    prompt = (f"{what}\n\n{payload}\n\nWhich one of these describes the run? Exactly one label.\n{catalog}\n\n"
              'Answer as JSON: {"label": "<one label from the list>"}')
    schema = {"type": "object", "properties": {"label": {"type": "string", "enum": list(LABEL_HELP)}},
              "required": ["label"], "additionalProperties": False}
    try:
        reply = client.complete_json(prompt, "You diagnose silent bugs in deep learning training.",
                                     schema=schema, schema_name="bug_label")
        label = str(reply.get("label", "")) if isinstance(reply, dict) else ""
    except Exception:  # noqa: BLE001 - a failed call counts as a wrong answer, not a crash
        label = ""
    return label if label in LABEL_HELP else "unanswered"


def evaluate_llm(task_model: str, kind: str, payloads: List[str], truth: List[str]) -> Optional[Dict[str, Any]]:
    """The LLM is asked about the hand-written test scripts only (it needs no training data)."""
    from silent_bug_agent.semantic.router import PROVIDERS, ModelRouter, TaskClient, TaskModel

    router = ModelRouter()
    provider, model = task_model.split("/", 1)
    if provider not in PROVIDERS or not router.key_for(provider):
        return None
    client = TaskClient(router, "compare", [TaskModel(provider, model, "low" if "20b" in model else "medium", 400)])
    started = time.perf_counter()
    predicted = [llm_label(client, payload, kind) for payload in payloads]
    return {"model": f"{model} ({'trace statistics' if kind == 'trace' else 'source code'})",
            "kind": f"LLM via {provider}, no training", "generated": {"n": 0}, "dev": {"n": 0},
            "test": metrics(truth, predicted), "test_predictions": predicted,
            "seconds_per_prediction": (time.perf_counter() - started) / max(1, len(payloads)),
            "bytes": None, "parameters": None}


# ------------------------------------------------------------------ main
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="traces.jsonl from silent_bug_agent.mutation.dataset")
    parser.add_argument("--seeds", type=int, default=5, help="retrain every learned model this many times")
    parser.add_argument("--rules", action="store_true", help="also run the hand-written detectors (slow)")
    parser.add_argument("--reuse-rules", default=None, help="take the detectors' answers from an earlier --json")
    parser.add_argument("--llm", action="store_true", help="also ask gpt-oss-120b and gpt-oss-20b (needs a key)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--json", default=None)
    options = parser.parse_args(argv)

    rows = list(read(Path(options.data)))
    if any("code" not in r for r in rows):
        print("This data file has no source code per run; rebuild it with the current dataset builder.",
              file=sys.stderr)
        return 2
    train = [r for r in rows if r["split"] in TRAIN_SPLITS]
    dev = [r for r in rows if r["split"] in DEV_SPLITS]
    test = [r for r in rows if r["split"] in TEST_SPLITS]

    def matrix(chosen: List[Dict[str, Any]]) -> np.ndarray:
        return np.asarray([vectorize(r["report"], r.get("code")) for r in chosen]).reshape(len(chosen), len(FEATURES))

    data = {"X": matrix(train), "y": [r["label"] for r in train], "groups": [r["group"] for r in train],
            "Xd": matrix(dev), "yd": [r["label"] for r in dev], "Xt": matrix(test), "yt": [r["label"] for r in test],
            "test_splits": [r["split"] for r in test]}
    print(f"{len(train)} generated runs, {len(dev)} development runs, {len(test)} test runs "
          f"({sum(1 for r in test if r['label'] == CLEAN)} healthy)")

    results = []
    for name in FAMILIES:
        factory = factory_for(name)
        if factory is None:
            print(f"skipped {name}: scikit-learn is not installed")
            continue
        results.append(evaluate_family(name, factory, data, range(options.seeds)))
        print(f"{name}: settings {results[-1]['settings']}")
    if options.reuse_rules or options.rules:
        earlier = None
        if options.reuse_rules:
            previous = json.loads(Path(options.reuse_rules).read_text(encoding="utf-8"))["results"]
            earlier = next((r.get("all_predictions") for r in previous if r["model"].startswith("Hand-written")), None)
        results.append(evaluate_rules(rows, data, options.workers, earlier))
    if options.llm:
        traces = [json.dumps(dict(zip(FEATURES, np.round(row, 4).tolist()))) for row in data["Xt"]]
        for task_model in ("groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b"):
            for kind, payloads in (("trace", traces), ("code", [r["code"] for r in test])):
                result = evaluate_llm(task_model, kind, payloads, data["yt"])
                if result is None:
                    print(f"skipped {task_model}: no key for it in the environment or .env")
                    break
                results.append(result)

    print(f"\n{'model':44} {'generated':>10} {'hand-written test':>18} {'healthy flagged':>16} {'size':>10}")
    for r in results:
        seeds = r.get("over_seeds")
        gen = (f"{seeds['generated_accuracy']['mean']:.1%}" if seeds
               else (f"{r['generated']['accuracy']:.1%}" if r["generated"].get("n") else "-"))
        tst = f"{seeds['test_accuracy']['mean']:.1%}" if seeds else f"{r['test']['accuracy']:.1%}"
        size = f"{r['bytes'] / 1024:.1f} KB" if r.get("bytes") else "-"
        print(f"{r['model'][:44]:44} {gen:>10} {tst:>18} {r['test']['false_alarms_on_healthy']:16.1%} {size:>10}")
    if options.json:
        Path(options.json).write_text(json.dumps({
            "test_labels": data["yt"], "test_groups": [r["group"] for r in test], "test_splits": data["test_splits"],
            "counts": {"generated": len(train), "dev": len(dev), "test": len(test),
                       "test_healthy": sum(1 for r in test if r["label"] == CLEAN),
                       **{name: sum(1 for r in test if r["split"] in splits) for name, splits in SUBSETS.items()},
                       **{f"{name}_healthy": sum(1 for r in test if r["split"] in splits and r["label"] == CLEAN)
                          for name, splits in SUBSETS.items()}},
            "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
