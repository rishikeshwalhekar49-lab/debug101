"""Compare the trace classifier with other ways of naming the bug in a training run.

    python -m silent_bug_agent.mutation.dataset --templates 80 --out traces.jsonl
    python examples/compare_models.py --data traces.jsonl --json comparison.json --rules

Every model answers the same question for the same runs: which of nine planted bugs,
or "clean". Two test sets:

* held-out templates: 5-fold cross-validation over generated scripts, holding out
  whole templates (models that need no training are simply run on every script);
* hand-written benchmarks: blind copies of the project's benchmarks, never trained on.

Models:

* majority class: always the most common label (the floor);
* hand-written rules (`--rules`): this project's static rules + runtime monitor +
  causal engine, offline, exactly as `diagnose --no-llm` runs them (slow: it trains
  every script once);
* the trace classifier with no hidden layer (softmax regression) and as shipped
  (one hidden layer of 32);
* scikit-learn models on the same 24 features (needs `pip install scikit-learn`):
  random forest, gradient boosting, k-nearest neighbours, RBF SVM;
* LLMs (`--llm`, needs GROQ_API_KEY or CEREBRAS_API_KEY): gpt-oss-120b and
  gpt-oss-20b, given the same trace statistics as JSON, and separately given the
  blind source code.
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
from silent_bug_agent.learned.train import _folds  # noqa: E402
from silent_bug_agent.mutation.dataset import benchmark_jobs, read, template_jobs  # noqa: E402

CLEAN = "clean"
REPORT_ONLY = {"unset_random_seed"}


# ----------------------------------------------------------------- metrics
def metrics(truth: Sequence[str], predicted: Sequence[str]) -> Dict[str, Any]:
    truth, predicted = list(truth), list(predicted)
    healthy = [p for t, p in zip(truth, predicted) if t == CLEAN]
    buggy = [(t, p) for t, p in zip(truth, predicted) if t != CLEAN]
    classes = sorted(set(truth))
    recall = {c: float(np.mean([p == c for t, p in zip(truth, predicted) if t == c])) for c in classes}
    return {"n": len(truth), "accuracy": float(np.mean([t == p for t, p in zip(truth, predicted)])),
            "macro_recall": float(np.mean(list(recall.values()))),
            "bugs_named_correctly": float(np.mean([t == p for t, p in buggy])) if buggy else None,
            "false_alarms_on_healthy": float(np.mean([p != CLEAN for p in healthy])) if healthy else None,
            "recall": recall}


# ------------------------------------------------------- feature models
def _ours(hidden: int) -> Callable[[], Any]:
    class Ours:
        def fit(self, X: np.ndarray, y: List[str]) -> "Ours":
            self.model = TraceClassifier.fit(X, y, hidden=hidden)
            return self

        def predict(self, X: np.ndarray) -> List[str]:
            return [self.model.classes[i] for i in self.model.probabilities(X).argmax(axis=1)]

        def size(self) -> Dict[str, Any]:
            blob = json.dumps(self.model.to_dict(), separators=(",", ":")).encode()
            params = sum(W.size + b.size for W, b in self.model.layers)
            return {"bytes": len(blob), "parameters": int(params)}

    return Ours


class Majority:
    def fit(self, X: np.ndarray, y: List[str]) -> "Majority":
        labels, counts = np.unique(y, return_counts=True)
        self.label = str(labels[int(np.argmax(counts))])
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        return [self.label] * len(X)

    def size(self) -> Dict[str, Any]:
        return {"bytes": len(self.label), "parameters": 0}


def _sklearn(name: str) -> Optional[Callable[[], Any]]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        return None
    builders = {
        "Random forest (300 trees)": lambda: RandomForestClassifier(n_estimators=300, random_state=0,
                                                                     class_weight="balanced"),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(random_state=0, class_weight="balanced"),
        "k-nearest neighbours (k=5)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "SVM (RBF kernel)": lambda: make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=0)),
    }

    class Wrapped:
        def fit(self, X: np.ndarray, y: List[str]) -> "Wrapped":
            self.model = builders[name]().fit(X, y)
            return self

        def predict(self, X: np.ndarray) -> List[str]:
            return [str(p) for p in self.model.predict(X)]

        def size(self) -> Dict[str, Any]:
            return {"bytes": len(pickle.dumps(self.model)), "parameters": None}

    return Wrapped


def evaluate_feature_model(name: str, factory: Callable[[], Any], X: np.ndarray, y: List[str], groups: List[str],
                           Xb: np.ndarray, yb: List[str]) -> Dict[str, Any]:
    predicted: List[str] = [""] * len(y)
    for test in _folds(groups, 5, 0):
        model = factory().fit(X[~test], [label for label, t in zip(y, test) if not t])
        for index, label in zip(np.flatnonzero(test), model.predict(X[test])):
            predicted[index] = label
    final = factory().fit(X, y)
    started = time.perf_counter()
    for row in Xb:
        final.predict(row[None, :])
    per_prediction = (time.perf_counter() - started) / max(1, len(Xb))
    return {"model": name, "kind": "trained on trace features", "templates": metrics(y, predicted),
            "benchmarks": metrics(yb, final.predict(Xb)), "benchmark_predictions": final.predict(Xb),
            "seconds_per_prediction": per_prediction, **final.size()}


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


def evaluate_rules(codes: List[str], y: List[str], codes_b: List[str], yb: List[str], workers: int) -> Dict[str, Any]:
    started = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        predicted = list(pool.map(rules_label, codes + codes_b))
    seconds = (time.perf_counter() - started) * workers / max(1, len(predicted))
    return {"model": "Hand-written rules (this project's detectors)", "kind": "rules, no training",
            "templates": metrics(y, predicted[:len(y)]), "benchmarks": metrics(yb, predicted[len(y):]),
            "benchmark_predictions": predicted[len(y):], "seconds_per_prediction": seconds,
            "bytes": None, "parameters": None, "note": "each prediction includes a 150-step training run"}


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


def evaluate_llm(task_model: str, kind: str, payloads: List[str], y: List[str], payloads_b: List[str],
                 yb: List[str], limit: Optional[int]) -> Optional[Dict[str, Any]]:
    from silent_bug_agent.semantic.router import PROVIDERS, ModelRouter, TaskClient, TaskModel

    router = ModelRouter()
    provider, model = task_model.split("/", 1)
    if provider not in PROVIDERS or not router.key_for(provider):
        return None
    client = TaskClient(router, "compare", [TaskModel(provider, model, "low" if "20b" in model else "medium", 400)])
    pick = list(range(len(y)))[:limit] if limit else list(range(len(y)))
    started = time.perf_counter()
    predicted = [llm_label(client, payloads[i], kind) for i in pick]
    predicted_b = [llm_label(client, p, kind) for p in payloads_b]
    calls = len(pick) + len(payloads_b)
    return {"model": f"{model} ({'trace statistics' if kind == 'trace' else 'source code'})",
            "kind": f"LLM via {provider}, no training", "templates": metrics([y[i] for i in pick], predicted),
            "benchmarks": metrics(yb, predicted_b), "benchmark_predictions": predicted_b,
            "seconds_per_prediction": (time.perf_counter() - started) / max(1, calls),
            "bytes": None, "parameters": None}


# ------------------------------------------------------------------ main
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="traces.jsonl from silent_bug_agent.mutation.dataset")
    parser.add_argument("--templates", type=int, default=80, help="the --templates the data was built with")
    parser.add_argument("--seed", type=int, default=0, help="the --seed the data was built with")
    parser.add_argument("--rules", action="store_true", help="also run the hand-written detectors (slow)")
    parser.add_argument("--llm", action="store_true", help="also ask gpt-oss-120b and gpt-oss-20b (needs a key)")
    parser.add_argument("--llm-limit", type=int, default=150, help="template runs to send to each LLM")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--json", default=None)
    options = parser.parse_args(argv)

    rows = list(read(Path(options.data)))
    jobs = template_jobs(options.templates, options.seed) + benchmark_jobs()
    if len(jobs) != len(rows) or any(r["group"] != j["group"] or r["label"] != j["label"] for r, j in zip(rows, jobs)):
        print("The scripts regenerated from --templates/--seed do not match the data file.", file=sys.stderr)
        return 2
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    bench = [i for i, r in enumerate(rows) if r["split"] == "benchmark"]
    X = np.asarray([vectorize(rows[i]["report"]) for i in train])
    Xb = np.asarray([vectorize(rows[i]["report"]) for i in bench])
    y, yb = [rows[i]["label"] for i in train], [rows[i]["label"] for i in bench]
    groups = [rows[i]["group"] for i in train]

    factories: Dict[str, Optional[Callable[[], Any]]] = {
        "Majority class": lambda: Majority(),
        "Softmax regression (ours, no hidden layer)": _ours(0),
        "Trace classifier (ours, as shipped)": _ours(32),
    }
    for name in ("Random forest (300 trees)", "Gradient boosting", "k-nearest neighbours (k=5)", "SVM (RBF kernel)"):
        factories[name] = _sklearn(name)
    results = []
    for name, factory in factories.items():
        if factory is None:
            print(f"skipped {name}: scikit-learn is not installed")
            continue
        results.append(evaluate_feature_model(name, factory, X, y, groups, Xb, yb))
        print(f"{name}: done")
    if options.rules:
        results.append(evaluate_rules([jobs[i]["code"] for i in train], y, [jobs[i]["code"] for i in bench], yb,
                                      options.workers))
        print("hand-written rules: done")
    if options.llm:
        traces = [json.dumps(dict(zip(FEATURES, np.round(row, 4).tolist()))) for row in X]
        traces_b = [json.dumps(dict(zip(FEATURES, np.round(row, 4).tolist()))) for row in Xb]
        for task_model in ("groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b"):
            for kind, payloads, payloads_b in (("trace", traces, traces_b),
                                               ("code", [jobs[i]["code"] for i in train],
                                                [jobs[i]["code"] for i in bench])):
                result = evaluate_llm(task_model, kind, payloads, y, payloads_b, yb, options.llm_limit)
                if result is None:
                    print(f"skipped {task_model}: no key for it in the environment or .env")
                    break
                results.append(result)
                print(f"{result['model']}: done")

    print(f"\n{'model':48} {'templates':>9} {'benchmarks':>10} {'healthy flagged':>15} {'size':>10} {'per prediction':>15}")
    for r in results:
        size = f"{r['bytes'] / 1024:.1f} KB" if r.get("bytes") else "-"
        alarms = r["benchmarks"]["false_alarms_on_healthy"]
        print(f"{r['model'][:48]:48} {r['templates']['accuracy']:9.1%} {r['benchmarks']['accuracy']:10.1%} "
              f"{alarms:15.1%} {size:>10} {r['seconds_per_prediction'] * 1e6:13.0f}us")
    if options.json:
        Path(options.json).write_text(json.dumps({"benchmark_labels": yb, "benchmark_groups":
                                                  [rows[i]["group"] for i in bench], "results": results},
                                                 indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
