"""The trace classifier against public open-weight LLMs, on the same test runs.

    python examples/compare_llms.py --data traces_v4_150.jsonl --json docs/images/llm_comparison.json
    python examples/plot_llms.py docs/images/llm_comparison.json docs/images/llm_comparison.png

Each LLM answers the same question as the classifier: which of nine silent bugs, or
"clean". It is asked twice per run:

* "code": the training script itself, which is what an LLM-based debugger reads;
* "code + run statistics": the script and the numbers the classifier reads from the
  same traced run, named (so the LLM has everything the classifier has).

The models are the public open-weight chat models Groq serves (GROQ_API_KEY, from the
environment or `.env`, as everywhere else in the project). Only Groq is used here; the
project's Cerebras support is not involved.

Before a script is sent, the local path of Colab's sample data is put back to
`/content/sample_data`, so nothing about this machine leaves it. Every answer is
cached in --cache, so a run stopped by a free-tier limit continues where it stopped.
A model is only charted once it has answered every run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silent_bug_agent.learned.features import FEATURES, vectorize  # noqa: E402
from silent_bug_agent.learned.model import load_default  # noqa: E402
from silent_bug_agent.mutation.dataset import read  # noqa: E402

CLEAN = "clean"
# provider, model id, name shown, total parameters, active parameters per token. Counts: OpenAI's published
# figures for gpt-oss; the size in the model's name for the others. Providers retire models often: the list is
# the open-weight chat models Groq served on 25 Nov 2026 (`--list-models` prints today's).
MODELS: List[Tuple[str, str, str, float, Optional[float]]] = [
    ("groq", "openai/gpt-oss-120b", "gpt-oss-120b", 117e9, 5.1e9),
    ("groq", "openai/gpt-oss-20b", "gpt-oss-20b", 21e9, 3.6e9),
    ("groq", "qwen/qwen3.8-27b", "Qwen 3.8 27B", 27e9, None),
    ("groq", "allam-2-7b", "ALLaM 2 7B", 7e9, None),
]
SMALL_CONTEXT = {"allam-2-7b"}  # a 4,096-token window: short replies, and no reasoning settings
LABEL_HELP = {
    "clean": "no bug: the script trains as intended",
    "missing_zero_grad": "gradients are never cleared between updates (no optimizer.zero_grad() or equivalent), so they accumulate",
    "train_eval_mode_mismatch": "the model has dropout or batch norm and is left in training mode for validation (model.eval() missing)",
    "softmax_before_cross_entropy": "softmax is applied to the outputs before a cross-entropy loss that expects logits",
    "wrong_loss_reduction": "the loss sums over the batch (reduction='sum') where the step size was set for a mean",
    "learning_rate_misconfigured": "the learning rate is far too high or far too low for this optimizer",
    "frozen_layer_no_grad": "some layers never update (requires_grad=False, or left out of the optimizer)",
    "dead_relu": "ReLU units are dead from the start (e.g. a large negative bias)",
    "non_normalized_inputs": "the inputs are not scaled (e.g. raw 0-255 pixels, or features in the thousands)",
    "label_data_misalignment": "labels no longer match their inputs (shuffled, shifted or resampled separately)",
}
SYSTEM = "You diagnose silent bugs in PyTorch training scripts. You answer with one label only."
SPLITS = {"Colab test": ("colab_test",), "old test": ("holdout_test", "benchmark")}


def sanitise(code: str) -> str:
    """The script as it was written in Colab: no local paths."""
    return re.sub(r'SAMPLE_DATA = "[^"]*"', 'SAMPLE_DATA = "/content/sample_data"', code)


def statistics(row: Dict[str, Any]) -> str:
    values = dict(zip(FEATURES, vectorize(row["report"], row.get("code"))))
    return "\n".join(f"{name}: {value:.4g}" for name, value in values.items())


def prompt_for(row: Dict[str, Any], kind: str) -> str:
    catalog = "\n".join(f"- {label}: {text}" for label, text in LABEL_HELP.items())
    parts = [f"A PyTorch training script:\n\n```python\n{sanitise(row['code'])}\n```"]
    if kind == "code+stats":
        parts.append("Statistics measured while it trained for 150 steps (names describe each number; ratios are "
                     "scale-free, logs are base 10, 'code_' ones are read from the script):\n" + statistics(row))
    parts.append(f"Which one of these describes it? Exactly one label.\n{catalog}\n\n"
                 'Answer as JSON: {"label": "<one label from the list>"}')
    return "\n\n".join(parts)


SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": list(LABEL_HELP)}},
          "required": ["label"], "additionalProperties": False}


class Cache:
    def __init__(self, path: Path) -> None:
        self.path, self.lock, self.answers = path, threading.Lock(), {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    self.answers[(item["model"], item["kind"], item["key"])] = item

    def get(self, model: str, kind: str, key: str) -> Optional[Dict[str, Any]]:
        return self.answers.get((model, kind, key))

    def put(self, item: Dict[str, Any]) -> None:
        with self.lock:
            self.answers[(item["model"], item["kind"], item["key"])] = item
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item) + "\n")


def ask(client: Any, row: Dict[str, Any], kind: str, waits: int = 6) -> Tuple[Optional[str], float, str]:
    """(label or None, seconds, error) for one run; waits out free-tier limits a few times."""
    from silent_bug_agent.semantic.client import LLMAuthError, LLMBadRequest, LLMError, LLMRateLimited

    prompt = prompt_for(row, kind)
    for attempt in range(waits + 1):
        started = time.perf_counter()
        try:
            reply = client.complete_json(prompt, SYSTEM, schema=SCHEMA, schema_name="bug_label")
            label = str(reply.get("label", "")) if isinstance(reply, dict) else ""
            return (label if label in LABEL_HELP else "unanswerable"), time.perf_counter() - started, ""
        except (LLMAuthError, LLMBadRequest) as exc:
            return None, 0.0, str(exc)
        except LLMRateLimited as exc:
            if attempt == waits:
                return None, 0.0, str(exc)
            time.sleep(30 * (attempt + 1))
        except LLMError as exc:
            if attempt == waits:
                return None, 0.0, str(exc)
            time.sleep(5)
    return None, 0.0, "gave up"


def list_models(router: Any, provider: str = "groq") -> int:
    """The chat models the provider serves today (providers retire models often)."""
    import httpx

    from silent_bug_agent.semantic.router import PROVIDERS

    key = router.key_for(provider)
    if key is None:
        print(f"no key for {provider}", file=sys.stderr)
        return 2
    response = httpx.get(f"{PROVIDERS[provider].base_url}/models", headers={"Authorization": f"Bearer {key}"},
                         timeout=30)
    if response.status_code != 200:
        print(f"{provider}: HTTP {response.status_code}", file=sys.stderr)
        return 1
    for model in sorted(response.json().get("data", []), key=lambda m: m.get("id", "")):
        print(f"{model.get('id')}  (owned by {model.get('owned_by')}, context {model.get('context_window')})")
    return 0


def metrics(truth: Sequence[str], predicted: Sequence[str]) -> Dict[str, Any]:
    healthy = [p for t, p in zip(truth, predicted) if t == CLEAN]
    return {"n": len(truth), "correct": int(sum(t == p for t, p in zip(truth, predicted))),
            "accuracy": float(np.mean([t == p for t, p in zip(truth, predicted)])) if truth else None,
            "healthy": len(healthy), "healthy_flagged": int(sum(p != CLEAN for p in healthy))}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="the v4 dataset (150-step traces)")
    parser.add_argument("--sets", nargs="+", default=["Colab test"], choices=list(SPLITS))
    parser.add_argument("--kinds", nargs="+", default=["code", "code+stats"], choices=["code", "code+stats"])
    parser.add_argument("--models", nargs="+", default=None, help="model ids to ask (default: all in MODELS)")
    parser.add_argument("--cache", default=str(REPO_ROOT / ".llm_answers.jsonl"))
    parser.add_argument("--json", default=None)
    parser.add_argument("--list-models", action="store_true", help="print the models the provider serves now, and stop")
    options = parser.parse_args(argv)

    from silent_bug_agent.semantic.router import ModelRouter

    router = ModelRouter()
    if options.list_models:
        return list_models(router)
    rows = [r for r in read(Path(options.data)) if any(r["split"] in SPLITS[s] for s in options.sets)]
    for index, row in enumerate(rows):
        row["key"] = f"{row['split']}:{row['group']}:{row['label']}:{index}"
    ours = load_default()
    if ours is None:
        raise SystemExit("no trained classifier in silent_bug_agent/learned/weights.json")
    cache = Cache(Path(options.cache))
    models = [m for m in MODELS if options.models is None or m[1] in options.models]
    keyed = [m for m in models if router.key_for(m[0])]
    if not keyed:
        print("No LLM key is set (GROQ_API_KEY or CEREBRAS_API_KEY in the environment or .env), so no LLM can be"
              " asked. Add a free Groq key to .env and run this again.", file=sys.stderr)
        return 2

    def run(model: Tuple[str, str, str, float, Optional[float]]) -> None:
        provider, model_id, name, _, _ = model
        from silent_bug_agent.semantic.router import TaskModel

        small_context = model_id in SMALL_CONTEXT
        client = router.client_for(TaskModel(provider, model_id, "low" if "gpt-oss" in model_id else None,
                                             256 if small_context else 1024))
        if small_context:
            client.extra_body = {}  # no reasoning to hide, and the provider rejects the setting for this model
        for kind in options.kinds:
            for row in rows:
                if cache.get(model_id, kind, row["key"]):
                    continue
                label, seconds, error = ask(client, row, kind)
                if label is None:
                    print(f"{name} ({kind}): stopped at {row['key']}: {error[:160]}", flush=True)
                    if "rejected" in error or "HTTP 4" in error:
                        return  # a bad key or a model the provider does not serve: no point in going on
                    break
                cache.put({"model": model_id, "kind": kind, "key": row["key"], "label": label,
                           "seconds": round(seconds, 2)})
            done = sum(1 for row in rows if cache.get(model_id, kind, row["key"]))
            print(f"{name} ({kind}): {done}/{len(rows)} answered", flush=True)

    with ThreadPoolExecutor(len(keyed)) as pool:  # one thread per model: free-tier limits are per model
        list(pool.map(run, keyed))

    result: Dict[str, Any] = {"sets": options.sets, "results": []}
    for set_name in options.sets:
        chosen = [r for r in rows if r["split"] in SPLITS[set_name]]
        truth = [r["label"] for r in chosen]
        predicted = [ours.classes[int(ours.probabilities(np.asarray([vectorize(r["report"], r.get("code"))])).argmax())]
                     for r in chosen]
        result["results"].append({"set": set_name, "model": "Trace classifier v4 (ours)", "kind": "trained",
                                  "parameters": int(sum(W.size + b.size for W, b in ours.layers)),
                                  **metrics(truth, predicted)})
        for provider, model_id, name, params, active in models:
            for kind in options.kinds:
                answers = [cache.get(model_id, kind, r["key"]) for r in chosen]
                complete = all(answers)
                entry = {"set": set_name, "model": name, "model_id": model_id, "provider": provider, "kind": kind,
                         "parameters": params, "active_parameters": active, "complete": complete,
                         "answered": sum(1 for a in answers if a)}
                if complete:
                    labels = [a["label"] for a in answers]
                    entry.update(metrics(truth, labels))
                    entry["median_seconds"] = float(np.median([a["seconds"] for a in answers]))
                result["results"].append(entry)
    for entry in result["results"]:
        if entry.get("accuracy") is not None:
            print(f"{entry['set']:10} {entry['model']:26} {entry.get('kind', ''):11} "
                  f"{entry['correct']}/{entry['n']} = {entry['accuracy']:.1%}")
    if options.json:
        Path(options.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
