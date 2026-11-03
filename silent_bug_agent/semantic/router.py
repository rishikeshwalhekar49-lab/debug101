"""Picks the model for each task and falls over between free providers.

Each task has an ordered list of (provider, model). A call tries them in turn, so
reaching Groq's free limit moves to Cerebras. If every provider fails the caller
gets `LLMUnavailable` and must say so: nothing silently pretends to be an LLM.

Override a choice in `.env` with `SBA_MODEL_<TASK>=provider/model`, for example
`SBA_MODEL_FIX=cerebras/gpt-oss-120b`.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..config import load_env
from .client import LLMClient, LLMError, LLMUnavailable, OpenAICompatibleClient


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    url_env: str
    extra_body: Any = ()


PROVIDERS: Dict[str, Provider] = {
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_BASE_URL",
                     extra_body=(("include_reasoning", False),)),
    "cerebras": Provider("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "CEREBRAS_BASE_URL"),
}


@dataclass(frozen=True)
class TaskModel:
    provider: str
    model: str
    reasoning_effort: Optional[str] = None
    max_tokens: int = 2048


_BIG = (TaskModel("groq", "openai/gpt-oss-120b", "medium", 3000), TaskModel("cerebras", "gpt-oss-120b", "medium", 3000))
_SMALL = (TaskModel("groq", "openai/gpt-oss-20b", "low", 1024), TaskModel("cerebras", "qwen-3.8-27b", None, 1024))

TASK_MODELS: Dict[str, Sequence[TaskModel]] = {
    "semantic_review": _BIG,  # Agent 2
    "diagnosis": _BIG,  # Agent 4
    "fix": _BIG,  # Agent 5
    "meta": _BIG,  # Agent 7
    "perception": _SMALL,  # Agent 1: consolidating alerts
    "validation": _SMALL,  # Agent 3: reading the statistics
}


def _override(task: str) -> Optional[TaskModel]:
    value = os.environ.get(f"SBA_MODEL_{task.upper()}", "").strip()
    if "/" not in value:
        return None
    provider, model = value.split("/", 1)
    return TaskModel(provider, model) if provider in PROVIDERS else None


class TaskClient(LLMClient):
    """An LLMClient for one task: tries its models in order and records every call."""

    def __init__(self, router: "ModelRouter", task: str, choices: Sequence[TaskModel]) -> None:
        self.router = router
        self.task = task
        self.choices = list(choices)

    def _attempt(self, method: str, *args: Any, **kwargs: Any) -> Any:
        failures: List[str] = []
        for choice in self.choices:
            client = self.router.client_for(choice)
            if client is None:
                continue
            try:
                result = getattr(client, method)(*args, **kwargs)
            except LLMError as exc:
                failures.append(str(exc))
                self.router.record(self.task, choice, None, str(exc))
                continue
            self.router.record(self.task, choice, client.last, None)
            return result
        raise LLMUnavailable("; ".join(failures) or "no API key is set")

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return self._attempt("complete", prompt, system_prompt)

    def complete_json(self, prompt: str, system_prompt: Optional[str] = None, *,
                      schema: Optional[Dict[str, Any]] = None, schema_name: str = "result") -> Any:
        return self._attempt("complete_json", prompt, system_prompt, schema=schema, schema_name=schema_name)


class ModelRouter:
    """Builds per-task clients from the keys in the environment."""

    def __init__(self, *, offline: bool = False) -> None:
        load_env()
        self.offline = offline
        self.calls: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def key_for(self, provider: str) -> Optional[str]:
        return os.environ.get(PROVIDERS[provider].key_env, "").strip() or None

    def client_for(self, choice: TaskModel) -> Optional[OpenAICompatibleClient]:
        key = self.key_for(choice.provider)
        if key is None or self.offline:
            return None
        provider = PROVIDERS[choice.provider]
        return OpenAICompatibleClient(os.environ.get(provider.url_env) or provider.base_url, key, choice.model,
                                      provider=provider.name, max_tokens=choice.max_tokens,
                                      reasoning_effort=choice.reasoning_effort, extra_body=dict(provider.extra_body))

    def choices(self, task: str) -> List[TaskModel]:
        override = _override(task)
        chain = ([override] if override else []) + list(TASK_MODELS[task])
        return [c for c in chain if self.key_for(c.provider) and not self.offline]

    def available(self) -> bool:
        return not self.offline and any(self.key_for(p) for p in PROVIDERS)

    def task(self, task: str) -> Optional[TaskClient]:
        """The client for a task, or None when no key is set (the caller then uses its offline path)."""
        choices = self.choices(task)
        return TaskClient(self, task, choices) if choices else None

    def record(self, task: str, choice: TaskModel, result: Any, error: Optional[str]) -> None:
        with self._lock:
            self.calls.append({"task": task, "provider": choice.provider, "model": choice.model, "ok": error is None,
                               "prompt_tokens": getattr(result, "prompt_tokens", 0),
                               "completion_tokens": getattr(result, "completion_tokens", 0),
                               "seconds": round(getattr(result, "seconds", 0.0), 2), "error": error})

    def summary(self) -> Dict[str, Any]:
        ok = [c for c in self.calls if c["ok"]]
        models = sorted({f"{c['provider']}/{c['model']}" for c in ok})
        return {"mode": "live" if self.available() else "offline", "calls": len(ok), "failed_calls": len(self.calls) - len(ok),
                "tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for c in ok), "models_used": models}

    def describe(self) -> str:
        if not self.available():
            return "offline (no LLM key set, so only rules and statistics are running)"
        names = [p for p in PROVIDERS if self.key_for(p)]
        return "live: " + " then ".join(names)
