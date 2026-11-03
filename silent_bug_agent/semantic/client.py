"""LLM clients: a real OpenAI-compatible HTTP client and a fake one for tests.

Groq and Cerebras both speak the OpenAI chat-completions protocol, so one client
serves both. Keys are read from the environment by the router and are never
logged or placed in an error message.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    """Base class for every failed LLM call."""


class LLMAuthError(LLMError):
    """The key was rejected."""


class LLMRateLimited(LLMError):
    """The free-tier limit was reached."""


class LLMBadRequest(LLMError):
    """The provider refused the request (bad parameter, model gone, prompt too large)."""


class LLMServerError(LLMError):
    """The provider failed or could not be reached."""


class LLMBadResponse(LLMError):
    """The reply was empty or was not the JSON that was asked for."""


class LLMUnavailable(LLMError):
    """No configured provider could answer."""


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating markdown fences and surrounding prose."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for chunk in candidates:
        try:
            return json.loads(chunk)
        except (TypeError, ValueError):
            continue
    raise LLMBadResponse("the reply was not valid JSON")


class LLMClient(ABC):
    """Abstract interface for LLM calls."""

    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompt to LLM and return raw response string."""

    def complete_json(self, prompt: str, system_prompt: Optional[str] = None, *,
                      schema: Optional[Dict[str, Any]] = None, schema_name: str = "result") -> Any:
        """Send prompt and return the parsed JSON reply."""
        return extract_json(self.complete(prompt, system_prompt))


class FakeLLMClient(LLMClient):
    """Deterministic client for testing without network or API keys."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        canned_findings: Optional[List[Dict[str, Any]]] = None,
    ):
        self.responses = list(responses) if responses is not None else []
        self.call_history: List[Dict[str, Optional[str]]] = []
        if canned_findings is not None:
            self.responses.append(json.dumps(canned_findings))

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.call_history.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.responses:
            return self.responses.pop(0)
        return "[]"


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class OpenAICompatibleClient(LLMClient):
    """Chat completions over HTTP for any OpenAI-compatible provider."""

    def __init__(self, base_url: str, api_key: str, model: str, *, provider: str = "openai-compatible",
                 max_tokens: int = 2048, temperature: float = 0.2, reasoning_effort: Optional[str] = None,
                 timeout: float = 90.0, max_retries: int = 2, max_wait: float = 20.0,
                 extra_body: Optional[Dict[str, Any]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.extra_body = dict(extra_body or {})
        self.last: Optional[ChatResult] = None

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatibleClient(provider={self.provider!r}, model={self.model!r})"

    def _payload(self, prompt: str, system_prompt: Optional[str],
                 response_format: Optional[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature,
                                   "max_completion_tokens": max_tokens, **self.extra_body}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if response_format:
            payload["response_format"] = response_format
        return payload

    def _post(self, payload: Dict[str, Any]) -> ChatResult:
        import httpx

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        attempt, started = 0, time.perf_counter()
        while True:
            attempt += 1
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
            except httpx.HTTPError as exc:
                if attempt <= self.max_retries:
                    time.sleep(min(2.0 * attempt, self.max_wait))
                    continue
                raise LLMServerError(f"{self.provider}: could not reach the service ({type(exc).__name__})") from None
            status = response.status_code
            if status == 200:
                return self._parse(response, started)
            detail = self._error_detail(response)
            if status in (401, 403):
                raise LLMAuthError(f"{self.provider}: the API key was rejected (HTTP {status})")
            if status == 429:
                wait = self._retry_after(response)
                if attempt <= self.max_retries and wait is not None and wait <= self.max_wait:
                    time.sleep(wait)
                    continue
                raise LLMRateLimited(f"{self.provider}: free-tier limit reached for {self.model}: {detail}")
            if status >= 500:
                if attempt <= self.max_retries:
                    time.sleep(min(2.0 * attempt, self.max_wait))
                    continue
                raise LLMServerError(f"{self.provider}: server error HTTP {status}: {detail}")
            raise LLMBadRequest(f"{self.provider}: HTTP {status} for {self.model}: {detail}")

    @staticmethod
    def _retry_after(response: Any) -> Optional[float]:
        raw = response.headers.get("retry-after")
        try:
            return max(0.0, float(raw)) if raw is not None else 2.0
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _error_detail(response: Any) -> str:
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else body
            message = error.get("message", error) if isinstance(error, dict) else error
            return str(message)[:300]
        except ValueError:
            return response.text[:300]

    def _parse(self, response: Any, started: float) -> ChatResult:
        try:
            body = response.json()
            choice = body["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMBadResponse(f"{self.provider}: unexpected reply format ({type(exc).__name__})") from None
        if not text.strip():
            raise LLMBadResponse(f"{self.provider}: empty reply (finish_reason={choice.get('finish_reason')})")
        usage = body.get("usage") or {}
        result = ChatResult(text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0),
                            time.perf_counter() - started)
        self.last = result
        return result

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            return self._post(self._payload(prompt, system_prompt, None, self.max_tokens)).text
        except LLMBadResponse:  # reasoning used up the budget: try once with more room
            return self._post(self._payload(prompt, system_prompt, None, self.max_tokens * 2)).text

    def complete_json(self, prompt: str, system_prompt: Optional[str] = None, *,
                      schema: Optional[Dict[str, Any]] = None, schema_name: str = "result") -> Any:
        """Strict schema first, then plain JSON mode, because providers differ in what they accept."""
        formats: List[Optional[Dict[str, Any]]] = []
        if schema is not None:
            formats.append({"type": "json_schema",
                            "json_schema": {"name": schema_name, "strict": True, "schema": schema}})
        formats.append({"type": "json_object"})
        failure: Optional[LLMError] = None
        for response_format in formats:
            try:
                result = self._post(self._payload(prompt, system_prompt, response_format, self.max_tokens * 2))
                return extract_json(result.text)
            except (LLMBadRequest, LLMBadResponse) as exc:
                failure = exc
        raise failure or LLMBadResponse("no reply")
