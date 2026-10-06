"""Pluggable LLM client interface for the semantic analyzer."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClient(ABC):
    """Abstract interface for LLM calls."""

    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompt to LLM and return raw response string."""
        pass


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


class AnthropicClient(LLMClient):
    """Anthropic API client with configurable model and API key."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for AnthropicClient. "
                    "Install it via `pip install anthropic`."
                )
            if not self.api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable or explicit api_key must be set."
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        client = self._get_client()
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        # Extract text from content blocks
        text_parts = [
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ]
        return "\n".join(text_parts)
