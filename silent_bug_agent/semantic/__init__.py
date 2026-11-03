"""Module 2: semantic LLM reviewer, plus the LLM clients and model router every agent shares."""

from .analyzer import SemanticAnalyzer
from .client import (
    FakeLLMClient,
    LLMAuthError,
    LLMBadRequest,
    LLMBadResponse,
    LLMClient,
    LLMError,
    LLMRateLimited,
    LLMServerError,
    LLMUnavailable,
    OpenAICompatibleClient,
    extract_json,
)
from .prompt import SYSTEM_PROMPT, build_analysis_prompt, format_semantic_catalog
from .router import PROVIDERS, TASK_MODELS, ModelRouter, TaskClient, TaskModel

__all__ = [
    "SemanticAnalyzer", "LLMClient", "FakeLLMClient", "OpenAICompatibleClient", "extract_json",
    "LLMError", "LLMAuthError", "LLMRateLimited", "LLMBadRequest", "LLMBadResponse", "LLMServerError",
    "LLMUnavailable", "ModelRouter", "TaskClient", "TaskModel", "PROVIDERS", "TASK_MODELS",
    "SYSTEM_PROMPT", "build_analysis_prompt", "format_semantic_catalog",
]
