"""silent_bug_agent: detect, diagnose and fix silent bugs in deep learning training."""

from .findings import Finding, Report, merge_reports, parse_severity
from .invariants.base import Invariant, InvariantResult, Severity, TrainingState
from .invariants.registry import get_default_registry, run_all
from .semantic import FakeLLMClient, LLMClient, ModelRouter, OpenAICompatibleClient, SemanticAnalyzer
from .static import BaseRule, StaticAnalyzer

__all__ = [
    "Invariant",
    "InvariantResult",
    "Severity",
    "TrainingState",
    "get_default_registry",
    "run_all",
    "Finding",
    "Report",
    "merge_reports",
    "parse_severity",
    "StaticAnalyzer",
    "BaseRule",
    "SemanticAnalyzer",
    "LLMClient",
    "FakeLLMClient",
    "OpenAICompatibleClient",
    "ModelRouter",
]

