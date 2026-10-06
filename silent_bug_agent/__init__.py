"""silent_bug_agent — Silent bug detection for deep learning training.

Phase 0: bug taxonomy and invariant library
Phase 2: AST static analyzer and LLM semantic reviewer
"""

from .findings import Finding, Report, merge_reports, parse_severity
from .invariants.base import Invariant, InvariantResult, Severity, TrainingState
from .invariants.registry import get_default_registry, run_all
from .semantic import AnthropicClient, FakeLLMClient, LLMClient, SemanticAnalyzer
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
    "AnthropicClient",
]

