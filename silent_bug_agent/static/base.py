"""Base class and common utilities for AST-based static analysis rules."""

import ast
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..findings import Finding
from ..invariants.base import Severity
from ..taxonomy.loader import BugPattern


def get_source_line(source_lines: List[str], lineno: int) -> str:
    """Retrieve line of source code (1-indexed), safely returning empty string if out of range."""
    if 1 <= lineno <= len(source_lines):
        return source_lines[lineno - 1]
    return ""


class BaseRule(ABC):
    """Abstract base rule for AST inspection.

    Attributes:
        bug_id: The taxonomy id corresponding to this rule.
        default_severity: Severity used if not overridden.
        default_confidence: Confidence score in [0.0, 1.0].
    """

    bug_id: str
    default_severity: Severity = Severity.WARNING
    default_confidence: float = 0.85

    @abstractmethod
    def check(
        self,
        tree: ast.AST,
        source_lines: List[str],
        file_path: str,
        pattern: Optional[BugPattern] = None,
    ) -> List[Finding]:
        """Inspect the AST and return any findings."""
        pass
