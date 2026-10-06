"""Static code analyzer running AST rules against training scripts."""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..findings import Finding, Report
from ..taxonomy.loader import BugPattern, load_taxonomy
from .base import BaseRule
from .rules import DEFAULT_STATIC_RULES


class StaticAnalyzer:
    """Orchestrates AST-based static analysis rules against files and directories."""

    def __init__(self, rules: Optional[List[BaseRule]] = None):
        self.rules = rules if rules is not None else list(DEFAULT_STATIC_RULES)
        self.taxonomy: Dict[str, BugPattern] = {p.id: p for p in load_taxonomy()}

        # Verify that all rule bug_ids exist in the taxonomy
        for rule in self.rules:
            if rule.bug_id not in self.taxonomy:
                raise ValueError(
                    f"Static rule {rule.__class__.__name__} references unknown bug_id '{rule.bug_id}'"
                )

    def analyze_code(self, source: str, file_path: str = "<string>") -> Report:
        """Analyze a snippet of Python source code."""
        report = Report(files_scanned=[file_path])
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            report.diagnostics.append(f"SyntaxError in {file_path}:{e.lineno}: {e.msg}")
            return report

        source_lines = source.splitlines()
        for rule in self.rules:
            pattern = self.taxonomy.get(rule.bug_id)
            findings = rule.check(tree, source_lines, file_path, pattern)
            report.findings.extend(findings)

        return report

    def analyze_file(self, file_path: Union[str, Path]) -> Report:
        """Analyze a single Python file."""
        p = Path(file_path).resolve()
        if not p.is_file():
            return Report(diagnostics=[f"File not found: {file_path}"])

        try:
            source = p.read_text(encoding="utf-8")
        except Exception as e:
            return Report(diagnostics=[f"Could not read {file_path}: {e}"])

        return self.analyze_code(source, file_path=str(p))

    def analyze_directory(
        self, dir_path: Union[str, Path], recursive: bool = True
    ) -> Report:
        """Analyze all .py files in a directory."""
        root = Path(dir_path).resolve()
        if not root.is_dir():
            return Report(diagnostics=[f"Directory not found: {dir_path}"])

        combined_report = Report()
        pattern = "**/*.py" if recursive else "*.py"
        skip_dirs = {".venv", "venv", "__pycache__", ".git", ".pytest_cache", "build", "dist"}

        for py_file in root.glob(pattern):
            if any(part in skip_dirs for part in py_file.parts):
                continue
            file_report = self.analyze_file(py_file)
            combined_report.extend(file_report)

        return combined_report
