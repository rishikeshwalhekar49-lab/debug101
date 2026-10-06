"""Semantic (LLM-based) analyzer reviewing training scripts for intent and logic bugs."""

import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Union

from ..findings import Finding, Report, parse_severity
from ..invariants.base import Severity
from ..taxonomy.loader import BugPattern, load_taxonomy
from .client import LLMClient
from .prompt import SYSTEM_PROMPT, build_analysis_prompt


class SemanticAnalyzer:
    """Uses an LLM to detect silent bugs that require intent understanding."""

    def __init__(self, client: LLMClient, taxonomy: Optional[List[BugPattern]] = None):
        self.client = client
        self.patterns = taxonomy if taxonomy is not None else load_taxonomy()
        self.pattern_map: Dict[str, BugPattern] = {p.id: p for p in self.patterns}

    def _extract_json_array(self, response_text: str) -> Optional[List[dict]]:
        """Extract a JSON array from LLM response text, handling markdown fences."""
        text = response_text.strip()
        # Strip ```json ... ``` or ``` ... ``` code blocks
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # Look for outermost array [ ... ]
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return None

    def _verify_evidence_grounding(
        self, evidence: str, line_no: int, source_lines: List[str], window: int = 3
    ) -> bool:
        """Check if the quoted evidence actually appears within `window` lines of `line_no`."""
        if not evidence or not source_lines:
            return False

        clean_evidence = " ".join(evidence.split()).strip()
        if not clean_evidence:
            return False

        start = max(0, line_no - 1 - window)
        end = min(len(source_lines), line_no + window)

        for l in source_lines[start:end]:
            clean_l = " ".join(l.split()).strip()
            if clean_evidence in clean_l or clean_l in clean_evidence:
                return True
        return False

    def analyze_code(self, source: str, file_path: str = "<string>") -> Report:
        """Run semantic analysis on a Python source string."""
        report = Report(files_scanned=[file_path])
        source_lines = source.splitlines()

        prompt = build_analysis_prompt(self.patterns, source, file_path=file_path)

        try:
            raw_response = self.client.complete(prompt, system_prompt=SYSTEM_PROMPT)
        except Exception as e:
            report.diagnostics.append(f"LLM client error for {file_path}: {e}")
            return report

        items = self._extract_json_array(raw_response)
        if items is None:
            report.diagnostics.append(
                f"Failed to parse valid JSON array from LLM response for {file_path}"
            )
            return report

        for item in items:
            if not isinstance(item, dict):
                continue

            bug_id = item.get("bug_id")
            if not bug_id or bug_id not in self.pattern_map:
                report.diagnostics.append(
                    f"Dropped finding with unrecognized bug_id '{bug_id}'"
                )
                continue

            pattern = self.pattern_map[bug_id]

            line_no = item.get("line", 1)
            try:
                line_no = int(line_no)
            except (ValueError, TypeError):
                line_no = 1

            if line_no < 1 or line_no > max(1, len(source_lines)):
                line_no = 1

            evidence = str(item.get("evidence", ""))
            confidence = float(item.get("confidence", 0.7))
            message = str(item.get("message", pattern.name))
            raw_sev = item.get("severity", "warning")
            try:
                severity = parse_severity(raw_sev)
            except Exception:
                severity = Severity.WARNING

            # Evidence grounding check: if evidence is not found near line, downgrade confidence
            extra = {}
            if evidence and source_lines:
                grounded = self._verify_evidence_grounding(evidence, line_no, source_lines)
                if not grounded:
                    confidence = min(confidence * 0.5, 0.4)
                    extra["grounding"] = "ungrounded_evidence"
                else:
                    extra["grounding"] = "verified"

            report.findings.append(
                Finding(
                    bug_id=bug_id,
                    detector="semantic",
                    message=message,
                    severity=severity,
                    confidence=confidence,
                    file=file_path,
                    line=line_no,
                    evidence=evidence,
                    suggested_fix=pattern.typical_fix,
                    rule="SemanticReviewer",
                    extra=extra,
                )
            )

        return report

    def analyze_file(self, file_path: Union[str, Path]) -> Report:
        """Run semantic analysis on a file."""
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
        """Run semantic analysis on all .py files in a directory."""
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
