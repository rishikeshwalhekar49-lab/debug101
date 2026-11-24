"""Checking a code snippet: a few lines pasted from a notebook, a chat or a review.

A snippet usually cannot run: it has no data, its model is defined elsewhere, and
it may be cut mid-statement. So snippet mode is static only. The AST rules (and,
with a key, the LLM review) read the code; nothing is executed, and no finding
from a snippet is a validated result. Every report says so.

To get a snippet to parse, the parser tries a few repairs, in order, and records
each one it used: markdown fences and `>>>` prompts are removed, common
indentation is removed, the code is wrapped in a function (so a `return` from a
method body is legal), and an unfinished last line or two is dropped.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

from ..findings import Report

SNIPPET_FILE = "<snippet>"
WRAPPER = "def _snippet():\n"
MAX_DROPPED_LINES = 3
NOT_VALIDATED = ("Snippet mode: the code was read, not run, so these findings are not validated. "
                 "Run `fix` on the full training script to test a repair.")

_FENCE = re.compile(r"^\s*```[\w+-]*\s*$")
_PROMPT = re.compile(r"^(\s*)(?:>>>|\.\.\.)(?: |$)")


# Rules that make no sense for a fragment: nobody seeds the RNG in a five-line snippet.
SKIPPED_IN_SNIPPETS = {"unset_random_seed"}


@dataclass
class ParsedSnippet:
    source: str  # code that parses
    line_offset: int = 0  # lines added above the code (the function wrapper)
    repairs: List[str] = field(default_factory=list)
    origin: List[int] = field(default_factory=list)  # origin[i]: line number in the pasted text of code line i + 1

    def original_line(self, line: int) -> int:
        index = line - self.line_offset - 1
        return self.origin[index] if 0 <= index < len(self.origin) else 0


def _clean(text: str, repairs: List[str]) -> tuple:
    """(cleaned lines, their line numbers in the pasted text)"""
    rows = list(enumerate(text.replace("\r\n", "\n").replace("\t", "    ").split("\n"), start=1))
    if any(_FENCE.match(line) for _, line in rows):
        rows = [(n, line) for n, line in rows if not _FENCE.match(line)]
        repairs.append("removed markdown code fences")
    if any(_PROMPT.match(line) for _, line in rows):
        rows = [(n, _PROMPT.sub(r"\1", line)) for n, line in rows]
        repairs.append("removed >>> prompts")
    while rows and not rows[0][1].strip():
        rows.pop(0)
    while rows and not rows[-1][1].strip():
        rows.pop()
    return [line for _, line in rows], [n for n, _ in rows]


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except (SyntaxError, ValueError):
        return False


def parse_snippet(text: str) -> Optional[ParsedSnippet]:
    """The snippet as code that parses, or None when no repair makes it parse."""
    repairs: List[str] = []
    lines, origin = _clean(text, repairs)
    if not lines:
        return None
    for dropped in range(min(MAX_DROPPED_LINES, len(lines) - 1) + 1):
        raw = "\n".join(lines[:len(lines) - dropped]) + "\n"
        kept = textwrap.dedent(raw)
        notes = list(repairs)
        if kept != raw:
            notes.append("removed the common indentation")
        if dropped:
            notes.append(f"dropped {dropped} unfinished line(s) at the end")
        lines_kept = origin[:len(origin) - dropped]
        if _parses(kept):
            return ParsedSnippet(kept, 0, notes, lines_kept)
        wrapped = WRAPPER + textwrap.indent(kept, "    ")
        if _parses(wrapped):
            return ParsedSnippet(wrapped, 1, notes + ["wrapped the code in a function so it parses on its own"],
                                 lines_kept)
    return None


def analyze_snippet(text: str, *, analyzer=None, semantic_client=None) -> Report:
    """Static rules over a snippet, plus the LLM review when a client is given. Nothing is run."""
    from .analyzer import StaticAnalyzer

    parsed = parse_snippet(text)
    if parsed is None:
        try:
            ast.parse("\n".join(_clean(text, [])[0]))
        except SyntaxError as error:
            reason = f"line {error.lineno}: {error.msg}"
        else:
            reason = "it is empty"
        return Report(files_scanned=[SNIPPET_FILE],
                      diagnostics=[f"The snippet could not be parsed as Python ({reason}), so nothing was checked."])
    static = analyzer or StaticAnalyzer(rules=[rule for rule in StaticAnalyzer().rules
                                               if rule.bug_id not in SKIPPED_IN_SNIPPETS])
    report = static.analyze_code(parsed.source, SNIPPET_FILE)
    if semantic_client is not None:
        from ..findings import merge_reports
        from ..semantic.analyzer import SemanticAnalyzer

        report = merge_reports(report, SemanticAnalyzer(semantic_client).analyze_code(parsed.source, SNIPPET_FILE))
    for finding in report.findings:
        finding.line = parsed.original_line(finding.line)
        finding.extra["validated"] = False
        finding.extra["mode"] = "snippet"
    report.diagnostics.insert(0, NOT_VALIDATED)
    if parsed.repairs:
        report.diagnostics.append("To parse the snippet: " + "; ".join(parsed.repairs) + ".")
    return report
