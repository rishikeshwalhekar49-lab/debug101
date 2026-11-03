"""Generating, checking and applying unified diffs, in pure Python.

No `git` or `patch` command is needed, so this works the same on Windows.
"""

from __future__ import annotations

import ast
import difflib
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NO_NEWLINE = "\\ No newline at end of file"
# code a repair has no reason to add: patches are executed, so these are refused outright
_FORBIDDEN = re.compile(r"\b(subprocess|socket|requests|urllib|httpx|ctypes|shutil\.rmtree|os\.system|os\.remove|"
                        r"os\.unlink|os\._exit|sys\.exit|__import__)|(?<![\w.])(eval|exec|exit|quit|compile)\(")


class MalformedDiffError(ValueError):
    """Raised when a patch cannot be applied safely."""


def generate_minimal_diff(original: str, updated: str, filename: str) -> str:
    """Return a unified diff for a single file."""
    if original == updated:
        return ""
    name = Path(filename).name or "target.py"
    out: List[str] = []
    for line in difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True),
                                     fromfile="a/" + name, tofile="b/" + name, n=3):
        out.append(line if line.endswith("\n") else line + "\n" + _NO_NEWLINE + "\n")
    return "".join(out)


def apply_unified_diff(original: str, diff: str) -> str:
    """Apply a single-file unified diff. Every context line must match exactly."""
    source = original.splitlines(keepends=True)
    lines = diff.splitlines(keepends=True)
    out: List[str] = []
    position, index, hunks = 0, 0, 0
    while index < len(lines):
        match = _HUNK.match(lines[index])
        index += 1
        if not match:
            continue
        hunks += 1
        start = int(match.group(1)) - (0 if match.group(2) == "0" else 1)
        if start < position or start > len(source):
            raise MalformedDiffError(f"hunk at line {match.group(1)} is out of place")
        out.extend(source[position:start])
        position = start
        last_kind = ""
        while index < len(lines) and not _HUNK.match(lines[index]):
            raw = lines[index]
            index += 1
            if raw.startswith("\\"):  # the line before it has no newline; for a removed line nothing is kept
                if last_kind in (" ", "+") and out and out[-1].endswith("\n"):
                    out[-1] = out[-1][:-1]
                continue
            kind, text = raw[:1], raw[1:]
            last_kind = kind
            if kind in (" ", "-"):
                current = source[position] if position < len(source) else None
                if current is None or current.rstrip("\r\n") != text.rstrip("\r\n"):
                    raise MalformedDiffError(f"line {position + 1} does not match the patch context")
                position += 1
                if kind == " ":
                    out.append(current)
            elif kind == "+":
                out.append(text)
            elif raw.strip():
                raise MalformedDiffError(f"unexpected patch line: {raw[:60]!r}")
    if hunks == 0:
        raise MalformedDiffError("the patch has no hunks")
    out.extend(source[position:])
    return "".join(out)


def patch_guard(original: str, updated: str) -> Optional[str]:
    """Why this patch must not run, or None. Patches come from a model and get executed."""
    try:
        ast.parse(updated)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    before, after = original.splitlines(), updated.splitlines()
    known = set(before)
    for line in after:
        if line not in known and _FORBIDDEN.search(line):
            return f"the patch adds a forbidden operation: {line.strip()[:80]}"
    for needle in (".backward(", ".step("):
        if original.count(needle) and updated.count(needle) < original.count(needle):
            return f"the patch removes a {needle.strip('.(')} call"
    if len(before) >= 10 and len(after) < 0.6 * len(before):
        return "the patch removes too much of the script"
    return None


def apply_diff_in_temp(target_file: str, diff: str) -> Tuple[bool, str, Optional[str]]:
    """Check a patch against the target without modifying it; returns (ok, message, patched text)."""
    source = Path(target_file)
    if not source.is_file():
        return False, "Target file does not exist.", None
    if not diff.strip():
        return False, "Patch is empty.", None

    expected_old, expected_new = "a/" + source.name, "b/" + source.name
    lines = diff.splitlines()
    if len(lines) < 4 or lines[0] != "--- " + expected_old or lines[1] != "+++ " + expected_new:
        return False, "Rejected patch: file paths do not match the target.", None
    if sum(line.startswith("--- ") for line in lines) != 1 or sum(line.startswith("+++ ") for line in lines) != 1:
        return False, "Rejected patch: multiple file sections are not allowed.", None

    try:
        original = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return False, "Could not read target file: " + str(exc), None
    try:
        updated = apply_unified_diff(original, diff)
    except MalformedDiffError as exc:
        return False, "Patch could not be applied: " + str(exc), None
    if source.suffix.lower() == ".py":
        try:
            compile(updated, str(source), "exec")
        except SyntaxError as exc:
            return False, "Python syntax validation failed: " + str(exc), None
    return True, "Patch applied to temporary copy; syntax check passed where applicable.", updated


def apply_patch_to_file(target_file: str, diff: str, *, backup: bool = True) -> Optional[Path]:
    """Write an accepted patch to the real file, keeping its line endings and a `.orig` copy.

    An existing `.orig` is never overwritten, so after several fixes it still holds the true original."""
    path = Path(target_file)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    crlf = "\r\n" in text
    updated = apply_unified_diff(text.replace("\r\n", "\n"), diff)
    backup_path = None
    if backup:
        backup_path = path.with_name(path.name + ".orig")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
    path.write_bytes((updated.replace("\n", "\r\n") if crlf else updated).encode("utf-8"))
    return backup_path
