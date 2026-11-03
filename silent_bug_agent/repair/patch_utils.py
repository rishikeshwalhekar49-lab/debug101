"""Utilities for generating and safely validating proposed unified diffs."""

from __future__ import annotations

import difflib
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


class MalformedDiffError(ValueError):
    """Raised when a patch cannot be applied safely."""


def generate_minimal_diff(
    original: str,
    updated: str,
    filename: str,
) -> str:
    """Return a unified diff for a single file."""
    if original == updated:
        return ""

    name = Path(filename).name or "target.py"
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="a/" + name,
            tofile="b/" + name,
            n=3,
        )
    )


def apply_diff_in_temp(
    target_file: str,
    diff: str,
) -> Tuple[bool, str, Optional[str]]:
    """Apply a single-file patch to a temporary copy; never modify the original."""
    source = Path(target_file)

    if not source.is_file():
        return False, "Target file does not exist.", None

    if not diff.strip():
        return False, "Patch is empty.", None

    # Only accept a standard, single-file unified diff targeting this file.
    expected_old = "a/" + source.name
    expected_new = "b/" + source.name
    lines = diff.splitlines()

    if len(lines) < 4 or lines[0] != "--- " + expected_old or lines[1] != "+++ " + expected_new:
        return False, "Rejected patch: file paths do not match the target.", None

    if sum(line.startswith("--- ") for line in lines) != 1:
        return False, "Rejected patch: multiple file sections are not allowed.", None

    if sum(line.startswith("+++ ") for line in lines) != 1:
        return False, "Rejected patch: multiple file sections are not allowed.", None

    try:
        original = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return False, "Could not read target file: " + str(exc), None

    with tempfile.TemporaryDirectory(prefix="silentbug_repair_") as temp_dir:
        temp_root = Path(temp_dir)
        temp_target = temp_root / source.name
        temp_target.write_text(original, encoding="utf-8")

        patch_file = temp_root / "candidate.patch"
        patch_file.write_text(diff, encoding="utf-8")

        check = subprocess.run(
            ["git", "apply", "--check", "-p1", "candidate.patch"],
            cwd=str(temp_root),
            capture_output=True,
            text=True,
            check=False,
        )

        if check.returncode != 0:
            message = (check.stderr or check.stdout).strip()
            return False, "Patch could not be applied: " + message, None

        applied = subprocess.run(
            ["git", "apply", "-p1", "candidate.patch"],
            cwd=str(temp_root),
            capture_output=True,
            text=True,
            check=False,
        )

        if applied.returncode != 0:
            message = (applied.stderr or applied.stdout).strip()
            return False, "Patch application failed: " + message, None

        try:
            updated = temp_target.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return False, "Could not read patched temporary file: " + str(exc), None

        if source.suffix.lower() == ".py":
            try:
                compile(updated, str(source), "exec")
            except SyntaxError as exc:
                return False, "Python syntax validation failed: " + str(exc), None

        return (
            True,
            "Patch applied to temporary copy; syntax check passed where applicable.",
            updated,
        )
