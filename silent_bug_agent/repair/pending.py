"""A validated patch that is waiting for the user's approval.

`fix --dry-run` validates a patch but leaves the script alone. The patch is kept
here so that `apply` can later write exactly the patch that was validated and
shown, and only if the script has not changed since. This is what makes an
approval step real: nothing is written before the user says so.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import _PACKAGE_PARENT, data_home
from .patch_utils import MalformedDiffError, apply_patch_to_file


class PendingPatchError(RuntimeError):
    """Why a pending patch cannot be applied, in words the user can act on."""


def bundled_example(target: Path) -> bool:
    """True for the example scripts shipped in a source checkout. The tests and the
    benchmark results depend on their planted bugs, so they are never edited in place."""
    examples = _PACKAGE_PARENT / "examples"
    if not (examples.is_dir() and (_PACKAGE_PARENT / "pyproject.toml").is_file()):
        return False
    try:
        Path(target).resolve().relative_to(examples.resolve())
        return True
    except ValueError:
        return False


BUNDLED_EXAMPLE_NOTE = ("This is one of the project's bundled examples. The tests and benchmark results rely on its "
                        "planted bug, so it is never edited in place. Copy it somewhere else and run the command on "
                        "the copy to apply a fix.")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slot(target: Path) -> Path:
    folder = data_home() / "pending"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (hashlib.sha256(str(Path(target).resolve()).encode("utf-8")).hexdigest()[:20] + ".json")


def save_pending(target: Path, diff: str, *, bug_id: str = "", run_id: str = "") -> Path:
    target = Path(target).resolve()
    record = {"target": str(target), "sha256": _digest(target), "diff": diff, "bug_id": bug_id, "run_id": run_id,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    slot = _slot(target)
    slot.write_text(json.dumps(record), encoding="utf-8")
    return slot


def load_pending(target: Path) -> Optional[Dict[str, Any]]:
    slot = _slot(Path(target))
    if not slot.is_file():
        return None
    try:
        record = json.loads(slot.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and record.get("diff") else None


def clear_pending(target: Path) -> None:
    _slot(Path(target)).unlink(missing_ok=True)


def apply_pending(target: Path) -> Dict[str, Any]:
    """Write the pending validated patch to `target`. Returns the record, with the backup's path added."""
    target = Path(target).resolve()
    if bundled_example(target):
        raise PendingPatchError(BUNDLED_EXAMPLE_NOTE)
    record = load_pending(target)
    if record is None:
        raise PendingPatchError(f"There is no validated patch waiting for {target.name}. Run "
                                f"`fix {target.name} --dry-run` first; it validates a patch and keeps it for `apply`.")
    if record.get("sha256") != _digest(target):
        clear_pending(target)
        raise PendingPatchError(f"{target.name} has changed since the patch was validated, so the patch was "
                                "discarded. Run `fix --dry-run` again.")
    try:
        backup = apply_patch_to_file(str(target), record["diff"])
    except MalformedDiffError as exc:
        raise PendingPatchError(f"The saved patch no longer fits the file: {exc}") from exc
    clear_pending(target)
    record["backup"] = str(backup) if backup else None
    return record
