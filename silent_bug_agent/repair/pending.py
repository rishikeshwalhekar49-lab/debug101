"""A validated patch that is waiting for the user's approval.

`fix --dry-run` validates a patch but leaves the script alone. The patch is kept
here so that `apply` can later write exactly the patch that was validated and
shown, and only if the script has not changed since. This is what makes an
approval step real: nothing is written before the user says so.

Two more things live here:

* the bundled examples are never edited, so for them `apply` writes the fixed
  version to a copy in `fixed_examples/` instead;
* a patch the user turned down is remembered, so that `fix --another` can look for
  a different one.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import _PACKAGE_PARENT, data_home
from .patch_utils import MalformedDiffError, apply_patch_to_file, apply_unified_diff

COPY_FOLDER = "fixed_examples"


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
                        "planted bug, so it is never edited in place. Applying the fix writes the fixed version to a "
                        f"copy in {COPY_FOLDER}/ instead.")


def copy_path(target: Path) -> Path:
    """Where `apply` writes the fixed version of a bundled example: a folder beside where the command runs."""
    return Path.cwd() / COPY_FOLDER / Path(target).name


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slot(target: Path, kind: str = "") -> Path:
    folder = data_home() / "pending"
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(str(Path(target).resolve()).encode("utf-8")).hexdigest()[:20]
    return folder / (name + (f".{kind}" if kind else "") + ".json")


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


# ----------------------------------------------------------- patches turned down
def load_rejected(target: Path) -> List[Dict[str, str]]:
    """The patches the user turned down for this exact version of the file, as {"diff", "bug_id"}."""
    target = Path(target).resolve()
    slot = _slot(target, "rejected")
    try:
        record = json.loads(slot.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(record, dict) or record.get("sha256") != _digest(target):
        return []  # the file changed, so the old patches no longer describe it
    return [item for item in record.get("patches", []) if isinstance(item, dict) and item.get("diff")]


def clear_rejected(target: Path) -> None:
    _slot(Path(target), "rejected").unlink(missing_ok=True)


def reject_pending(target: Path) -> List[Dict[str, str]]:
    """The user turned the pending patch down. Remember it, and return every patch turned down so far."""
    target = Path(target).resolve()
    rejected = load_rejected(target)
    record = load_pending(target)
    if record is not None and record.get("sha256") == _digest(target) \
            and record["diff"] not in [item["diff"] for item in rejected]:
        rejected.append({"diff": record["diff"], "bug_id": str(record.get("bug_id", ""))})
    clear_pending(target)
    _slot(target, "rejected").write_text(json.dumps({"sha256": _digest(target), "patches": rejected}),
                                         encoding="utf-8")
    return rejected


def apply_pending(target: Path) -> Dict[str, Any]:
    """Write the pending validated patch. Returns the record with `written` (the file that was
    written), `backup` and `copy` added.

    An ordinary script is patched in place, with a `.orig` backup. A bundled example is left as
    it is, and its fixed version is written to `fixed_examples/` instead."""
    target = Path(target).resolve()
    record = load_pending(target)
    if record is None:
        raise PendingPatchError(f"There is no validated patch waiting for {target.name}. Run "
                                f"`fix {target.name} --dry-run` first; it validates a patch and keeps it for `apply`.")
    if record.get("sha256") != _digest(target):
        clear_pending(target)
        raise PendingPatchError(f"{target.name} has changed since the patch was validated, so the patch was "
                                "discarded. Run `fix --dry-run` again.")
    try:
        if bundled_example(target):
            raw = target.read_bytes().decode("utf-8")
            fixed = apply_unified_diff(raw.replace("\r\n", "\n"), record["diff"])
            written = copy_path(target)
            written.parent.mkdir(parents=True, exist_ok=True)
            written.write_bytes((fixed.replace("\n", "\r\n") if "\r\n" in raw else fixed).encode("utf-8"))
            record.update({"written": str(written), "backup": None, "copy": True})
        else:
            backup = apply_patch_to_file(str(target), record["diff"])
            record.update({"written": str(target), "backup": str(backup) if backup else None, "copy": False})
    except MalformedDiffError as exc:
        raise PendingPatchError(f"The saved patch no longer fits the file: {exc}") from exc
    clear_pending(target)
    clear_rejected(target)
    return record
