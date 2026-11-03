"""Project settings: `.env` loading and the folder that holds the agent's memory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_PACKAGE_PARENT = Path(__file__).resolve().parent.parent
_loaded = False


def _parse_line(line: str) -> Optional[tuple]:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    key, value = line.split("=", 1)
    key, value = key.strip(), value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return (key, value) if key else None


def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """The nearest `.env` in the working directory or its parents, else the one beside the package."""
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents, _PACKAGE_PARENT]:
        candidate = folder / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(path: Optional[Path] = None, *, force: bool = False) -> Optional[Path]:
    """Read `.env` into the environment. Values already set in the environment win."""
    global _loaded
    if os.environ.get("SBA_NO_DOTENV"):  # tests and CI must never pick up someone's real keys
        return None
    if _loaded and not force and path is None:
        return None
    _loaded = True
    env_file = Path(path) if path else find_env_file()
    if env_file is None or not env_file.is_file():
        return None
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        parsed = _parse_line(line)
        if parsed and not os.environ.get(parsed[0]):
            os.environ[parsed[0]] = parsed[1]
    return env_file


def data_home() -> Path:
    """Where the experience memory lives; `SBA_HOME` overrides it."""
    home = Path(os.environ.get("SBA_HOME") or Path.home() / ".silent_bug_agent")
    home.mkdir(parents=True, exist_ok=True)
    return home
