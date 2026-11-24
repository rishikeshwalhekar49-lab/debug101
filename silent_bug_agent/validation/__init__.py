"""Validation package containing empirical verification agents.

The names below are imported on first use. Every validation run starts a fresh
`python -m silent_bug_agent.validation.script_runner`, and importing this package
eagerly would load scipy and the agents into each of those processes for nothing
(about a second each).
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MultiseedAgent": ".multiseedagent",
    "compare_distributions": ".multiseedagent",
    "compute_cohens_d": ".multiseedagent",
    "compute_confidence_interval": ".multiseedagent",
    "OverfitOracleAgent": ".overfit_agent",
    "ExecutionResult": ".sandbox",
    "SandboxConfig": ".sandbox",
    "SubprocessSandbox": ".sandbox",
    "TelemetryForwarder": ".sandbox",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
