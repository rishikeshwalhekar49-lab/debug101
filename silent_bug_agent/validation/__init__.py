"""Validation package containing empirical verification agents."""

from .multiseedagent import MultiseedAgent, compare_distributions, compute_cohens_d, compute_confidence_interval
from .overfit_agent import OverfitOracleAgent
from .sandbox import ExecutionResult, SandboxConfig, SubprocessSandbox, TelemetryForwarder

__all__ = [
    "OverfitOracleAgent",
    "MultiseedAgent",
    "SubprocessSandbox",
    "SandboxConfig",
    "ExecutionResult",
    "TelemetryForwarder",
    "compare_distributions",
    "compute_confidence_interval",
    "compute_cohens_d",
]
