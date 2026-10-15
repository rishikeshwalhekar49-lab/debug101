"""Module 1: runtime perception for the Runtime Monitor Agent.

Non-intrusive PyTorch hooks that populate the framework-agnostic
``TrainingState`` snapshots checked by the Module 0 invariants. Requires torch
(``pip install -e ".[torch]"``); the rest of ``silent_bug_agent`` does not
import this package, so the core library stays torch-free.
"""

from .hooks import (
    ActivationCollector,
    HookConfig,
    LossCollector,
    ParameterCollector,
    RuntimeHooks,
    SnapshotStream,
    TensorStatsExtractor,
    module_label,
    select_layers,
)
from .numerics import EmaZScore, spike_indices

__all__ = [
    "ActivationCollector",
    "EmaZScore",
    "HookConfig",
    "LossCollector",
    "ParameterCollector",
    "RuntimeHooks",
    "SnapshotStream",
    "TensorStatsExtractor",
    "module_label",
    "select_layers",
    "spike_indices",
]
