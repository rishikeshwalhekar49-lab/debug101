"""Demo: run the invariant library against synthetic TrainingState snapshots.

No torch dependency — this simulates the shape of data Phase 1's PyTorch
hooks will eventually produce, so the invariant library can be exercised
and tested before that integration exists.

Run: python examples/demo_invariant_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from silent_bug_agent import Severity, TrainingState, get_default_registry, run_all


def print_results(label, results):
    print(f"\n--- {label} ---")
    for r in results:
        marker = "ok " if r.passed else f"!! [{r.severity.value}]"
        print(f"  {marker} {r.invariant_id}: {r.message}")


def scenario_healthy_training():
    """A normal, noisy-but-improving training run: everything should pass."""
    registry = get_default_registry()
    loss_history = []
    for step in range(1, 100):
        loss_history.append(1.0 / (1 + 0.02 * step) + 0.01 * ((-1) ** step))
        state = TrainingState(
            step=step,
            loss_history=list(loss_history),
            grad_norms={"layer1.weight": 0.5, "layer2.weight": 0.3},
            weight_deltas={"layer1.weight": 0.001, "layer2.weight": 0.0008},
            requires_grad={"layer1.weight": True, "layer2.weight": True},
            activation_stats={"layer1": {"mean": 0.1, "std": 0.5, "nan_count": 0, "inf_count": 0, "zero_frac": 0.2}},
            module_modes={"model.batchnorm1": "train"},
            is_eval_phase=False,
        )
        results = run_all(registry, state)
    print_results("Healthy training (final step)", results)


def scenario_frozen_layer():
    """layer2 never updates despite requires_grad=True: frozen_weights should fire."""
    registry = get_default_registry()
    results = None
    for step in range(1, 30):
        state = TrainingState(
            step=step,
            loss_history=[1.0 - 0.001 * step] * step,
            grad_norms={"layer1.weight": 0.4, "layer2.weight": 0.3},
            weight_deltas={"layer1.weight": 0.002, "layer2.weight": 0.0},  # layer2 frozen
            requires_grad={"layer1.weight": True, "layer2.weight": True},
            activation_stats={},
            module_modes={},
            is_eval_phase=False,
        )
        results = run_all(registry, state)
    print_results("Frozen layer (layer2 never updates)", results)


def scenario_exploding_gradients():
    registry = get_default_registry()
    state = TrainingState(
        step=42,
        loss_history=[1.0, 0.9, 0.8, float("nan")],
        grad_norms={"layer1.weight": 2500.0, "layer2.weight": 0.4},
        weight_deltas={"layer1.weight": 5.0, "layer2.weight": 0.001},
        requires_grad={"layer1.weight": True, "layer2.weight": True},
        activation_stats={},
        module_modes={},
        is_eval_phase=False,
    )
    results = run_all(registry, state)
    print_results("Exploding gradients", results)


def scenario_eval_mode_bug():
    """Validation pass, but batchnorm was never switched to eval mode."""
    registry = get_default_registry()
    state = TrainingState(
        step=100,
        loss_history=[0.5] * 90,
        grad_norms={},
        weight_deltas={},
        requires_grad={},
        activation_stats={},
        module_modes={"model.batchnorm1": "train", "model.dropout1": "train"},
        is_eval_phase=True,
    )
    results = run_all(registry, state)
    print_results("Eval pass with modules stuck in train mode", results)


if __name__ == "__main__":
    scenario_healthy_training()
    scenario_frozen_layer()
    scenario_exploding_gradients()
    scenario_eval_mode_bug()

    print("\nDone. See failing invariants (!!) above for the bugs each scenario is meant to surface.")
