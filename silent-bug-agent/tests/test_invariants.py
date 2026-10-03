"""Plain-assert tests, runnable with `python -m pytest` or directly.

Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from silent_bug_agent import Severity, TrainingState
from silent_bug_agent.invariants import (
    ActivationHealthInvariant,
    FrozenWeightsInvariant,
    GradientNormBoundsInvariant,
    LossTrendInvariant,
    TrainEvalModeInvariant,
)


def test_gradient_norm_bounds_passes_on_healthy_gradients():
    inv = GradientNormBoundsInvariant(min_norm=1e-6, max_norm=100)
    state = TrainingState(step=1, grad_norms={"w": 0.5})
    result = inv.check(state)
    assert result.passed


def test_gradient_norm_bounds_flags_explosion():
    inv = GradientNormBoundsInvariant(min_norm=1e-6, max_norm=100)
    state = TrainingState(step=1, grad_norms={"w": 5000})
    result = inv.check(state)
    assert not result.passed
    assert result.severity == Severity.CRITICAL


def test_gradient_norm_bounds_flags_nan():
    inv = GradientNormBoundsInvariant()
    state = TrainingState(step=1, grad_norms={"w": float("nan")})
    result = inv.check(state)
    assert not result.passed


def test_frozen_weights_needs_patience_before_flagging():
    inv = FrozenWeightsInvariant(delta_threshold=1e-9, patience=3)
    for step in range(1, 3):
        state = TrainingState(step=step, weight_deltas={"w": 0.0}, requires_grad={"w": True})
        result = inv.check(state)
        assert result.passed, "should not flag before patience window elapses"

    state = TrainingState(step=3, weight_deltas={"w": 0.0}, requires_grad={"w": True})
    result = inv.check(state)
    assert not result.passed


def test_frozen_weights_ignores_intentionally_frozen_params():
    inv = FrozenWeightsInvariant(patience=1)
    state = TrainingState(step=1, weight_deltas={"w": 0.0}, requires_grad={"w": False})
    result = inv.check(state)
    assert result.passed


def test_train_eval_mode_flags_batchnorm_left_in_train_during_eval():
    inv = TrainEvalModeInvariant()
    state = TrainingState(step=1, module_modes={"model.batchnorm1": "train"}, is_eval_phase=True)
    result = inv.check(state)
    assert not result.passed
    assert result.severity == Severity.CRITICAL


def test_train_eval_mode_ignores_mode_insensitive_modules():
    inv = TrainEvalModeInvariant()
    state = TrainingState(step=1, module_modes={"model.linear1": "train"}, is_eval_phase=True)
    result = inv.check(state)
    assert result.passed


def test_activation_health_flags_nan():
    inv = ActivationHealthInvariant()
    state = TrainingState(step=1, activation_stats={"layer1": {"nan_count": 3, "inf_count": 0, "zero_frac": 0.0}})
    result = inv.check(state)
    assert not result.passed
    assert result.severity == Severity.CRITICAL


def test_activation_health_flags_dead_units_as_warning_not_critical():
    inv = ActivationHealthInvariant(dead_frac_threshold=0.9)
    state = TrainingState(step=1, activation_stats={"layer1": {"nan_count": 0, "inf_count": 0, "zero_frac": 0.95}})
    result = inv.check(state)
    assert not result.passed
    assert result.severity == Severity.WARNING


def test_loss_trend_waits_for_full_window():
    inv = LossTrendInvariant(window=9, tolerance=0.01)
    state = TrainingState(step=1, loss_history=[1.0, 0.9, 0.8])
    result = inv.check(state)
    assert result.passed  # not enough data yet, should not fail


def test_loss_trend_flags_stalled_loss():
    inv = LossTrendInvariant(window=9, tolerance=0.05)
    flat_history = [1.0] * 9  # no improvement at all
    state = TrainingState(step=9, loss_history=flat_history)
    result = inv.check(state)
    assert not result.passed


def test_loss_trend_passes_on_noisy_but_improving_loss():
    inv = LossTrendInvariant(window=9, tolerance=0.05)
    # noisy but clearly trending down: not monotonic, still should pass
    history = [1.0, 1.05, 0.95, 0.8, 0.85, 0.7, 0.6, 0.65, 0.5]
    state = TrainingState(step=9, loss_history=history)
    result = inv.check(state)
    assert result.passed


def test_taxonomy_loads_and_validates():
    from silent_bug_agent.taxonomy import coverage_report, load_taxonomy

    patterns = load_taxonomy()
    assert len(patterns) >= 15
    ids = [p.id for p in patterns]
    assert len(ids) == len(set(ids)), "taxonomy ids must be unique"

    report = coverage_report(patterns)
    assert report["total_patterns"] == len(patterns)


if __name__ == "__main__":
    # allow running without pytest installed
    import inspect

    failures = 0
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
