"""Tests verifying the 8+ real-world silent bug training benchmarks."""

from pathlib import Path
import pytest

from silent_bug_agent.validation.multiseedagent import compare_distributions
from silent_bug_agent.validation.sandbox import SubprocessSandbox

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "examples" / "sample_scripts"

BENCHMARK_FILES = [
    "benchmark_01_missing_zero_grad.py",
    "benchmark_02_train_eval_mode.py",
    "benchmark_03_frozen_layer.py",
    "benchmark_04_softmax_before_crossentropy.py",
    "benchmark_05_learning_rate_too_high.py",
    "benchmark_06_dead_relu.py",
    "benchmark_07_wrong_loss_reduction.py",
    "benchmark_08_unnormalized_inputs.py",
]


def test_all_benchmark_files_exist():
    assert len(BENCHMARK_FILES) >= 8
    for fname in BENCHMARK_FILES:
        path = BENCHMARKS_DIR / fname
        assert path.is_file(), f"Benchmark file {fname} not found in {BENCHMARKS_DIR}"


@pytest.mark.parametrize("benchmark_file", BENCHMARK_FILES)
def test_benchmark_runs_in_sandbox(benchmark_file):
    sandbox = SubprocessSandbox()
    path = BENCHMARKS_DIR / benchmark_file

    # Run buggy version on 1 seed to verify it executes cleanly without loud crash
    buggy_res = sandbox.run_script(path, seed=42, timeout=15.0)
    assert buggy_res.success, f"{benchmark_file} failed with stderr: {buggy_res.stderr}"
    assert "final_loss" in buggy_res.metrics

    # Run fixed version on 1 seed to verify fix executes cleanly
    fixed_res = sandbox.run_script(path, args=["--fix"], seed=42, timeout=15.0)
    assert fixed_res.success, f"{benchmark_file} with --fix failed with stderr: {fixed_res.stderr}"
    assert "final_loss" in fixed_res.metrics

    # Verify that the fixed version has lower or equal loss than buggy version
    buggy_loss = buggy_res.metrics["final_loss"]
    fixed_loss = fixed_res.metrics["final_loss"]
    assert fixed_loss <= buggy_loss, (
        f"{benchmark_file}: fixed loss ({fixed_loss}) was greater than buggy loss ({buggy_loss})"
    )
