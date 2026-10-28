"""Benchmark Suite Runner for Silent DL Bugs.

Runs the curated suite of silent bug benchmarks across multiple random seeds,
compares loss distributions between buggy baseline and fixed variants using
Kolmogorov-Smirnov / paired t-tests, and verifies that the fix genuinely
resolves the silent bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repository root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silent_bug_agent.validation.multiseedagent import (
    DistributionComparisonResult,
    compare_distributions,
)
from silent_bug_agent.validation.sandbox import SubprocessSandbox


BENCHMARKS = [
    ("benchmark_01_missing_zero_grad.py", "Missing optimizer.zero_grad()"),
    ("benchmark_02_train_eval_mode.py", "BatchNorm/Dropout mode mismatch"),
    ("benchmark_03_frozen_layer.py", "Accidentally frozen trainable layer"),
    ("benchmark_04_softmax_before_crossentropy.py", "Softmax before CrossEntropyLoss"),
    ("benchmark_05_learning_rate_too_high.py", "Learning rate misconfigured (too high)"),
    ("benchmark_06_dead_relu.py", "Dead ReLU units from bias initialization"),
    ("benchmark_07_wrong_loss_reduction.py", "Wrong loss reduction (sum vs mean)"),
    ("benchmark_08_unnormalized_inputs.py", "Missing input normalization"),
]


def run_benchmark_on_seeds(
    script_path: Path,
    seeds: List[int],
    fix: bool = False,
    sandbox: Optional[SubprocessSandbox] = None,
) -> List[float]:
    """Execute a benchmark across the given seeds and return final losses."""
    box = sandbox or SubprocessSandbox()
    losses = []
    args = ["--fix"] if fix else []

    for seed in seeds:
        res = box.run_script(script_path, args=args, seed=seed, timeout=20.0)
        if res.success and "final_loss" in res.metrics:
            losses.append(float(res.metrics["final_loss"]))
        elif res.success and "Final Loss:" in res.stdout:
            try:
                line = [l for l in res.stdout.splitlines() if "Final Loss:" in l][-1]
                loss_val = float(line.split(":")[-1].strip())
                losses.append(loss_val)
            except Exception:
                losses.append(999.0)
        else:
            losses.append(999.0)

    return losses


def run_all_benchmarks(seeds: Optional[List[int]] = None) -> List[Dict]:
    """Run all 8 benchmarks across 3-5 seeds and report statistical verification."""
    eval_seeds = seeds or [42, 1337, 2024, 7, 99]
    scripts_dir = Path(__file__).resolve().parent
    sandbox = SubprocessSandbox()

    results = []

    print("=" * 80)
    print(f"Running Silent Bug Benchmark Suite ({len(BENCHMARKS)} benchmarks, {len(eval_seeds)} seeds)")
    print(f"Seeds: {eval_seeds}")
    print("=" * 80)

    for filename, description in BENCHMARKS:
        path = scripts_dir / filename
        if not path.is_file():
            print(f"[-] Missing benchmark file: {filename}")
            continue

        print(f"\n[+] Evaluating: {filename} ({description})")
        # Run buggy baseline
        buggy_losses = run_benchmark_on_seeds(path, eval_seeds, fix=False, sandbox=sandbox)
        # Run fixed variant
        fixed_losses = run_benchmark_on_seeds(path, eval_seeds, fix=True, sandbox=sandbox)

        # Statistical comparison
        comparison = compare_distributions(
            baseline_losses=buggy_losses,
            patched_losses=fixed_losses,
            alpha=0.05,
            prefer_test="t_test",
        )

        status = "PASSED (Resolved)" if comparison.is_significant else "FAILED (Inconclusive/Noise)"
        print(f"    Buggy Mean Loss: {comparison.mean_baseline:.4f}")
        print(f"    Fixed Mean Loss: {comparison.mean_patched:.4f}")
        print(f"    Mean Reduction:  {comparison.mean_difference:.4f}")
        print(f"    p-value:         {comparison.p_value:.6f}")
        print(f"    Effect Size (d): {comparison.effect_size:.4f}")
        print(f"    95% CI:          [{comparison.confidence_interval_95[0]:.4f}, {comparison.confidence_interval_95[1]:.4f}]")
        print(f"    Verdict:         {status}")

        results.append({
            "benchmark": filename,
            "description": description,
            "comparison": comparison,
            "passed": comparison.is_significant,
        })

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    all_passed = True
    for r in results:
        status_sym = "[OK]" if r["passed"] else "[FAIL]"
        comp: DistributionComparisonResult = r["comparison"]
        print(f"{status_sym} {r['benchmark']:<42} p={comp.p_value:<8.4f} diff={comp.mean_difference:<8.4f} {r['description']}")
        if not r["passed"]:
            all_passed = False

    print("=" * 80)
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Total benchmarks verified: {passed_count}/{total}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Silent Bug Multi-Seed Benchmark Runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024, 7, 99], help="Seeds to evaluate")
    args = parser.parse_args()
    results = run_all_benchmarks(seeds=args.seeds)
    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
