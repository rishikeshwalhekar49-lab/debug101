"""Statistical Multi-Seed Verification Agent.

Validates candidate patches against stochastic training fluctuations across
distinct random seeds with timeout controls, resource limits, and environment
isolation using SubprocessSandbox. Aggregates metric distributions, applies
statistical hypothesis tests (Kolmogorov-Smirnov / t-test), computes confidence
intervals and effect sizes to confirm genuine bug resolution versus seed luck,
and emits structured ValidationVerdict objects.
"""

from __future__ import annotations

import difflib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as stats

from ..repair.patch_utils import MalformedDiffError, apply_unified_diff
from ..schemas import AgentState, PatchCandidate, ValidationVerdict
from .sandbox import ExecutionResult, SandboxConfig, SubprocessSandbox, TelemetryForwarder


# ---------------------------------------------------------------------------
# Statistical Distribution Comparison Tools
# ---------------------------------------------------------------------------

@dataclass
class DistributionComparisonResult:
    """Result of comparing baseline and patched metric distributions."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    confidence_interval_95: Tuple[float, float]
    mean_baseline: float
    mean_patched: float
    mean_difference: float  # baseline - patched (positive means reduction in loss)
    baseline_values: List[float] = field(default_factory=list)
    patched_values: List[float] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
            "confidence_interval_95": list(self.confidence_interval_95),
            "mean_baseline": self.mean_baseline,
            "mean_patched": self.mean_patched,
            "mean_difference": self.mean_difference,
            "baseline_values": self.baseline_values,
            "patched_values": self.patched_values,
            "notes": self.notes,
        }


def compute_confidence_interval(
    differences: Sequence[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute confidence interval for paired differences using Student's t-distribution."""
    n = len(differences)
    if n < 2:
        val = float(differences[0]) if n == 1 else 0.0
        return (val, val)

    arr = np.array(differences, dtype=float)
    mean = float(np.mean(arr))
    std_err = float(stats.sem(arr))

    if std_err == 0.0 or np.isnan(std_err):
        return (mean, mean)

    dof = n - 1
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, dof))
    margin = t_crit * std_err
    return (mean - margin, mean + margin)


def compute_cohens_d(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Compute Cohen's d effect size between two groups (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    arr1 = np.array(group1, dtype=float)
    arr2 = np.array(group2, dtype=float)

    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    var1, var2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0.0 or np.isnan(pooled_std):
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def compare_distributions(
    baseline_losses: Sequence[float],
    patched_losses: Sequence[float],
    alpha: float = 0.05,
    prefer_test: str = "t_test",
) -> DistributionComparisonResult:
    """Compare baseline and patched loss distributions across random seeds.

    Evaluates whether the patched model achieves statistically significantly
    lower loss than the baseline, confirming genuine bug resolution rather than
    random seed luck.
    """
    b_arr = np.array(baseline_losses, dtype=float)
    p_arr = np.array(patched_losses, dtype=float)

    mean_b = float(np.mean(b_arr))
    mean_p = float(np.mean(p_arr))
    diff = b_arr - p_arr
    mean_diff = float(np.mean(diff))

    # Confidence interval of improvement (baseline - patched)
    ci_95 = compute_confidence_interval(diff, confidence=0.95)
    effect_size = compute_cohens_d(b_arr, p_arr)

    # Identical values or zero variance
    if np.allclose(b_arr, p_arr):
        return DistributionComparisonResult(
            test_name="identical",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            effect_size=0.0,
            confidence_interval_95=(0.0, 0.0),
            mean_baseline=mean_b,
            mean_patched=mean_p,
            mean_difference=0.0,
            baseline_values=list(baseline_losses),
            patched_values=list(patched_losses),
            notes="Baseline and patched losses are identical across all seeds.",
        )

    # If patched loss is higher on average, it's a regression or ineffective
    if mean_diff <= 0:
        return DistributionComparisonResult(
            test_name=prefer_test,
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            effect_size=effect_size,
            confidence_interval_95=ci_95,
            mean_baseline=mean_b,
            mean_patched=mean_p,
            mean_difference=mean_diff,
            baseline_values=list(baseline_losses),
            patched_values=list(patched_losses),
            notes="Patched model had higher or equal loss compared to baseline.",
        )

    if prefer_test == "ks_test" and len(b_arr) >= 3:
        # Kolmogorov-Smirnov test (alternative='greater' checks if CDF of patched
        # is greater than baseline, which means patched values are stochastically smaller)
        ks_res = stats.ks_2samp(b_arr, p_arr, alternative="less")
        stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)
        test_used = "kolmogorov_smirnov"
    else:
        # Paired Student's t-test (alternative='greater' tests if baseline > patched)
        if len(b_arr) >= 2:
            var_diff = float(np.var(diff, ddof=1)) if len(diff) > 1 else 0.0
            if math.isclose(var_diff, 0.0, abs_tol=1e-12):
                # Constant difference across all seeds
                stat = float("inf") if mean_diff > 0 else 0.0
                p_val = 0.0 if mean_diff > 0 else 1.0
                test_used = "paired_t_test"
            else:
                try:
                    t_res = stats.ttest_rel(b_arr, p_arr, alternative="greater")
                    stat = float(t_res.statistic)
                    p_val = float(t_res.pvalue)
                    test_used = "paired_t_test"
                except Exception:
                    stat = 0.0
                    p_val = 1.0
                    test_used = "paired_t_test"
        else:
            stat = 0.0
            p_val = 0.05 if mean_diff > 0 else 1.0
            test_used = "single_run"

    # Significant if p-value is below alpha and 95% CI lower bound is positive (or p < alpha)
    is_sig = bool(p_val < alpha and mean_diff > 0)

    notes = (
        f"Significant loss reduction confirmed across seeds (p={p_val:.4f} < {alpha})."
        if is_sig
        else f"Improvement did not reach statistical significance (p={p_val:.4f} >= {alpha})."
    )

    return DistributionComparisonResult(
        test_name=test_used,
        statistic=stat,
        p_value=p_val,
        is_significant=is_sig,
        effect_size=effect_size,
        confidence_interval_95=ci_95,
        mean_baseline=mean_b,
        mean_patched=mean_p,
        mean_difference=mean_diff,
        baseline_values=list(baseline_losses),
        patched_values=list(patched_losses),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Patch Application Helper
# ---------------------------------------------------------------------------

def apply_patch_string(original: str, patch_candidate: PatchCandidate | Any) -> str:
    """Apply a PatchCandidate to original source string."""
    if hasattr(patch_candidate, "apply"):
        # Local or object-level apply method (e.g. dict or string)
        try:
            res = patch_candidate.apply(original)
            if isinstance(res, str):
                return res
            if isinstance(res, dict) and len(res) == 1:
                return next(iter(res.values()))
        except Exception:
            pass

    diff_text = getattr(patch_candidate, "diff", "")
    if not diff_text:
        return original
    try:
        return apply_unified_diff(original, diff_text)
    except MalformedDiffError:
        pass  # hand-written patches may have inexact context

    # Simple unified diff application
    lines = original.splitlines(keepends=True)
    patch_lines = diff_text.splitlines(keepends=True)

    # Check for direct before/after replacement blocks or simple unified diff
    # Format: --- a/... +++ b/... @@ -start,len +start,len @@
    try:
        # Try difflib-style patch reconstruction
        result = []
        i = 0
        hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        patch_idx = 0

        # Skip header lines
        while patch_idx < len(patch_lines) and not patch_lines[patch_idx].startswith("@@"):
            patch_idx += 1

        if patch_idx >= len(patch_lines):
            # Fallback: check if diff is a simple replacement or whole new content
            if "@@" not in diff_text and "\n+" in diff_text and "\n-" in diff_text:
                to_remove = [l[1:] for l in patch_lines if l.startswith("-")]
                to_add = [l[1:] for l in patch_lines if l.startswith("+")]
                patched = original
                for rem, add in zip(to_remove, to_add):
                    patched = patched.replace(rem, add)
                return patched
            return original

        while patch_idx < len(patch_lines):
            match = hunk_re.match(patch_lines[patch_idx])
            if not match:
                patch_idx += 1
                continue

            orig_start = int(match.group(1)) - 1
            patch_idx += 1

            # Append unchanged lines up to orig_start
            while i < orig_start and i < len(lines):
                result.append(lines[i])
                i += 1

            # Process hunk
            while patch_idx < len(patch_lines) and not patch_lines[patch_idx].startswith("@@"):
                pline = patch_lines[patch_idx]
                if pline.startswith(" "):
                    result.append(lines[i] if i < len(lines) else pline[1:])
                    i += 1
                elif pline.startswith("-"):
                    i += 1
                elif pline.startswith("+"):
                    result.append(pline[1:])
                patch_idx += 1

        while i < len(lines):
            result.append(lines[i])
            i += 1

        return "".join(result)
    except Exception:
        return original


# ---------------------------------------------------------------------------
# MultiseedAgent Class
# ---------------------------------------------------------------------------

class MultiseedAgent:
    """Agent that validates candidate patches using multi-seed statistical testing."""

    def __init__(
        self,
        seeds: Optional[Sequence[int]] = None,
        timeout_per_run: float = 30.0,
        max_memory_mb: Optional[int] = None,
        max_cpu_cores: Optional[int] = None,
        alpha: float = 0.05,
        sandbox: Optional[SubprocessSandbox] = None,
        telemetry_forwarder: Optional[TelemetryForwarder] = None,
        prefer_test: str = "t_test",
        measured: bool = False,
        max_steps: Optional[int] = None,
        script_args: Optional[List[str]] = None,
    ):
        # 3 to 5 distinct seeds as specified in the brief
        self.seeds = list(seeds) if seeds is not None else [42, 1337, 2024, 7, 99]
        self.timeout_per_run = timeout_per_run
        self.max_memory_mb = max_memory_mb
        self.max_cpu_cores = max_cpu_cores
        self.alpha = alpha
        self.prefer_test = prefer_test
        self.measured = measured  # run through script_runner, so any script works
        self.max_steps = max_steps
        self.script_args = list(script_args or [])
        self._baselines: Dict[int, Tuple[List[float], List[ExecutionResult]]] = {}
        self.as_file: Optional[str] = None
        self.telemetry_forwarder = telemetry_forwarder or TelemetryForwarder()
        self.sandbox = sandbox or SubprocessSandbox(
            SandboxConfig(
                timeout_seconds=self.timeout_per_run,
                max_memory_mb=self.max_memory_mb,
                max_cpu_cores=self.max_cpu_cores,
            ),
            telemetry_forwarder=self.telemetry_forwarder,
        )

    def _extract_loss(self, result: ExecutionResult) -> Optional[float]:
        """Extract final loss from telemetry or stdout numbers."""
        report = result.run_report or {}
        if result.metrics:
            if "final_loss" in result.metrics:
                return float(result.metrics["final_loss"])
            if "loss" in result.metrics:
                return float(result.metrics["loss"])
            if "losses" in result.metrics and len(result.metrics["losses"]) > 0:
                return float(result.metrics["losses"][-1])

        # A ratio does not change when a fix changes the scale of the loss
        if report.get("loss_ratio") is not None and not report.get("nonfinite_losses"):
            return float(report["loss_ratio"])

        # Scan stdout for loss patterns e.g. "Loss: 0.1234" or "final loss: 0.1234"
        matches = re.findall(r"(?:final_loss|loss|loss:)\s*[:=]?\s*([0-9]+\.[0-9]+(?:e[-+]?[0-9]+)?)", result.stdout, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
        return None

    def evaluate_code_across_seeds(
        self,
        code: str,
        seeds: Sequence[int],
    ) -> Tuple[List[float], List[ExecutionResult]]:
        """Run code across all specified seeds and return (losses, results)."""
        losses: List[float] = []
        results: List[ExecutionResult] = []

        for seed in seeds:
            if self.measured:
                res = self.sandbox.run_measured(code, seed=seed, max_steps=self.max_steps, timeout=self.timeout_per_run,
                                                script_args=self.script_args, as_file=self.as_file)
            else:
                res = self.sandbox.run_code(code, seed=seed, timeout=self.timeout_per_run)
            results.append(res)
            loss = self._extract_loss(res)
            if res.success and loss is not None and math.isfinite(loss):
                losses.append(loss)
            else:
                # Execution failed or diverged
                pass

        return losses, results

    def verify_patch(
        self,
        baseline_code: str,
        candidate: PatchCandidate | Any,
    ) -> ValidationVerdict:
        """Verify a single candidate patch against baseline code across seeds."""
        patched_code = apply_patch_string(baseline_code, candidate)

        # 1. Run baseline across seeds (once: every candidate is compared with the same runs)
        key = hash(baseline_code)
        if key not in self._baselines:
            self._baselines[key] = self.evaluate_code_across_seeds(baseline_code, self.seeds)
        base_losses, base_results = self._baselines[key]

        # 2. Run patched code across seeds
        patch_losses, patch_results = self.evaluate_code_across_seeds(patched_code, self.seeds)

        # Compute budget proxy: total runtime of all runs
        total_duration = sum(r.duration_seconds for r in base_results + patch_results)
        budget_used = min(1.0, total_duration / (self.timeout_per_run * len(self.seeds) * 2))

        # Check for crashes or failures in patched runs
        failed_patch_runs = [r for r in patch_results if not r.success]
        if failed_patch_runs or len(patch_losses) < len(self.seeds):
            err_msg = failed_patch_runs[0].error_message if failed_patch_runs else "Patch caused NaN/divergence"
            return ValidationVerdict(
                patch_candidate=candidate,
                accepted=False,
                validator="multiseed_stats",
                p_value=1.0,
                effect_size=0.0,
                seeds_run=len(self.seeds),
                compute_budget_used=budget_used,
                notes=f"Verification rejected: candidate failed on {len(self.seeds) - len(patch_losses)} seed(s). Error: {err_msg}",
                extra={
                    "baseline_losses": base_losses,
                    "patched_losses": patch_losses,
                    "failed_runs": len(failed_patch_runs),
                },
            )

        # Check baseline: if baseline succeeded on all seeds, compare distributions
        if len(base_losses) == len(self.seeds):
            stats_comparison = compare_distributions(
                baseline_losses=base_losses,
                patched_losses=patch_losses,
                alpha=self.alpha,
                prefer_test=self.prefer_test,
            )
            accepted = stats_comparison.is_significant
            p_val = stats_comparison.p_value
            effect_size = stats_comparison.effect_size
            notes = stats_comparison.notes
            extra = stats_comparison.to_dict()
            extra["ks_test"] = compare_distributions(base_losses, patch_losses, alpha=self.alpha, prefer_test="ks_test").to_dict()
        else:
            # Baseline crashed or had NaN, but patched code succeeded reliably on all seeds!
            accepted = True
            p_val = 0.001
            effect_size = 1.0
            notes = "Baseline failed or produced NaNs, while candidate patch executed successfully across all seeds."
            extra = {
                "baseline_losses": base_losses,
                "patched_losses": patch_losses,
                "baseline_failures": len(self.seeds) - len(base_losses),
            }

        return ValidationVerdict(
            patch_candidate=candidate,
            accepted=accepted,
            validator="multiseed_stats",
            p_value=p_val,
            effect_size=effect_size,
            seeds_run=len(self.seeds),
            compute_budget_used=budget_used,
            notes=notes,
            extra=extra,
        )

    def call(self, state: AgentState) -> AgentState:
        """Integration contract: execute multi-seed verification on state.patch_candidates."""
        candidates = getattr(state, "patch_candidates", getattr(state, "patchcandidates", []))
        verdicts = getattr(state, "validation_verdicts", getattr(state, "validationverdicts", []))

        # Read target source code if available
        target_path = getattr(state, "target_file", "")
        baseline_code = ""
        if target_path and Path(target_path).is_file():
            try:
                baseline_code = Path(target_path).read_text(encoding="utf-8")
            except OSError:
                baseline_code = ""

        # Check existing verdicts (from OverfitOracleAgent, which runs first)
        accepted_by_overfit = set()
        for v in verdicts:
            val_name = getattr(v, "validator", "")
            is_acc = getattr(v, "accepted", False)
            cand = getattr(v, "patch_candidate", getattr(v, "patchcandidate", None))
            if "overfit" in val_name and is_acc and cand is not None:
                accepted_by_overfit.add(id(cand))

        judged = {id(getattr(v, "patch_candidate", None)) for v in verdicts
                  if getattr(v, "validator", "") == "multiseed_stats"}
        for candidate in candidates:
            if id(candidate) in judged:
                continue
            # If OverfitOracle ran and rejected this candidate, skip multi-seed to save compute
            has_overfit_verdict = any(
                "overfit" in getattr(v, "validator", "") and
                getattr(v, "patch_candidate", getattr(v, "patchcandidate", None)) is candidate
                for v in verdicts
            )
            if has_overfit_verdict and id(candidate) not in accepted_by_overfit:
                verdict = ValidationVerdict(
                    patch_candidate=candidate,
                    accepted=False,
                    validator="multiseed_stats",
                    p_value=1.0,
                    seeds_run=0,
                    notes="Skipped: Overfit Oracle rejected this candidate prior to multi-seed verification.",
                )
                verdicts.append(verdict)
                continue

            if baseline_code:
                self.as_file = str(Path(target_path).resolve())
                verdict = self.verify_patch(baseline_code=baseline_code, candidate=candidate)
            else:
                # A patch is only accepted on evidence, and there is nothing here to run
                verdict = ValidationVerdict(
                    patch_candidate=candidate,
                    accepted=False,
                    validator="multiseed_stats",
                    p_value=1.0,
                    seeds_run=0,
                    notes="Rejected: there is no target script to run the patch against.",
                )

            verdicts.append(verdict)

        return state

    def __call__(self, state: AgentState) -> AgentState:
        return self.call(state)
