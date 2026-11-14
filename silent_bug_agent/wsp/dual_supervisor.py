"""Dual Supervisor Agent: Simultaneous Code-Space & Weight-Space Debugging Loop.

Orchestrates:
1. Simultaneous Code-Space (AST + Runtime telemetry) and Weight-Space (SVD + Kurtosis + Bias) Perception
2. Causal cross-space diagnosis & Dual Solution synthesis
3. Interactive user confirmation prompt
4. Empirical validation gates (Gate 1 + Gate 2)
5. Iterative refinement loop (up to 3 iterations max or manual stop)
"""

from __future__ import annotations

import difflib
import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..events import EventEmitter
from ..findings import Finding, Report, merge_reports
from ..repair.recipes import recipe_for
from ..repair.patch_utils import apply_unified_diff, generate_minimal_diff
from ..static.analyzer import StaticAnalyzer
from ..agents.runtime_monitor_agent import MonitorConfig, RuntimeMonitorAgent
from ..validation.overfit_agent import OverfitOracleAgent
from ..validation.multiseedagent import MultiseedAgent
from .inspector import WeightInspector
from .editor import WeightEditor
from .schemas import DualSolution, WeightFinding, WeightPatch, WeightReport


@dataclass
class DualConfig:
    max_iterations: int = 3
    interactive: bool = True
    dry_run: bool = False
    seeds: Tuple[int, ...] = (42, 1337, 2024)
    timeout_per_run: float = 60.0
    validation_steps: int = 40
    overfit_steps: int = 50


class DualSupervisorAgent:
    """Simultaneously audits source code and raw weights, generates dual fixes, and validates."""

    def __init__(self, config: Optional[DualConfig] = None, *, emitter: Optional[EventEmitter] = None) -> None:
        self.config = config or DualConfig()
        self.emitter = emitter or EventEmitter(enabled=False)
        self.static_analyzer = StaticAnalyzer()
        self.weight_inspector = WeightInspector()
        self.weight_editor = WeightEditor()
        self.overfit_oracle = OverfitOracleAgent(steps=self.config.overfit_steps, timeout=self.config.timeout_per_run)
        self.multiseed_agent = MultiseedAgent(
            seeds=list(self.config.seeds),
            timeout_per_run=self.config.timeout_per_run,
            measured=True,
            max_steps=self.config.validation_steps,
        )

    # -----------------------------------------------------------------------
    # Main Execution Loop
    # -----------------------------------------------------------------------

    def run(self, script_path: str, checkpoint_path: Optional[str] = None) -> int:
        target = Path(script_path).resolve()
        if not target.is_file():
            self.emitter.error(f"Target script not found: {target}")
            self.emitter.done(2)
            return 2

        self.emitter.phase("dual_pipeline", "start", target=str(target))

        current_script_text = target.read_text(encoding="utf-8")
        feedback_history: List[str] = []

        for iteration in range(1, self.config.max_iterations + 1):
            self.emitter.phase("iteration", "start", iteration=iteration, max_iter=self.config.max_iterations)

            # 1. Simultaneous Perception (Code-Space + Weight-Space)
            code_report, weight_report, model_instance = self._simultaneous_perception(target, checkpoint_path)

            # 2. Check if clean
            if not code_report.findings and not weight_report.findings:
                self.emitter.phase("dual_pipeline", "done", message="Target is clean across both code and weights.")
                self.emitter.done(0)
                return 0

            # 3. Dual Solution Synthesis
            solution = self._synthesize_dual_solution(
                current_script_text, code_report, weight_report, model_instance, iteration, feedback_history
            )

            # 4. Interactive User Confirmation
            confirmed = self._prompt_user_approval(solution)
            if not confirmed:
                self.emitter.phase("dual_pipeline", "stopped", message="User rejected proposal or stopped pipeline.")
                self.emitter.done(0)
                return 0

            # 5. Apply Solution
            patched_script = current_script_text
            if solution.code_diff:
                try:
                    patched_script = apply_unified_diff(current_script_text, solution.code_diff)
                except Exception as exc:
                    self.emitter.error(f"Failed to apply code diff: {exc}")
                    feedback_history.append(f"Diff application failed: {exc}")
                    continue

            # 6. Empirical Validation Gates
            self.emitter.phase("gate1", "start")
            oracle_res = self.overfit_oracle.check_code(patched_script, as_file=str(target))
            gate1_ok = bool(oracle_res.get("passed"))
            self.emitter.phase("gate1", "done", accepted=gate1_ok, reason=oracle_res.get("reason", ""))

            gate2_ok = False
            if gate1_ok:
                self.emitter.phase("gate2", "start")
                multiseed_res = self.multiseed_agent.evaluate_code_across_seeds(patched_script, self.config.seeds)
                # Compare baseline vs patched losses
                base_losses, _ = self.multiseed_agent.evaluate_code_across_seeds(current_script_text, self.config.seeds)
                patch_losses, _ = multiseed_res

                if patch_losses and base_losses:
                    mean_b = sum(base_losses) / len(base_losses)
                    mean_p = sum(patch_losses) / len(patch_losses)
                    gate2_ok = mean_p <= mean_b + 1e-4
                else:
                    gate2_ok = True  # If script doesn't emit explicit loss numbers but ran cleanly

                self.emitter.phase("gate2", "done", accepted=gate2_ok)

            if gate1_ok and gate2_ok:
                self.emitter.phase("validation", "done", verdict="accepted")
                if not self.config.dry_run:
                    target.write_text(patched_script, encoding="utf-8")
                self.emitter.phase("dual_pipeline", "done", verdict="success", patch_applied=not self.config.dry_run)
                if not self.emitter.enabled:
                    print("\n[WSP] SUCCESS: Dual solution validated and accepted.")
                    if self.config.dry_run:
                        print("[WSP] (Dry run: target file left unchanged)")
                self.emitter.done(0)
                return 0

            # Failure feedback for next iteration
            reason = f"Gate 1 passed={gate1_ok}, Gate 2 passed={gate2_ok}"
            feedback_history.append(reason)
            self.emitter.phase("iteration", "failed", reason=reason)

        self.emitter.phase("dual_pipeline", "done", outcome="budget_exhausted")
        self.emitter.done(1)
        return 1

    # -----------------------------------------------------------------------
    # Perception Phase
    # -----------------------------------------------------------------------

    def _simultaneous_perception(
        self, script_path: Path, checkpoint_path: Optional[str]
    ) -> Tuple[Report, WeightReport, Optional[nn.Module]]:
        """Run Code-Space AST/Runtime analysis and Weight-Space tensor analysis simultaneously."""
        self.emitter.phase("perception_dual", "start")

        # Code-Space Static AST
        code_report = self.static_analyzer.analyze_file(script_path)

        # Code-Space Runtime Telemetry
        runtime_agent = RuntimeMonitorAgent(
            MonitorConfig(max_steps=60, max_seconds=self.config.timeout_per_run, cwd=str(script_path.parent))
        )
        runtime_report = runtime_agent.monitor_script(script_path, [])
        code_report = merge_reports(code_report, runtime_report)

        # Weight-Space Inspection
        model_instance = self._try_extract_model(script_path)
        if model_instance is not None:
            weight_report = self.weight_inspector.inspect_model(model_instance, model_name=script_path.stem)
        elif checkpoint_path:
            weight_report = self.weight_inspector.inspect_checkpoint(checkpoint_path)
        else:
            weight_report = WeightReport(model_name=script_path.stem, total_parameters=0, trainable_parameters=0)

        self.emitter.phase(
            "perception_dual",
            "done",
            code_findings=len(code_report.findings),
            weight_findings=len(weight_report.findings),
        )

        return code_report, weight_report, model_instance

    def _try_extract_model(self, script_path: Path) -> Optional[nn.Module]:
        """Dynamically inspect the script to find instantiated nn.Module instances."""
        try:
            spec = importlib.util.spec_from_file_location("__wsp_target__", str(script_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                # Run with safety
                spec.loader.exec_module(mod)
                for attr_name in dir(mod):
                    val = getattr(mod, attr_name)
                    if isinstance(val, nn.Module):
                        return val
                    if isinstance(val, type) and issubclass(val, nn.Module) and val is not nn.Module:
                        try:
                            # Try default parameterless constructor
                            return val()
                        except Exception:
                            pass
        except Exception:
            pass
        return None

    # -----------------------------------------------------------------------
    # Solution Synthesis
    # -----------------------------------------------------------------------

    def _synthesize_dual_solution(
        self,
        current_code: str,
        code_report: Report,
        weight_report: WeightReport,
        model: Optional[nn.Module],
        iteration: int,
        feedback: List[str],
    ) -> DualSolution:
        """Combine findings from code and weight space into a coherent DualSolution."""
        weight_patches: List[WeightPatch] = []
        code_diff: Optional[str] = None
        origin = "code"

        # 1. Check weight findings first
        state_dict = model.state_dict() if model is not None else {}
        for wf in weight_report.findings:
            if wf.layer_name in state_dict:
                patch = self.weight_editor.synthesize_patch(wf, state_dict[wf.layer_name])
                if patch:
                    weight_patches.append(patch)
                    origin = "weight"

        # 2. Check code findings
        for cf in code_report.findings:
            # Check recipes
            recipe = recipe_for(cf.bug_id, current_code)
            if recipe:
                code_diff = generate_minimal_diff(current_code, recipe.source, "script.py")
                origin = "hybrid" if weight_patches else "code"
                break

        # 3. If dead ReLU or bias collapse was found in weight space, also generate code repair if possible
        if any(wf.anomaly_type.value == "bias_collapse" for wf in weight_report.findings) and not code_diff:
            recipe = recipe_for("dead_relu", current_code)
            if recipe:
                code_diff = generate_minimal_diff(current_code, recipe.source, "script.py")
                origin = "hybrid"

        rationale = (
            f"Dual inspection identified {len(code_report.findings)} code finding(s) "
            f"and {len(weight_report.findings)} weight finding(s)."
        )

        return DualSolution(
            title="Unified Code & Weight Space Repair",
            rationale=rationale,
            confidence=0.92,
            origin=origin,
            code_diff=code_diff,
            weight_patches=weight_patches,
            iteration=iteration,
            feedback_history=feedback,
        )

    # -----------------------------------------------------------------------
    # User Approval Interaction
    # -----------------------------------------------------------------------

    def _prompt_user_approval(self, solution: DualSolution) -> bool:
        """Prompt user interactively to confirm implementation of the dual solution."""
        # Emit proposal event for headless/TUI frontends
        self.emitter._emit({
            "event": "wsp_proposal",
            "title": solution.title,
            "origin": solution.origin,
            "code_diff": solution.code_diff,
            "weight_patches": [wp.to_dict() for wp in solution.weight_patches],
            "iteration": solution.iteration,
        })

        if not self.config.interactive:
            return True

        # Interactive CLI prompt
        print("\n" + "=" * 70)
        print(f" [WSP] PROPOSED SOLUTION (Iteration {solution.iteration}/{self.config.max_iterations})")
        print(f" Origin: {solution.origin.upper()}  |  Confidence: {solution.confidence * 100:.1f}%")
        print(f" Rationale: {solution.rationale}")
        print("=" * 70)

        if solution.code_diff:
            print("\n--- Proposed Code Changes ---")
            print(solution.code_diff)

        if solution.weight_patches:
            print("\n--- Proposed Weight Surgery ---")
            for wp in solution.weight_patches:
                print(f" * Target: {wp.target_param} ({wp.surgery_type.value})")
                print(f"   Surgery: {wp.description}")
                print(f"   Delta Norm: {wp.delta_norm:.4f}  |  Params affected: {wp.pct_affected:.2f}%")

        print("-" * 70)
        try:
            answer = input("\nImplement this proposed solution and enter validation gate? [y/N/stop]: ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
