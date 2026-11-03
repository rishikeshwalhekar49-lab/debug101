"""Autonomous repair orchestration with temporary-copy validation."""

from __future__ import annotations

from typing import Any, Dict

from ..schemas import AgentState, ValidationVerdict
from ..learning.bandit import ContextualBandit
from .patch_utils import apply_diff_in_temp


class AutonomousRepairLoop:
    """Synthesize, validate, and record a candidate without editing the source."""

    def __init__(self, synthesizer: Any, bandit: ContextualBandit = None) -> None:
        self.synthesizer = synthesizer
        self.bandit = bandit or ContextualBandit()

    def run(self, state: AgentState) -> AgentState:
        if state.should_abort():
            state.status = state.status.__class__.ABORTED
            state.abort_reason = state.abort_reason or "Repair iteration limit reached."
            return state

        before = len(state.patch_candidates)
        self.synthesizer.run(state)
        state.iteration += 1

        if len(state.patch_candidates) == before:
            state.extra["repair_loop_result"] = {
                "accepted": False,
                "reason": state.extra.get("repair_error", "No candidate generated."),
            }
            return state

        candidate = state.patch_candidates[-1]
        ok, message, updated_text = apply_diff_in_temp(state.target_file, candidate.diff)

        verdict = ValidationVerdict(
            patch_candidate=candidate,
            accepted=False,
            validator="temporary_patch_and_syntax_check",
            notes=message,
            extra={
                "patch_applied_to_temp": ok,
                "syntax_checked": state.target_file.lower().endswith(".py"),
                "source_modified": False,
            },
        )
        state.validation_verdicts.append(verdict)

        candidate.extra["validated"] = bool(ok)
        candidate.extra["validation_message"] = message
        candidate.extra["status"] = "structurally_valid" if ok else "rejected"

        # Structural validity is not proof that the ML bug is fixed.
        state.extra["repair_loop_result"] = {
            "accepted": False,
            "structurally_valid": bool(ok),
            "message": message,
            "source_modified": False,
            "requires_behavioral_validation": bool(ok),
        }

        return state

    def __call__(self, state: AgentState) -> AgentState:
        return self.run(state)
