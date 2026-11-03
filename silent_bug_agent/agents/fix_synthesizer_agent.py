"""Minimal, conservative patch synthesis using explicit repair instructions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schemas import AgentState, PatchCandidate, RepairStrategy
from ..repair.patch_utils import generate_minimal_diff
from ..learning.bandit import ContextualBandit


class FixSynthesizerAgent:
    """Create a patch candidate only when the diagnosis gives a concrete edit.

    Expected optional hypothesis.extra["repair_spec"]:
        {"search": "exact old text", "replacement": "new text"}

    An optional "strategy" may contain a RepairStrategy value. The original
    target file is never edited by this class.
    """

    def __init__(self, bandit: Optional[ContextualBandit] = None) -> None:
        self.bandit = bandit or ContextualBandit()

    def run(self, state: AgentState) -> AgentState:
        if not state.target_file:
            state.extra["repair_error"] = "No target_file was supplied."
            return state

        if not state.hypotheses:
            state.extra["repair_error"] = "No diagnostic hypotheses are available."
            return state

        hypothesis = max(state.hypotheses, key=lambda item: item.confidence)
        spec = hypothesis.extra.get("repair_spec", {})

        if not isinstance(spec, dict):
            state.extra["repair_error"] = "repair_spec must be a dictionary."
            return state

        search_text = spec.get("search")
        replacement = spec.get("replacement")

        if not isinstance(search_text, str) or not search_text:
            state.extra["repair_error"] = (
                "No patch generated: hypothesis.extra['repair_spec']['search'] "
                "must contain the exact source text to replace."
            )
            return state

        if not isinstance(replacement, str):
            state.extra["repair_error"] = (
                "No patch generated: repair_spec['replacement'] must be a string."
            )
            return state

        try:
            original = Path(state.target_file).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            state.extra["repair_error"] = "Could not read target file: " + str(exc)
            return state

        occurrences = original.count(search_text)
        if occurrences != 1:
            state.extra["repair_error"] = (
                "Expected the search text exactly once; found " + str(occurrences)
                + ". Refusing an ambiguous edit."
            )
            return state

        updated = original.replace(search_text, replacement, 1)
        if updated == original:
            state.extra["repair_error"] = "The proposed replacement makes no change."
            return state

        try:
            strategy = RepairStrategy(
                spec.get("strategy", hypothesis.suggested_strategy.value)
            )
        except ValueError:
            strategy = RepairStrategy.UNKNOWN

        if strategy == RepairStrategy.UNKNOWN:
            strategy = self.bandit.select_strategy(hypothesis, state.extra)

        diff = generate_minimal_diff(original, updated, state.target_file)
        changed_lines = sum(
            1 for line in diff.splitlines()
            if (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        )

        candidate = PatchCandidate(
            hypothesis_bug_id=hypothesis.bug_id,
            diff=diff,
            strategy=strategy,
            source="rules",
            confidence=max(0.0, min(1.0, float(hypothesis.confidence))),
            changed_files=[Path(state.target_file).name],
            changed_lines=changed_lines,
            extra={
                "status": "candidate",
                "search": search_text,
                "replacement": replacement,
                "rationale": hypothesis.description,
                "validated": False,
            },
        )
        state.patch_candidates.append(candidate)
        state.extra.pop("repair_error", None)
        state.extra["last_repair_candidate"] = candidate.to_dict()
        return state

    # Alias for agent frameworks that call agents with state.
    def __call__(self, state: AgentState) -> AgentState:
        return self.run(state)
