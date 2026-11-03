"""Agent 5: writes a candidate patch for the top diagnosed bug.

With a key it runs a ReAct-style repair: the model reads the diagnosis, remembered
repairs and the reason earlier attempts were rejected, then proposes exact edits.
Without one, or if the model's patch is unusable, it falls back to the rule-based
recipes. The target file is never edited here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..learning.bandit import ContextualBandit
from ..repair.patch_utils import generate_minimal_diff, patch_guard
from ..repair.recipes import recipe_for
from ..schemas import AgentState, DiagnosticHypothesis, PatchCandidate, RepairStrategy
from ..semantic.client import LLMClient, LLMError
from ..taxonomy.loader import load_taxonomy
from . import prompts

MAX_ATTEMPTS_PER_BUG = 2
MAX_CHANGED_LINES = 40


class FixSynthesizerAgent:
    """Create a patch candidate for the best hypothesis that still has attempts left.

    An explicit edit can also be given in hypothesis.extra["repair_spec"]:
        {"search": "exact old text", "replacement": "new text"}
    """

    def __init__(self, bandit: Optional[ContextualBandit] = None, client: Optional[LLMClient] = None,
                 memory: Any = None) -> None:
        self.bandit = bandit or ContextualBandit()
        self.client = client
        self.memory = memory
        self._fix_guides: Dict[str, str] = {}

    def _typical_fix(self, bug_id: str) -> str:
        if not self._fix_guides:
            self._fix_guides = {p.id: p.typical_fix for p in load_taxonomy()}
        return self._fix_guides.get(bug_id, "")

    @staticmethod
    def _next_hypothesis(state: AgentState) -> Optional[DiagnosticHypothesis]:
        attempts = state.extra.get("attempts_by_bug", {})
        remaining = [h for h in state.hypotheses if attempts.get(h.bug_id, 0) < MAX_ATTEMPTS_PER_BUG]
        return max(remaining, key=lambda item: item.confidence) if remaining else None

    def run(self, state: AgentState) -> AgentState:
        if not state.target_file:
            state.extra["repair_error"] = "No target_file was supplied."
            return state
        if not state.hypotheses:
            state.extra["repair_error"] = "No diagnostic hypotheses are available."
            return state
        hypothesis = self._next_hypothesis(state)
        if hypothesis is None:
            state.extra["repair_error"] = "Every diagnosed bug has used up its repair attempts."
            return state
        try:
            original = Path(state.target_file).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            state.extra["repair_error"] = "Could not read target file: " + str(exc)
            return state

        state.extra.setdefault("attempts_by_bug", {})
        state.extra["attempts_by_bug"][hypothesis.bug_id] = state.extra["attempts_by_bug"].get(hypothesis.bug_id, 0) + 1

        if "repair_spec" in hypothesis.extra:
            built = self._from_spec(state, hypothesis, original)
        else:  # the model first; a refused or repeated model patch falls back to the rules in the same attempt
            built = None
            for maker in (self._from_llm, self._from_recipe):
                built = maker(state, hypothesis, original)
                if built is None:
                    continue
                problem = self._problem(state, original, built[0])
                if problem is None:
                    break
                state.extra.setdefault("repair_feedback", []).append(
                    {"bug_id": hypothesis.bug_id, "diff": generate_minimal_diff(original, built[0], state.target_file),
                     "reason": problem})
                state.extra.setdefault("llm_notes", []).append(f"fix: {built[2]} patch refused ({problem})")
                built = None
        if built is None:
            state.extra.setdefault("repair_error", f"No patch generated for {hypothesis.bug_id}: "
                                                   "no LLM is available and no repair recipe applies.")
            return state
        updated, strategy, source, confidence, note, extra = built
        self._add_candidate(state, hypothesis, original, updated, strategy, source, confidence, note, extra)
        return state

    def _from_spec(self, state: AgentState, hypothesis: DiagnosticHypothesis, original: str) -> Optional[Tuple]:
        spec = hypothesis.extra.get("repair_spec", {})
        if not isinstance(spec, dict):
            state.extra["repair_error"] = "repair_spec must be a dictionary."
            return None
        search, replacement = spec.get("search"), spec.get("replacement")
        if not isinstance(search, str) or not search:
            state.extra["repair_error"] = ("No patch generated: hypothesis.extra['repair_spec']['search'] "
                                           "must contain the exact source text to replace.")
            return None
        if not isinstance(replacement, str):
            state.extra["repair_error"] = "No patch generated: repair_spec['replacement'] must be a string."
            return None
        occurrences = original.count(search)
        if occurrences != 1:
            state.extra["repair_error"] = ("Expected the search text exactly once; found " + str(occurrences)
                                           + ". Refusing an ambiguous edit.")
            return None
        updated = original.replace(search, replacement, 1)
        if updated == original:
            state.extra["repair_error"] = "The proposed replacement makes no change."
            return None
        try:
            strategy = RepairStrategy(spec.get("strategy", hypothesis.suggested_strategy.value))
        except ValueError:
            strategy = RepairStrategy.UNKNOWN
        if strategy == RepairStrategy.UNKNOWN:
            strategy = self.bandit.select_strategy(hypothesis, state.extra)
        return updated, strategy, "rules", hypothesis.confidence, hypothesis.description, {"search": search, "replacement": replacement}

    def _from_recipe(self, state: AgentState, hypothesis: DiagnosticHypothesis, original: str) -> Optional[Tuple]:
        names = [hypothesis.bug_id] + list(hypothesis.extra.get("alternatives", []))
        for name in names:
            recipe = recipe_for(name, original)
            if recipe is not None and self._digest(recipe.source) not in state.extra.get("tried_patches", []):
                return recipe.source, recipe.strategy, "rules", hypothesis.confidence, recipe.note, {"recipe": name}
        return None

    def _from_llm(self, state: AgentState, hypothesis: DiagnosticHypothesis, original: str) -> Optional[Tuple]:
        if self.client is None:
            return None
        strategy = self.bandit.select_strategy(hypothesis, state.extra)
        exemplars = self.memory.recall(f"{hypothesis.bug_id} {hypothesis.description}", k=2) if self.memory else []
        lessons = self.memory.active_lessons("fix") if self.memory else []
        feedback = [f for f in state.extra.get("repair_feedback", []) if f.get("bug_id") == hypothesis.bug_id]
        lines = [hypothesis.extra.get("line") or 0]
        prompt = prompts.fix_prompt(hypothesis, self._typical_fix(hypothesis.bug_id), strategy.value,
                                    prompts.numbered_source(original, lines), exemplars, feedback[-2:], lessons)
        try:
            reply = self.client.complete_json(prompt, prompts.FIX_SYSTEM, schema=prompts.FIX_SCHEMA, schema_name="repair")
        except LLMError as exc:
            state.extra.setdefault("llm_notes", []).append(f"fix: no usable reply, using recipes ({exc})")
            return None
        updated, problem = self._apply_edits(original, reply.get("edits") if isinstance(reply, dict) else None)
        if updated is None:
            state.extra.setdefault("repair_feedback", []).append({"bug_id": hypothesis.bug_id, "diff": "", "reason": problem})
            state.extra.setdefault("llm_notes", []).append(f"fix: model patch refused ({problem})")
            return None
        try:
            chosen = RepairStrategy(reply.get("strategy", strategy.value))
        except ValueError:
            chosen = strategy
        try:
            confidence = max(0.0, min(1.0, float(reply.get("confidence", hypothesis.confidence))))
        except (TypeError, ValueError):
            confidence = hypothesis.confidence
        extra = {"exemplar_ids": [e.get("id") for e in exemplars], "thought": str(reply.get("thought", ""))[:800]}
        return updated, chosen, "llm" if not exemplars else "hybrid", confidence, str(reply.get("explanation", ""))[:300], extra

    @staticmethod
    def _apply_edits(original: str, edits: Any) -> Tuple[Optional[str], str]:
        if not isinstance(edits, list) or not edits:
            return None, "the reply contained no edits"
        updated = original
        for edit in edits:
            if not isinstance(edit, dict) or not isinstance(edit.get("search"), str) \
                    or not isinstance(edit.get("replacement"), str) or not edit["search"]:
                return None, "an edit was not in the required form"
            count = updated.count(edit["search"])
            if count != 1:
                return None, f"search text found {count} times, it must appear exactly once: {edit['search'][:70]!r}"
            updated = updated.replace(edit["search"], edit["replacement"], 1)
        return (None, "the edits changed nothing") if updated == original else (updated, "")

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _problem(self, state: AgentState, original: str, updated: str) -> Optional[str]:
        problem = patch_guard(original, updated)
        if problem is None and self._digest(updated) in state.extra.get("tried_patches", []):
            problem = "this exact patch was already tried and rejected"
        return problem

    def _add_candidate(self, state: AgentState, hypothesis: DiagnosticHypothesis, original: str, updated: str,
                       strategy: RepairStrategy, source: str, confidence: float, note: str, extra: Dict[str, Any]) -> None:
        problem = self._problem(state, original, updated)
        digest = self._digest(updated)
        diff = generate_minimal_diff(original, updated, state.target_file)
        changed = sum(1 for line in diff.splitlines()
                      if (line.startswith("+") and not line.startswith("+++"))
                      or (line.startswith("-") and not line.startswith("---")))
        if problem is None and changed > MAX_CHANGED_LINES:
            problem = f"the patch changes {changed} lines, more than a minimal repair should"
        if problem is not None:
            state.extra["repair_error"] = "Patch refused: " + problem
            state.extra.setdefault("repair_feedback", []).append({"bug_id": hypothesis.bug_id, "diff": diff, "reason": problem})
            return
        state.extra.setdefault("tried_patches", []).append(digest)
        candidate = PatchCandidate(
            hypothesis_bug_id=hypothesis.bug_id, diff=diff, strategy=strategy, source=source,
            confidence=max(0.0, min(1.0, float(confidence))), changed_files=[Path(state.target_file).name],
            changed_lines=changed,
            extra={"status": "candidate", "rationale": note or hypothesis.description, "validated": False, **extra},
        )
        state.patch_candidates.append(candidate)
        state.extra.pop("repair_error", None)
        state.extra["last_repair_candidate"] = candidate.to_dict()

    # Alias for agent frameworks that call agents with state.
    def __call__(self, state: AgentState) -> AgentState:
        return self.run(state)
