"""Final validation decision, plus the LLM reading of the statistics.

The rule is deterministic. A patch must pass Gate 1, and then either show a
statistically significant improvement, or remove the detected bug without making
results worse (a correct `model.eval()` fix need not change the training loss).
The second route needs a warning or critical finding: an informational hint must
prove itself with statistics, so working code is not "fixed" on a hunch.
The LLM can only turn an acceptance into a rejection, never the other way round.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..agents import prompts
from ..schemas import FINAL_VALIDATOR, PatchCandidate, ValidationVerdict
from ..semantic.client import LLMClient, LLMError

WORSE_TOLERANCE = 0.02


@dataclass
class Decision:
    accepted: bool
    reason: str
    route: str
    llm_verdict: Optional[str] = None


class ValidationAnalyzerAgent:
    def __init__(self, client: Optional[LLMClient] = None) -> None:
        self.client = client

    @staticmethod
    def _not_worse(multiseed: ValidationVerdict) -> bool:
        base, patched = multiseed.extra.get("mean_baseline"), multiseed.extra.get("mean_patched")
        if base is None or patched is None:
            return False
        return patched <= base + WORSE_TOLERANCE * abs(base)

    def rule(self, overfit: ValidationVerdict, multiseed: ValidationVerdict, bug_gone: Optional[bool],
             strong_evidence: bool = True) -> Decision:
        if not overfit.accepted:
            return Decision(False, f"Gate 1 failed: {overfit.notes}", "gate1")
        if bug_gone is False and not multiseed.accepted:
            return Decision(False, "The same bug is still detected after the patch, and results did not improve.", "bug_remains")
        if multiseed.accepted:
            # "statistics" means a significance test was passed; a baseline that could not run is a different route
            route = "baseline_failed" if multiseed.extra.get("baseline_failures") else "statistics"
            return Decision(True, f"Gate 2 passed: {multiseed.notes}", route)
        if bug_gone and strong_evidence and self._not_worse(multiseed):
            return Decision(True, "The detected bug is gone and results did not get worse, "
                                  f"though the improvement was not significant ({multiseed.notes})", "bug_removed")
        return Decision(False, f"Gate 2 failed: {multiseed.notes}", "gate2")

    def analyze(self, candidate: PatchCandidate, overfit: ValidationVerdict, multiseed: ValidationVerdict,
                bug_gone: Optional[bool], strong_evidence: bool = True) -> ValidationVerdict:
        decision = self.rule(overfit, multiseed, bug_gone, strong_evidence)
        if decision.accepted and self.client is not None:
            decision = self._second_opinion(candidate, multiseed, decision)
        return ValidationVerdict(
            patch_candidate=candidate, accepted=decision.accepted, validator=FINAL_VALIDATOR,
            p_value=multiseed.p_value, effect_size=multiseed.effect_size, seeds_run=multiseed.seeds_run,
            compute_budget_used=multiseed.compute_budget_used, notes=decision.reason,
            extra={"route": decision.route, "bug_gone": bug_gone, "llm_verdict": decision.llm_verdict},
        )

    def _second_opinion(self, candidate: PatchCandidate, multiseed: ValidationVerdict, decision: Decision) -> Decision:
        facts: Dict[str, Any] = {
            "bug": candidate.hypothesis_bug_id, "seeds": multiseed.seeds_run, "p_value": multiseed.p_value,
            "cohens_d": multiseed.effect_size,
            "baseline": multiseed.extra.get("baseline_values"), "patched": multiseed.extra.get("patched_values"),
            "ci_95_of_improvement": multiseed.extra.get("confidence_interval_95"), "route": decision.route,
        }
        prompt = (f"Results of repeated training runs (lower metric is better):\n{json.dumps(facts, default=str)}\n\n"
                  "Decide whether this is a true_fix, seed_luck, or unclear, with a one-sentence reason.")
        try:
            reply = self.client.complete_json(prompt, prompts.VALIDATION_SYSTEM, schema=prompts.VALIDATION_SCHEMA,
                                              schema_name="verdict")
        except LLMError as exc:
            decision.reason += f" (LLM second opinion unavailable: {exc})"
            return decision
        verdict = str(reply.get("verdict")) if isinstance(reply, dict) else "unclear"
        decision.llm_verdict = verdict
        if verdict == "seed_luck":
            decision.accepted = False
            decision.route = "llm_veto"
            decision.reason = f"Rejected as seed luck: {str(reply.get('reason', ''))[:300]}"
        return decision
