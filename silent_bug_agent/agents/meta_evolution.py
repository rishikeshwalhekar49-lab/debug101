"""Agent 7: audits the memory and improves how future repairs are done.

It runs every few pipeline runs and does four things:

1. **Exemplar pruning (ablation).** For each remembered repair, compare the success
   rate of attempts that used it with attempts on the same kind of bug that did not.
   An example that makes results worse is retired.
2. **Bandit priors.** Rebuilds strategy statistics from the full history, so the
   choice of strategy reflects validated outcomes only.
3. **Prompt tuning.** Turns recent rejections into short lessons for the Fix
   Synthesizer, written by an LLM when one is available and by rules otherwise.
4. **Safety check.** If the success rate falls with the current lessons, it goes
   back to the previous version.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..learning.experience_store import ExperienceStore
from ..semantic.client import LLMClient, LLMError
from . import prompts

MIN_USES = 3  # evidence needed before an example is judged
HARM_MARGIN = 0.2  # how much worse than "without it" an example must be
MAX_LESSONS = 5
MAX_LESSON_LENGTH = 220


@dataclass
class MetaReport:
    trajectories: int = 0
    pruned_exemplars: List[str] = field(default_factory=list)
    bandit_contexts: int = 0
    lessons: List[str] = field(default_factory=list)
    lesson_source: Optional[str] = None
    rolled_back: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetaEvolutionAgent:
    def __init__(self, store: ExperienceStore, client: Optional[LLMClient] = None) -> None:
        self.store = store
        self.client = client

    def run(self) -> MetaReport:
        history = self.store.trajectories()
        report = MetaReport(trajectories=len(history))
        if not history:
            report.notes.append("no history yet")
            return report
        report.pruned_exemplars = self._prune(history)
        report.bandit_contexts = self._rebuild_bandit(history)
        report.rolled_back = self._check_lessons()
        if not report.rolled_back:
            lessons, source = self._write_lessons(history, report)
            if lessons and lessons != self.store.active_lessons("fix"):
                self.store.set_lessons("fix", lessons, source)
                report.lessons, report.lesson_source = lessons, source
        return report

    def _prune(self, history: List[Dict[str, Any]]) -> List[str]:
        retired: List[str] = []
        for exemplar in [t for t in history if t["accepted"] and t["active"]]:
            same_kind = [t for t in history if t["category"] == exemplar["category"] and t["id"] != exemplar["id"]]
            used = [t for t in same_kind if exemplar["id"] in t["exemplar_ids"]]
            unused = [t for t in same_kind if exemplar["id"] not in t["exemplar_ids"]]
            if len(used) < MIN_USES or not unused:
                continue
            with_it = sum(t["accepted"] for t in used) / len(used)
            without_it = sum(t["accepted"] for t in unused) / len(unused)
            if with_it + HARM_MARGIN < without_it:
                retired.append(exemplar["id"])
        self.store.prune(retired)
        return retired

    def _rebuild_bandit(self, history: List[Dict[str, Any]]) -> int:
        stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        for item in history:
            if not item["strategy"] or item["reward"] is None:
                continue
            arm = stats.setdefault(item["category"] or "global", {}).setdefault(item["strategy"], {"count": 0.0, "reward_sum": 0.0})
            arm["count"] += 1.0
            arm["reward_sum"] += float(item["reward"])
        self.store.replace_bandit(stats)
        return len(stats)

    def _check_lessons(self) -> bool:
        versions = self.store.lesson_versions("fix")
        if len(versions) < 2:
            return False
        current, previous = versions[-1], versions[-2]
        if not current["active"] or current["uses"] < 5 or previous["uses"] < 5:
            return False
        if current["wins"] / current["uses"] + HARM_MARGIN < previous["wins"] / previous["uses"]:
            return self.store.rollback_lessons("fix")
        return False

    def _write_lessons(self, history: List[Dict[str, Any]], report: MetaReport) -> tuple:
        failures = [t for t in history if not t["accepted"]][-15:]
        if not failures:
            return [], ""
        if self.client is not None:
            facts = [{"bug": t["bug_id"], "strategy": t["strategy"], "why_rejected": (t["notes"] or "")[:200]} for t in failures]
            prompt = (f"Rejected repair attempts:\n{json.dumps(facts)}\n\nWrite at most {MAX_LESSONS} short, general lessons "
                      "for a repair agent, each under 25 words. Give your `reasoning`, then `lessons`.")
            try:
                reply = self.client.complete_json(prompt, prompts.META_SYSTEM, schema=prompts.META_SCHEMA, schema_name="lessons")
                lessons = [str(l).strip()[:MAX_LESSON_LENGTH] for l in reply.get("lessons", []) if str(l).strip()]
                if lessons:
                    return lessons[:MAX_LESSONS], "llm"
            except (LLMError, AttributeError) as exc:
                report.notes.append(f"LLM unavailable, using rule-based lessons ({exc})")
        reasons = Counter((t["notes"] or "").split(":")[0].strip() for t in failures)
        lessons = [f"Repairs were often rejected for this reason: {reason[:140]}." for reason, _ in reasons.most_common(3) if reason]
        return lessons, "rules"
