"""A lightweight contextual bandit for choosing repair strategies."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

from ..schemas import DiagnosticHypothesis, RepairStrategy


class ContextualBandit:
    """Choose strategies using confidence, history, and an exploration bonus.

    Statistics can be persisted in AgentState.extra["repair_bandit"].
    A reward should be recorded only after a validation outcome is available.
    """

    def __init__(self, exploration: float = 1.0) -> None:
        self.exploration = max(0.0, float(exploration))

    @staticmethod
    def _stats(state_extra: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        existing = state_extra.setdefault("repair_bandit", {})
        if not isinstance(existing, dict):
            existing = {}
            state_extra["repair_bandit"] = existing
        return existing

    def select_strategy(
        self,
        hypothesis: DiagnosticHypothesis,
        state_extra: Dict[str, Any],
        available: Optional[Iterable[RepairStrategy]] = None,
    ) -> RepairStrategy:
        strategies = list(available or [
            RepairStrategy.DIRECT_EDIT,
            RepairStrategy.REFACTOR,
            RepairStrategy.HYPERPARAM,
            RepairStrategy.FRAMEWORK_CONFIG,
        ])
        if not strategies:
            return RepairStrategy.UNKNOWN

        stats = self._stats(state_extra)
        total = sum(float(stats.get(s.value, {}).get("count", 0)) for s in strategies)

        # Prefer the diagnosis recommendation when scores are tied.
        recommended = hypothesis.suggested_strategy
        best_strategy = strategies[0]
        best_score = float("-inf")

        for strategy in strategies:
            item = stats.get(strategy.value, {})
            count = float(item.get("count", 0))
            reward_sum = float(item.get("reward_sum", 0))
            average_reward = reward_sum / count if count else 0.0
            explore = (
                self.exploration * math.sqrt(math.log(total + 1.0) / count)
                if count > 0 else self.exploration + 1.0
            )
            context_bonus = 0.25 if strategy == recommended else 0.0
            score = average_reward + explore + context_bonus

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy

    def record_reward(
        self,
        state_extra: Dict[str, Any],
        strategy: RepairStrategy,
        reward: float,
    ) -> None:
        """Update the chosen strategy using a bounded reward in [-1, 1]."""
        bounded = max(-1.0, min(1.0, float(reward)))
        stats = self._stats(state_extra)
        item = stats.setdefault(strategy.value, {"count": 0.0, "reward_sum": 0.0})
        item["count"] = float(item.get("count", 0)) + 1.0
        item["reward_sum"] = float(item.get("reward_sum", 0)) + bounded
