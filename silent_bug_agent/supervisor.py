"""Validation pipeline coordinator: runs the validation agents over existing patch candidates."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .schemas import AgentState, AgentStatus, ValidationVerdict
from .validation.multiseedagent import MultiseedAgent
from .validation.overfit_agent import OverfitOracleAgent


class SupervisorAgent:
    """Runs validation only. The full graph is in `agents/supervisor.py`."""

    def __init__(
        self,
        validation_agents: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ):
        # Support both validation_agents and validationagents parameter names
        agents = validation_agents or kwargs.pop("validationagents", None)
        if kwargs:
            raise TypeError(f"Unknown SupervisorAgent argument(s): {sorted(kwargs)}. "
                            "The full pipeline is silent_bug_agent.agents.supervisor.SupervisorAgent.")
        if agents is None:
            # Default pipeline registration: OverfitOracle first, Multiseed second
            agents = [OverfitOracleAgent(), MultiseedAgent()]
        self.validation_agents = list(agents)

    def run(self, state: AgentState) -> AgentState:
        """Run validation pipeline sequentially across registered agents."""
        state.status = AgentStatus.RUNNING

        for agent in self.validation_agents:
            if state.should_abort():
                break
            if hasattr(agent, "call"):
                state = agent.call(state)
            elif callable(agent):
                state = agent(state)

        # Update status based on verdicts
        if state.accepted_patches:
            state.status = AgentStatus.DONE
        elif state.validation_verdicts:
            state.status = AgentStatus.FAILED

        return state

    def __call__(self, state: AgentState) -> AgentState:
        return self.run(state)
