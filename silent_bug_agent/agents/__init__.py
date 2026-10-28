"""Specialist agents driven by the Supervisor through ``AgentState`` (see ``silent_bug_agent/schemas.py``).

Import each agent from its own module, e.g.
``from silent_bug_agent.agents.runtime_monitor_agent import RuntimeMonitorAgent``;
nothing is imported here, so loading one agent never pulls in another agent's
dependencies (the runtime monitor needs torch only when it runs).
"""
