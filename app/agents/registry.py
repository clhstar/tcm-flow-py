from collections.abc import Callable
from typing import Any


AgentFactory = Callable[[dict[str, Any] | None], Any]


def resolve_agent_factory(assistant_id: str) -> AgentFactory:
    """
    Resolve the Agent Factory for an assistant_id.

    Registered assistant ids:
    - lead_agent
    - workflow_agent
    """

    if assistant_id == "lead_agent":
        from app.agents.lead_agent.agent import make_lead_agent

        return make_lead_agent

    if assistant_id == "workflow_agent":
        from app.agents.workflow_agent.agent import make_workflow_agent

        return make_workflow_agent

    raise ValueError(f"Unknown assistant_id: {assistant_id}")
