from collections.abc import Callable
from typing import Any

from app.agents.lead_agent.agent import make_lead_agent


AgentFactory = Callable[[dict[str, Any] | None], Any]


def resolve_agent_factory(assistant_id: str) -> AgentFactory:
    """
    根据 assistant_id 解析 Agent Factory。

    对齐 DeerFlow 的 assistant_id → agent_factory 思路。
    后续可以扩展：
    - lead_agent
    - tcm_agent
    - research_agent
    - subagent
    """

    if assistant_id == "lead_agent":
        return make_lead_agent

    raise ValueError(f"Unknown assistant_id: {assistant_id}")