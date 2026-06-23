from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.agents.lead_agent.state import LeadAgentState
from app.config import get_settings
from app.agents.lead_agent.prompt import SYSTEM_PROMPT
from app.middlewares.clarification_middleware import ClarificationMiddleware
from app.runtime import state as runtime_state
from app.tools.tools import get_available_tools


def build_lead_agent(context: dict[str, Any] | None, checkpointer):
    context = context or {}
    settings = get_settings()

    model = ChatOpenAI(
        model=context.get("model_name") or settings.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=context.get("temperature", 0.3),
        streaming=context.get("streaming", True),
    )

    tools = get_available_tools(context=context)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[ClarificationMiddleware()],
    )

    return agent


def make_lead_agent(context: dict[str, Any] | None = None):
    """
    创建 Lead Agent。

    对齐 DeerFlow 的 lead_agent/agent.py：
    - 读取 context
    - 创建模型
    - 动态加载 tools
    - 应用 system_prompt
    - 绑定 checkpointer
    """

    context = context or {}
    return build_lead_agent(
        context=context,
        checkpointer=runtime_state.state.checkpointer,
    )
