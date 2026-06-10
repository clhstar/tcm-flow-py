from typing import Any, TypedDict


class LeadAgentState(TypedDict, total=False):
    """
    Lead Agent 状态定义。

    当前 LangGraph create_agent 内部已经维护 messages。
    这里先作为后续扩展占位：
    - agent_trace
    - retrieval_context
    - artifacts
    - validation
    """

    messages: list[dict[str, Any]]
    agent_trace: list[dict[str, Any]]
    retrieval_context: dict[str, Any]
    artifacts: list[dict[str, Any]]
    validation: dict[str, Any]