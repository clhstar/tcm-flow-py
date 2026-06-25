from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def append_trace(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [*(left or []), *(right or [])]


class WorkflowState(TypedDict, total=False):
    run_id: str
    user_text: str
    conversation: list[dict[str, Any]]
    messages: Annotated[list[BaseMessage], add_messages]
    inquiry: dict[str, Any]
    evidence: dict[str, Any]
    syndrome: dict[str, Any]
    answer: dict[str, Any]
    safety: dict[str, Any]
    needs_clarification: bool
    final_text: str
    agent_trace: Annotated[list[dict[str, Any]], append_trace]
