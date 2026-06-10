import json
from typing import Any


def extract_agent_trace_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    从 messages 中提取 agent_trace。

    当前 V1.0 主要解析 task 工具返回的 subagent_result。
    """
    trace: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("type") != "tool":
            continue

        if msg.get("name") != "task":
            continue

        content = msg.get("content", "")

        if not isinstance(content, str):
            continue

        try:
            payload = json.loads(content)
        except Exception:
            continue

        if payload.get("type") != "subagent_result":
            continue

        trace.append(
            {
                "agent": payload.get("agent_name", "dynamic_subagent"),
                "action": "complete",
                "task_description": payload.get("task_description", ""),
                "expected_output": payload.get("expected_output", ""),
                "summary": str(payload.get("content", ""))[:300],
            }
        )

    return trace