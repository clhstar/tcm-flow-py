from typing import Any


def extract_latest_clarification_question(messages: list[dict[str, Any]]) -> str:
    """
    主路径：只在 ask_clarification 工具执行完成后中断。

    """
    if not messages:
        return ""

    latest = messages[-1]

    if latest.get("type") == "tool" and latest.get("name") == "ask_clarification":
        content = latest.get("content", "")

        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""
