def extract_latest_clarification_question(messages: list[dict]) -> str:
    """
    只在 ask_clarification 工具执行完成后中断。

    不能在 AI tool_calls 阶段中断，否则会导致下一轮 tool_call 缺少 tool message。
    """
    if not messages:
        return ""

    latest = messages[-1]

    if latest.get("type") == "tool" and latest.get("name") == "ask_clarification":
        content = latest.get("content", "")

        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""