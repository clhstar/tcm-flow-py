from typing import Any

from app.middlewares.clarification_controller import normalize_question_items


def _message_content(message: dict[str, Any]) -> str:
    """把存储态 message content 转成前端可直接展示的文本。"""
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return str(content or "")


def _extract_clarification_questions(message: dict[str, Any]) -> list[str]:
    questions: list[Any] = []
    for tool_call in message.get("tool_calls", []):
        if tool_call.get("name") != "ask_clarification":
            continue
        args = tool_call.get("args") or {}
        if isinstance(args, dict) and isinstance(args.get("questions"), list):
            questions.extend(args["questions"])
    return normalize_question_items(questions)


def _merge_clarification_content(content: str, questions: list[str]) -> str:
    lines = [content.strip()] if content.strip() else []
    lines.extend(
        f"{index}. {question}"
        for index, question in enumerate(questions, start=1)
    )
    return "\n".join(lines)


def extract_latest_assistant_message(messages: list[dict[str, Any]]) -> str:
    """提取澄清场景要放入 final 事件的助手文本。"""
    for message in reversed(messages):
        if message.get("type") != "ai":
            continue

        content = _message_content(message)
        questions = _extract_clarification_questions(message)
        if questions:
            merged_content = _merge_clarification_content(content, questions)
            if merged_content:
                return merged_content

        if content.strip() and not message.get("tool_calls"):
            return content

    return ""


def extract_pending_clarification(
    messages: list[dict[str, Any]],
) -> list[str] | None:
    """只从标准 ask_clarification tool_call 中读取待补充问题。"""
    for message in reversed(messages):
        questions = _extract_clarification_questions(message)
        if questions:
            return questions

    return None


def build_chat_response(
    *,
    thread_id: str,
    run_id: str,
    status: str,
    assistant_message: str,
    pending_clarification: list[str] | None = None,
    references: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "run_id": run_id,
        "status": status,
        "assistant_message": assistant_message,
        "pending_clarification": pending_clarification,
        "references": references or [],
    }
