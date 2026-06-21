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


def build_visible_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只保留 human 和可展示的 ai 消息，隐藏工具调用细节。"""
    visible: list[dict[str, Any]] = []

    for message in messages:
        msg_type = message.get("type")
        content = _message_content(message)

        if msg_type == "human":
            if not content.strip():
                continue
            payload = {
                "type": "human",
                "content": content,
            }
            if message.get("id"):
                payload["id"] = message["id"]
            visible.append(payload)
            continue

        if msg_type != "ai":
            continue

        questions = _extract_clarification_questions(message)
        if questions:
            merged_content = _merge_clarification_content(content, questions)
            if not merged_content:
                continue
            payload = {
                "type": "ai",
                "content": merged_content,
            }
            if message.get("id"):
                payload["id"] = message["id"]
            visible.append(payload)
            continue

        if not content.strip() or message.get("tool_calls"):
            continue

        payload = {
            "type": "ai",
            "content": content,
        }
        if message.get("id"):
            payload["id"] = message["id"]
        visible.append(payload)

    return visible


def extract_latest_assistant_message(messages: list[dict[str, Any]]) -> str:
    """提取当前可见消息里的最后一条助手回复。"""
    for message in reversed(build_visible_messages(messages)):
        if message.get("type") == "ai" and message.get("content"):
            return str(message["content"])
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
