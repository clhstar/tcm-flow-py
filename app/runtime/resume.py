from typing import Any


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )

    return str(content or "")


def extract_user_text(input_data: dict[str, Any]) -> str:
    for msg in input_data.get("messages", []):
        if msg.get("type") == "human":
            return extract_text_from_content(msg.get("content", "")).strip()

    return ""


def has_pending_clarification(thread_values: dict[str, Any]) -> bool:
    pending = thread_values.get("pending_clarification")

    return isinstance(pending, dict) and bool(pending.get("question"))


def build_resume_input_data(
    thread_values: dict[str, Any],
    input_data: dict[str, Any],
) -> dict[str, Any]:
    """
    将用户本轮输入包装成“回答上一轮澄清问题”的上下文。
    """
    pending = thread_values.get("pending_clarification") or {}

    pending_question = pending.get("question", "")
    required_fields = pending.get("required_fields", [])
    resume_count = int(pending.get("resume_count") or 0)

    user_text = extract_user_text(input_data)

    resume_prompt = f"""
用户正在回答上一轮澄清问题。

上一轮澄清问题：
{pending_question}

需要补充的关键信息：
{", ".join(required_fields) if required_fields else "未明确"}

用户本轮补充：
{user_text}

当前恢复次数：
{resume_count + 1}

请结合用户补充信息、历史对话和可用工具，继续完成中医健康咨询。

要求：
1. 不要重复询问用户已经补充过的信息。
2. 如果仍缺少关键风险信息，可以再次调用 ask_clarification。
3. 如果信息已经足够，请继续进行知识检索、必要的 task 委派、答案校验和最终回答。
4. 最终回答不能直接下诊断，不能开处方。
""".strip()

    return {
        "messages": [
            {
                "type": "human",
                "content": resume_prompt,
            }
        ]
    }


def build_resume_record(
    thread_values: dict[str, Any],
    user_text: str,
) -> dict[str, Any]:
    pending = thread_values.get("pending_clarification") or {}

    return {
        "was_resume": True,
        "previous_clarification": pending,
        "user_reply": user_text,
    }


def increment_resume_count(
    pending_clarification: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(pending_clarification, dict):
        return pending_clarification

    copied = dict(pending_clarification)
    copied["resume_count"] = int(copied.get("resume_count") or 0) + 1

    return copied