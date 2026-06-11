import json
from typing import Any


def normalize_question_items(
    questions: list[Any],
    max_questions: int = 3,
) -> list[str]:
    """
    去除空白、去重并限制结构化问题数量。
    """
    normalized: list[str] = []
    seen: set[str] = set()

    for question in questions:
        text = str(question).strip()

        if not text or text in seen:
            continue

        seen.add(text)
        normalized.append(text)

        if len(normalized) >= max_questions:
            break

    return normalized


def format_clarification_questions(questions: list[Any]) -> str:
    normalized = normalize_question_items(questions)

    if not normalized:
        return ""

    lines = ["为了更准确地帮您分析，请先补充以下关键信息："]
    lines.extend(
        f"{index}. {question}"
        for index, question in enumerate(normalized, start=1)
    )

    return "\n".join(lines)


def extract_latest_clarification_question(messages: list[dict[str, Any]]) -> str:
    """
    检测需要暂停当前 Run 的澄清请求：
    1. Lead Agent 主动调用 ask_clarification；
    2. task 子 Agent 结构化返回 needs_clarification。
    """
    if not messages:
        return ""

    latest = messages[-1]

    if latest.get("type") == "tool" and latest.get("name") == "ask_clarification":
        content = latest.get("content", "")

        if isinstance(content, str) and content.strip():
            return content.strip()

    if latest.get("type") == "tool" and latest.get("name") == "task":
        content = latest.get("content", "")

        if not isinstance(content, str):
            return ""

        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return ""

        if not payload.get("needs_clarification"):
            return ""

        questions = payload.get("clarification_questions")
        if not isinstance(questions, list):
            return ""

        return format_clarification_questions(questions)

    return ""
