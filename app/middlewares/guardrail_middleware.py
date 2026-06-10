from typing import Any

from app.guardrails.answer_rewriter import rewrite_answer
from app.guardrails.answer_validator import validate_answer


def extract_allowed_terms_from_messages(messages: list[dict]) -> list[str]:
    """
    从最近一次 retrieve_tcm_knowledge 的 tool message 中解析 allowed_terms。
    """
    for msg in reversed(messages):
        if msg.get("type") == "tool" and msg.get("name") == "retrieve_tcm_knowledge":
            content = msg.get("content", "")

            if not isinstance(content, str):
                continue

            lines = content.splitlines()
            collecting = False
            terms = []

            for line in lines:
                text = line.strip()

                if text.startswith("允许使用的专业术语"):
                    collecting = True
                    continue

                if collecting and text.startswith("回答约束"):
                    break

                if collecting and text.startswith("-"):
                    term = text.replace("-", "", 1).strip()

                    if term:
                        terms.append(term)

            if terms:
                return list(dict.fromkeys(terms))

    return []


def extract_latest_retrieval_evidence(messages: list[dict]) -> str:
    """
    提取最近一次 retrieve_tcm_knowledge 的完整工具返回内容。
    """
    for msg in reversed(messages):
        if msg.get("type") == "tool" and msg.get("name") == "retrieve_tcm_knowledge":
            content = msg.get("content", "")

            if isinstance(content, str):
                return content

    return ""


async def apply_guardrails(
    final_text: str,
    messages: list[dict],
) -> dict[str, Any]:
    """
    对最终答案应用 V0.8 Guardrails。

    返回：
    - final_text
    - validation
    - validation_before_rewrite
    - rewritten
    - allowed_terms
    """

    allowed_terms = extract_allowed_terms_from_messages(messages)
    evidence_text = extract_latest_retrieval_evidence(messages)

    validation_before = validate_answer(
        answer=final_text,
        allowed_terms=allowed_terms,
    )

    rewritten = False
    validation_after = validation_before

    if final_text and allowed_terms and not validation_before.get("passed"):
        unsupported_terms = validation_before.get("unsupported_terms", [])

        rewritten_text = await rewrite_answer(
            answer=final_text,
            allowed_terms=allowed_terms,
            unsupported_terms=unsupported_terms,
            evidence_text=evidence_text,
        )

        if rewritten_text:
            rewritten = True
            final_text = rewritten_text

            validation_after = validate_answer(
                answer=final_text,
                allowed_terms=allowed_terms,
            )

    return {
        "final_text": final_text,
        "validation": validation_after,
        "validation_before_rewrite": validation_before,
        "rewritten": rewritten,
        "allowed_terms": allowed_terms,
    }