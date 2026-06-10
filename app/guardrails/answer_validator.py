from typing import Any

from app.guardrails.term_checker import extract_terms, normalize_terms


def validate_answer_terms(
    answer: str,
    allowed_terms: list[str],
) -> dict[str, Any]:
    """
    校验回答中的中医专业术语是否来自 allowed_terms。

    规则：
    1. 如果没有 allowed_terms，说明本轮可能没有 RAG 检索，先跳过严格校验。
    2. 如果回答中出现了 allowed_terms 之外的专业术语，则判定不通过。
    """
    answer_terms = normalize_terms(extract_terms(answer))
    allowed_terms = normalize_terms(allowed_terms)

    if not allowed_terms:
        return {
            "passed": True,
            "reason": "no_allowed_terms_skip_validation",
            "answer_terms": answer_terms,
            "allowed_terms": allowed_terms,
            "unsupported_terms": [],
        }

    allowed_set = set(allowed_terms)
    unsupported_terms = [
        term for term in answer_terms
        if term not in allowed_set
    ]

    return {
        "passed": len(unsupported_terms) == 0,
        "reason": "ok" if not unsupported_terms else "unsupported_terms_found",
        "answer_terms": answer_terms,
        "allowed_terms": allowed_terms,
        "unsupported_terms": unsupported_terms,
    }


def validate_answer(
    answer: str,
    allowed_terms: list[str],
) -> dict[str, Any]:
    """
    V0.8 总校验入口。
    后面可以继续加：
    - 绝对诊断表达校验
    - 处方剂量校验
    - 危险信号提醒校验
    """
    return validate_answer_terms(
        answer=answer,
        allowed_terms=allowed_terms,
    )