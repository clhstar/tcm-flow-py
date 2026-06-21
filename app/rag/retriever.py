import inspect

from app.config import get_settings
from app.rag.ancient_books.query import detect_chief_symptom, rewrite_query
from app.rag.terms import extract_terms
from app.rag.vector_store import get_configured_retrieval_engine


def collect_allowed_terms(results: list[dict]) -> list[str]:
    terms = []
    for result in results:
        terms.extend(extract_terms(result.get("content", "")))
    return list(dict.fromkeys(terms))


def resolve_retrieval_result(result):
    if inspect.isawaitable(result):
        raise RuntimeError("database RAG retrieval is async; use aretrieve_tcm_docs")
    return result


async def resolve_retrieval_result_async(result):
    if inspect.isawaitable(result):
        return await result
    return result


def _format_payload(
    query: str,
    rewritten_query: str,
    chief_symptom: str | None,
    result: dict,
) -> dict:
    results = result["results"]
    retrieval_mode = f"{result['retrieval_mode']}_parent"
    return {
        "status": result["status"],
        "retrieval_mode": retrieval_mode,
        "degraded": result["degraded"],
        "degraded_reason": result["degraded_reason"],
        "original_query": query,
        "rewritten_query": rewritten_query,
        "chief_symptom": chief_symptom,
        "results": results,
        "allowed_terms": collect_allowed_terms(results),
    }


def retrieve_tcm_docs(
    query: str,
    k: int = 5,
    mode: str = "hybrid",
) -> dict:
    """Retrieve production evidence while preserving the existing public API."""

    chief_symptom = detect_chief_symptom(query)
    rewritten_query = rewrite_query(query)
    engine = get_configured_retrieval_engine(get_settings())
    result = resolve_retrieval_result(
        engine.retrieve(
            rewritten_query,
            chief_symptom=chief_symptom,
            mode=mode,
            top_k=k,
        )
    )
    return _format_payload(query, rewritten_query, chief_symptom, result)


async def aretrieve_tcm_docs(
    query: str,
    k: int = 5,
    mode: str = "hybrid",
) -> dict:
    """Async retrieval path for database-backed engines."""

    chief_symptom = detect_chief_symptom(query)
    rewritten_query = rewrite_query(query)
    engine = get_configured_retrieval_engine(get_settings())
    result = await resolve_retrieval_result_async(
        engine.retrieve(
            rewritten_query,
            chief_symptom=chief_symptom,
            mode=mode,
            top_k=k,
        )
    )
    return _format_payload(query, rewritten_query, chief_symptom, result)


def format_retrieval_results(payload: dict) -> str:
    results = payload.get("results", [])
    lines = [
        f"检索状态：{payload.get('status', 'insufficient_evidence')}",
        f"检索模式：{payload.get('retrieval_mode', 'hybrid_parent')}",
        f"原始检索问题：{payload.get('original_query', '')}",
        f"改写后检索问题：{payload.get('rewritten_query', '')}",
    ]
    if payload.get("degraded"):
        lines.append(f"降级检索：是（{payload.get('degraded_reason') or '模型不可用'}）")
    else:
        lines.append("降级检索：否")

    if not results:
        lines.append("未检索到足够的《景岳全书》证据，请继续补充问诊信息。")
    for item in results:
        location = " / ".join(
            part
            for part in (
                item.get("volume", ""),
                item.get("chapter", ""),
                item.get("section", ""),
            )
            if part
        )
        lines.append(
            "\n".join(
                [
                    f"[{item['citation_id']}]",
                    f"主症：{payload.get('chief_symptom') or '未识别'}",
                    f"证据角色：{item.get('evidence_role', '')}",
                    f"原文：{item.get('content', '')}",
                    f"命中片段：{item.get('matched_child', '')}",
                    f"来源：《{item.get('book_title', '')}》 {location}".rstrip(),
                    (
                        f"证据ID：parent_id={item.get('parent_id', '')}，"
                        f"chunk_id={item.get('chunk_id', '')}"
                    ),
                ]
            )
        )

    allowed_terms = payload.get("allowed_terms", [])
    if allowed_terms:
        lines.append("允许使用的专业术语：")
        lines.extend(f"- {term}" for term in allowed_terms)
    lines.append(
        "回答约束：\n"
        "- 以上古籍证据仅用于健康咨询与问诊信息整理，不等同于诊断。\n"
        "- 只能使用本次检索证据支持的证候和病机术语。\n"
        "- 不得据此推荐方剂、药物、剂量或煎服法。"
    )
    return "\n\n".join(lines)
