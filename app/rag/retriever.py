from typing import Any

from app.rag.terms import extract_terms, rewrite_tcm_query, detect_topic
from app.rag.vector_store import get_vector_store


def calculate_keyword_hits(content: str, query: str) -> int:
    """
    计算 query 中的术语在 chunk 里命中了多少。
    """
    query_terms = extract_terms(query)
    content_terms = extract_terms(content)

    return len(set(query_terms) & set(content_terms))


def rerank_docs(
    raw_results: list[tuple[Any, float]],
    original_query: str,
    rewritten_query: str,
) -> list[dict]:
    """
    对向量检索结果做简单规则重排。
    Chroma score 通常是距离，越小越相似。
    这里额外加入 topic_bonus 和 keyword_hits。
    """
    topic = detect_topic(original_query) or detect_topic(rewritten_query)
    reranked = []

    for doc, distance_score in raw_results:
        content = doc.page_content
        metadata = doc.metadata

        keyword_hits = calculate_keyword_hits(content, rewritten_query)

        topic_bonus = 0
        if topic and metadata.get("topic") == topic:
            topic_bonus = 3

        section_bonus = 0
        section = metadata.get("section", "")
        if "问诊" in section:
            section_bonus += 1
        if "调护" in section:
            section_bonus += 1
        if "危险" in section:
            section_bonus += 1

        # distance_score 越小越好，所以这里用负数抵消
        rerank_score = keyword_hits * 2 + topic_bonus + section_bonus - float(distance_score)

        reranked.append(
            {
                "content": content,
                "source": metadata.get("source", "unknown"),
                "filename": metadata.get("filename", "unknown"),
                "topic": metadata.get("topic", "unknown"),
                "section": metadata.get("section", "unknown"),
                "chunk_id": metadata.get("chunk_id"),
                "distance_score": float(distance_score),
                "keyword_hits": keyword_hits,
                "topic_bonus": topic_bonus,
                "section_bonus": section_bonus,
                "rerank_score": rerank_score,
                "terms": extract_terms(content),
            }
        )

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    return reranked


def retrieve_tcm_docs(query: str, k: int = 3, candidate_k: int = 8) -> dict:
    """
    V0.6 检索入口：
    1. query rewrite
    2. vector search
    3. simple rerank
    4. allowed_terms extraction
    """
    vector_store = get_vector_store()

    rewritten_query = rewrite_tcm_query(query)

    raw_results = vector_store.similarity_search_with_score(
        rewritten_query,
        k=candidate_k,
    )

    reranked_results = rerank_docs(
        raw_results=raw_results,
        original_query=query,
        rewritten_query=rewritten_query,
    )

    final_results = reranked_results[:k]

    allowed_terms = []
    for item in final_results:
        allowed_terms.extend(item.get("terms", []))

    # 去重并保持顺序
    seen = set()
    unique_allowed_terms = []
    for term in allowed_terms:
        if term not in seen:
            seen.add(term)
            unique_allowed_terms.append(term)

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "results": final_results,
        "allowed_terms": unique_allowed_terms,
    }


def format_retrieval_results(payload: dict) -> str:
    results = payload.get("results", [])
    allowed_terms = payload.get("allowed_terms", [])

    if not results:
        return "未检索到相关中医知识。"

    lines = [
        "检索到以下中医知识依据：",
        f"原始检索问题：{payload.get('original_query', '')}",
        f"改写后检索问题：{payload.get('rewritten_query', '')}",
    ]

    for index, item in enumerate(results, start=1):
        lines.append(
            f"""
[{index}]
主题：{item["topic"]}
章节：{item["section"]}
内容：
{item["content"]}

来源：{item["filename"]}，chunk_id={item["chunk_id"]}
向量距离：{item["distance_score"]:.4f}
关键词命中数：{item["keyword_hits"]}
重排分数：{item["rerank_score"]:.4f}
""".strip()
        )

    if allowed_terms:
        lines.append("\n允许使用的专业术语：")
        for term in allowed_terms:
            lines.append(f"- {term}")

    lines.append(
        """
回答约束：
- 回答中的中医专业术语应优先来自“允许使用的专业术语”。
- 如果检索结果中没有出现某个证候、病机或治法术语，不要主动引入。
- 检索结果只能作为健康咨询依据，不能作为诊断结论。
""".strip()
    )

    return "\n\n".join(lines)