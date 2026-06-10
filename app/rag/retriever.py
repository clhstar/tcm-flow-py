from typing import Any

from app.rag.bm25_retriever import bm25_search
from app.rag.terms import extract_terms, rewrite_tcm_query, detect_topic
from app.rag.vector_store import get_vector_store


def doc_key(metadata: dict[str, Any]) -> str:
    filename = metadata.get("filename", "unknown")
    chunk_id = metadata.get("chunk_id", "unknown")
    return f"{filename}-{chunk_id}"


def calculate_keyword_hits(content: str, query: str) -> int:
    """
    计算 query 中的中医术语在 chunk 中命中了多少个。
    """
    query_terms = extract_terms(query)
    content_terms = extract_terms(content)

    return len(set(query_terms) & set(content_terms))


def vector_search(query: str, k: int = 8) -> list[dict]:
    """
    Chroma 向量检索。
    distance_score 越小越相似。
    """
    vector_store = get_vector_store()

    raw_results = vector_store.similarity_search_with_score(query, k=k)

    results = []

    for rank, (doc, distance_score) in enumerate(raw_results, start=1):
        metadata = doc.metadata

        results.append(
            {
                "key": doc_key(metadata),
                "content": doc.page_content,
                "source": metadata.get("source", "unknown"),
                "filename": metadata.get("filename", "unknown"),
                "topic": metadata.get("topic", "unknown"),
                "section": metadata.get("section", "unknown"),
                "chunk_id": metadata.get("chunk_id"),
                "distance_score": float(distance_score),
                "vector_rank": rank,
                "terms": extract_terms(doc.page_content),
            }
        )

    return results


def merge_results(
    vector_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """
    合并向量检索和 BM25 检索结果。
    同一个 chunk 如果被两路检索同时命中，会合并为一条。
    """
    merged: dict[str, dict] = {}

    for item in vector_results:
        key = item["key"]
        merged[key] = {
            **item,
            "retrieval_sources": ["vector"],
            "bm25_score": None,
            "bm25_rank": None,
        }

    for item in bm25_results:
        key = item["key"]

        if key in merged:
            merged[key]["retrieval_sources"].append("bm25")
            merged[key]["bm25_score"] = item.get("bm25_score")
            merged[key]["bm25_rank"] = item.get("bm25_rank")

            # 如果 BM25 结果里有 terms，也合并一下
            terms = merged[key].get("terms", []) + item.get("terms", [])
            merged[key]["terms"] = list(dict.fromkeys(terms))
        else:
            merged[key] = {
                **item,
                "retrieval_sources": ["bm25"],
                "distance_score": None,
                "vector_rank": None,
            }

    return list(merged.values())


def calculate_fusion_score(
    item: dict,
    original_query: str,
    rewritten_query: str,
) -> float:
    """
    混合检索融合分数。

    这里使用一个简化版融合策略：
    1. 向量排名越靠前，加分越高
    2. BM25 排名越靠前，加分越高
    3. 两路都命中，加 overlap bonus
    4. 术语命中越多，加分越高
    5. topic 匹配加分
    6. section 类型加分
    """
    topic = detect_topic(original_query) or detect_topic(rewritten_query)

    vector_rank = item.get("vector_rank")
    bm25_rank = item.get("bm25_rank")

    # rank 越小越好，所以用倒数
    vector_rrf = 0.0
    if vector_rank is not None:
        vector_rrf = 1.0 / (60 + vector_rank)

    bm25_rrf = 0.0
    if bm25_rank is not None:
        bm25_rrf = 1.0 / (60 + bm25_rank)

    overlap_bonus = 0
    retrieval_sources = item.get("retrieval_sources", [])
    if "vector" in retrieval_sources and "bm25" in retrieval_sources:
        overlap_bonus = 3

    keyword_hits = calculate_keyword_hits(item.get("content", ""), rewritten_query)

    topic_bonus = 0
    if topic and item.get("topic") == topic:
        topic_bonus = 3

    section_bonus = 0
    section = item.get("section", "")
    if "问诊" in section:
        section_bonus += 1
    if "调护" in section:
        section_bonus += 1
    if "危险" in section:
        section_bonus += 1
    if "概述" in section:
        section_bonus += 0.5

    # 放大 RRF，方便和规则分数在一个量级上
    fusion_score = (
        vector_rrf * 100
        + bm25_rrf * 100
        + overlap_bonus
        + keyword_hits * 2
        + topic_bonus
        + section_bonus
    )

    return fusion_score


def rerank_merged_results(
    merged_results: list[dict],
    original_query: str,
    rewritten_query: str,
) -> list[dict]:
    """
    对融合后的结果进行最终排序。
    """
    reranked = []

    for item in merged_results:
        keyword_hits = calculate_keyword_hits(item.get("content", ""), rewritten_query)
        fusion_score = calculate_fusion_score(
            item=item,
            original_query=original_query,
            rewritten_query=rewritten_query,
        )

        item["keyword_hits"] = keyword_hits
        item["fusion_score"] = fusion_score

        reranked.append(item)

    reranked.sort(key=lambda item: item["fusion_score"], reverse=True)

    return reranked


def collect_allowed_terms(results: list[dict]) -> list[str]:
    allowed_terms = []

    for item in results:
        allowed_terms.extend(item.get("terms", []))

    seen = set()
    unique_terms = []

    for term in allowed_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    return unique_terms


def retrieve_tcm_docs(
    query: str,
    k: int = 3,
    candidate_k: int = 8,
    mode: str = "hybrid",
) -> dict:
    """
    V0.7 检索入口。

    mode:
    - hybrid：向量检索 + BM25 检索
    - vector：只使用向量检索
    - keyword：只使用 BM25 检索
    """
    if mode not in {"hybrid", "vector", "keyword"}:
        mode = "hybrid"

    rewritten_query = rewrite_tcm_query(query)

    vector_results = []
    bm25_results = []

    if mode in {"hybrid", "vector"}:
        vector_results = vector_search(rewritten_query, k=candidate_k)

    if mode in {"hybrid", "keyword"}:
        bm25_results = bm25_search(rewritten_query, k=candidate_k)

    merged_results = merge_results(
        vector_results=vector_results,
        bm25_results=bm25_results,
    )

    reranked_results = rerank_merged_results(
        merged_results=merged_results,
        original_query=query,
        rewritten_query=rewritten_query,
    )

    final_results = reranked_results[:k]

    allowed_terms = collect_allowed_terms(final_results)

    return {
        "retrieval_mode": mode,
        "original_query": query,
        "rewritten_query": rewritten_query,
        "vector_results": vector_results,
        "bm25_results": bm25_results,
        "merged_count": len(merged_results),
        "results": final_results,
        "allowed_terms": allowed_terms,
    }


def format_retrieval_results(payload: dict) -> str:
    results = payload.get("results", [])
    allowed_terms = payload.get("allowed_terms", [])

    if not results:
        return "未检索到相关中医知识。"

    lines = [
        "检索到以下中医知识依据：",
        f"检索模式：{payload.get('retrieval_mode', 'hybrid')}",
        f"原始检索问题：{payload.get('original_query', '')}",
        f"改写后检索问题：{payload.get('rewritten_query', '')}",
        f"向量检索候选数：{len(payload.get('vector_results', []))}",
        f"BM25检索候选数：{len(payload.get('bm25_results', []))}",
        f"融合后候选数：{payload.get('merged_count', 0)}",
    ]

    for index, item in enumerate(results, start=1):
        retrieval_sources = " + ".join(item.get("retrieval_sources", []))

        distance_score = item.get("distance_score")
        distance_text = f"{distance_score:.4f}" if distance_score is not None else "无"

        bm25_score = item.get("bm25_score")
        bm25_text = f"{bm25_score:.4f}" if bm25_score is not None else "无"

        lines.append(
            f"""
            [{index}]
            检索来源：{retrieval_sources}
            主题：{item["topic"]}
            章节：{item["section"]}
            内容：
            {item["content"]}

            来源：{item["filename"]}，chunk_id={item["chunk_id"]}
            向量排名：{item.get("vector_rank")}
            向量距离：{distance_text}
            BM25排名：{item.get("bm25_rank")}
            BM25分数：{bm25_text}
            关键词命中数：{item.get("keyword_hits")}
            融合分数：{item.get("fusion_score"):.4f}
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