from langchain.tools import tool

from app.rag.retrieval_log import write_retrieval_log
from app.rag.retriever import aretrieve_tcm_docs, format_retrieval_results


@tool("retrieve_tcm_knowledge")
async def retrieve_tcm_knowledge(query: str, mode: str = "hybrid") -> str:
    """
    当用户询问中医症状、证候、病机、问诊要点或日常调护建议时调用。

    输入 query 应包含用户主诉、主要症状、伴随症状和需要检索的中医概念。

    mode 表示检索模式：
    - hybrid：默认，向量检索 + BM25 关键词检索
    - vector：只使用向量检索
    - keyword：只使用 BM25 关键词检索

    工具会自动进行 query rewrite、混合检索、结果融合、简单重排，并返回相关中医知识依据和允许使用的专业术语。
    本工具只提供知识检索依据，不直接生成诊断结论，也不提供处方剂量。
    """
    if mode not in {"hybrid", "vector", "keyword"}:
        mode = "hybrid"

    payload = await aretrieve_tcm_docs(
        query=query,
        k=5,
        candidate_k=20,
        mode=mode,
    )

    write_retrieval_log(
        {
            "tool": "retrieve_tcm_knowledge",
            "retrieval_mode": payload.get("retrieval_mode"),
            "status": payload.get("status"),
            "degraded": payload.get("degraded"),
            "degraded_reason": payload.get("degraded_reason"),
            "chief_symptom": payload.get("chief_symptom"),
            "original_query": payload.get("original_query"),
            "rewritten_query": payload.get("rewritten_query"),
            "allowed_terms": payload.get("allowed_terms"),
            "final_results": [
                {
                    "citation_id": item.get("citation_id"),
                    "source_type": item.get("source_type"),
                    "book_title": item.get("book_title"),
                    "source_file": item.get("source_file"),
                    "volume": item.get("volume"),
                    "chapter": item.get("chapter"),
                    "section": item.get("section"),
                    "evidence_role": item.get("evidence_role"),
                    "parent_id": item.get("parent_id"),
                    "chunk_id": item.get("chunk_id"),
                    "retrieval_sources": item.get("retrieval_sources"),
                    "dense_rank": item.get("dense_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                }
                for item in payload.get("results", [])
            ],
        }
    )

    return format_retrieval_results(payload)
