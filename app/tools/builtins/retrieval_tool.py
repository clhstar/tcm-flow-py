from langchain.tools import tool

from app.rag.retrieval_log import write_retrieval_log
from app.rag.retriever import retrieve_tcm_docs, format_retrieval_results


@tool("retrieve_tcm_knowledge")
def retrieve_tcm_knowledge(query: str, mode: str = "hybrid") -> str:
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

    payload = retrieve_tcm_docs(
        query=query,
        k=3,
        candidate_k=8,
        mode=mode,
    )

    write_retrieval_log(
        {
            "tool": "retrieve_tcm_knowledge",
            "retrieval_mode": payload.get("retrieval_mode"),
            "original_query": payload.get("original_query"),
            "rewritten_query": payload.get("rewritten_query"),
            "allowed_terms": payload.get("allowed_terms"),
            "vector_results": [
                {
                    "filename": item.get("filename"),
                    "topic": item.get("topic"),
                    "section": item.get("section"),
                    "chunk_id": item.get("chunk_id"),
                    "vector_rank": item.get("vector_rank"),
                    "distance_score": item.get("distance_score"),
                }
                for item in payload.get("vector_results", [])
            ],
            "bm25_results": [
                {
                    "filename": item.get("filename"),
                    "topic": item.get("topic"),
                    "section": item.get("section"),
                    "chunk_id": item.get("chunk_id"),
                    "bm25_rank": item.get("bm25_rank"),
                    "bm25_score": item.get("bm25_score"),
                }
                for item in payload.get("bm25_results", [])
            ],
            "final_results": [
                {
                    "filename": item.get("filename"),
                    "topic": item.get("topic"),
                    "section": item.get("section"),
                    "chunk_id": item.get("chunk_id"),
                    "retrieval_sources": item.get("retrieval_sources"),
                    "vector_rank": item.get("vector_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                    "keyword_hits": item.get("keyword_hits"),
                    "fusion_score": item.get("fusion_score"),
                }
                for item in payload.get("results", [])
            ],
        }
    )

    return format_retrieval_results(payload)