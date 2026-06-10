from langchain.tools import tool

from app.rag.retrieval_log import write_retrieval_log
from app.rag.retriever import retrieve_tcm_docs, format_retrieval_results


@tool("retrieve_tcm_knowledge")
def retrieve_tcm_knowledge(query: str) -> str:
    """
    当用户询问中医症状、证候、病机、问诊要点或日常调护建议时调用。

    输入 query 应包含用户主诉、主要症状、伴随症状和需要检索的中医概念。
    工具会自动进行 query rewrite、向量检索、简单重排，并返回相关中医知识依据和允许使用的专业术语。
    本工具只提供知识检索依据，不直接生成诊断结论，也不提供处方剂量。
    """
    payload = retrieve_tcm_docs(query=query, k=3, candidate_k=8)

    write_retrieval_log(
        {
            "tool": "retrieve_tcm_knowledge",
            "original_query": payload.get("original_query"),
            "rewritten_query": payload.get("rewritten_query"),
            "allowed_terms": payload.get("allowed_terms"),
            "results": [
                {
                    "filename": item.get("filename"),
                    "topic": item.get("topic"),
                    "section": item.get("section"),
                    "chunk_id": item.get("chunk_id"),
                    "distance_score": item.get("distance_score"),
                    "keyword_hits": item.get("keyword_hits"),
                    "rerank_score": item.get("rerank_score"),
                }
                for item in payload.get("results", [])
            ],
        }
    )

    return format_retrieval_results(payload)