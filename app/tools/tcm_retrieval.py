from langchain.tools import tool

from app.rag.retriever import retrieve_tcm_docs, format_retrieval_results


@tool("retrieve_tcm_knowledge")
def retrieve_tcm_knowledge(query: str) -> str:
    """
    当用户询问中医症状、证候、病机、问诊要点或日常调护建议时调用。

    输入 query 应包含用户主诉、主要症状、伴随症状和需要检索的中医概念。
    返回相关中医知识依据和来源。
    本工具只提供知识检索依据，不直接生成诊断结论，也不提供处方剂量。
    """
    results = retrieve_tcm_docs(query=query, k=3)
    return format_retrieval_results(results)