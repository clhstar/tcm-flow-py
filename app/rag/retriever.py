from app.rag.vector_store import get_vector_store


def retrieve_tcm_docs(query: str, k: int = 3) -> list[dict]:
    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(query, k=k)

    docs = []

    for doc, score in results:
        docs.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "filename": doc.metadata.get("filename", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "score": float(score),
            }
        )

    return docs


def format_retrieval_results(results: list[dict]) -> str:
    if not results:
        return "未检索到相关中医知识。"

    lines = ["检索到以下中医知识依据："]

    for index, item in enumerate(results, start=1):
        lines.append(
            f"""
[{index}]
内容：
{item["content"]}

来源：{item["filename"]}，chunk_id={item["chunk_id"]}，score={item["score"]:.4f}
""".strip()
        )

    return "\n\n".join(lines)