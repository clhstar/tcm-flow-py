from app.rag.retriever import retrieve_tcm_docs, format_retrieval_results


def test_hybrid():
    query = "胃胀 饭后加重 中医 病机 调护"

    payload = retrieve_tcm_docs(
        query=query,
        k=3,
        candidate_k=8,
        mode="hybrid",
    )

    print(format_retrieval_results(payload))


if __name__ == "__main__":
    test_hybrid()