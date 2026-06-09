from app.rag.documents import load_markdown_documents, split_documents
from app.rag.vector_store import get_vector_store


def build_index():
    documents = load_markdown_documents()

    if not documents:
        print("No documents found in data/raw")
        return

    chunks = split_documents(documents)

    vector_store = get_vector_store()

    # 避免重复插入，先重置 collection
    try:
        vector_store.reset_collection()
    except Exception as exc:
        print(f"Reset collection failed, continue anyway: {exc}")

    ids = [
        f"{doc.metadata.get('filename', 'doc')}-{doc.metadata.get('chunk_id', i)}"
        for i, doc in enumerate(chunks)
    ]

    vector_store.add_documents(chunks, ids=ids)

    print(f"Indexed {len(chunks)} chunks.")
    print("Index path: data/index/chroma")


if __name__ == "__main__":
    build_index()