import shutil

from app.rag.documents import load_markdown_documents, split_documents
from app.rag.vector_store import get_vector_store, INDEX_DIR


def build_index():
    documents = load_markdown_documents()

    if not documents:
        print("No documents found in data/raw")
        return

    chunks = split_documents(documents)

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    vector_store = get_vector_store()

    ids = [
        f"{doc.metadata.get('filename', 'doc')}-{doc.metadata.get('chunk_id', i)}"
        for i, doc in enumerate(chunks)
    ]

    vector_store.add_documents(chunks, ids=ids)

    print(f"Indexed {len(chunks)} chunks.")
    print("Index path:", INDEX_DIR)

    print("\nChunk preview:")
    for doc in chunks[:5]:
        print(
            {
                "topic": doc.metadata.get("topic"),
                "section": doc.metadata.get("section"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content[:80],
            }
        )


if __name__ == "__main__":
    build_index()