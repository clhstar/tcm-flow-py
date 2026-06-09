from dotenv import load_dotenv
from app.rag.vector_store import get_embeddings

load_dotenv()


def test_embedding():
    embeddings = get_embeddings()

    text = "胃胀饭后加重，伴随嗳气，大便不成形。"
    vector = embeddings.embed_query(text)

    print("Embedding success.")
    print("Vector dimension:", len(vector))
    print("First 5 values:", vector[:5])


if __name__ == "__main__":
    test_embedding()