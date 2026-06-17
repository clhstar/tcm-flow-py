import jieba

from app.rag.ancient_books.query import detect_chief_symptom
from app.rag.terms import TCM_TERMS
from app.rag.vector_store import clear_production_engine_cache, get_production_engine


def tokenize_text(text: str) -> list[str]:
    if not text:
        return []
    tokens = [token.strip() for token in jieba.lcut(text, HMM=False) if token.strip()]
    tokens.extend(term for term in TCM_TERMS if term in text)
    tokens.extend(token for token in text.split() if token)
    return list(dict.fromkeys(tokens))


def get_bm25_index():
    engine = get_production_engine()
    return engine.index.rows, engine.index.bm25


def clear_bm25_cache() -> None:
    clear_production_engine_cache()


def bm25_search(query: str, k: int = 8) -> list[dict]:
    result = get_production_engine().retrieve(
        query,
        chief_symptom=detect_chief_symptom(query),
        mode="keyword",
        top_k=min(max(int(k), 1), 5),
    )
    return result["results"]
