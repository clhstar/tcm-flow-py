from functools import lru_cache
from typing import Any

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.rag.documents import load_markdown_documents, split_documents
from app.rag.terms import TCM_TERMS, extract_terms


def tokenize_text(text: str) -> list[str]:
    """
    中文 BM25 分词。
    1. 使用 jieba 做基础分词
    2. 额外保留中医术语，避免“脾胃虚弱”等专业词被切碎
    """
    if not text:
        return []

    tokens = [token.strip() for token in jieba.lcut(text) if token.strip()]

    # 保留完整中医术语
    for term in TCM_TERMS:
        if term in text:
            tokens.append(term)

    # query rewrite 后通常有空格，也把空格切开的词加入
    for token in text.split():
        token = token.strip()
        if token:
            tokens.append(token)

    return tokens


def doc_key(metadata: dict[str, Any]) -> str:
    filename = metadata.get("filename", "unknown")
    chunk_id = metadata.get("chunk_id", "unknown")
    return f"{filename}-{chunk_id}"


@lru_cache(maxsize=1)
def get_bm25_index():
    """
    构建 BM25 索引。
    当前版本直接从 data/raw 加载并切分。
    数据量小的时候这样够用。
    后面数据量大了可以持久化 BM25 索引。
    """
    raw_documents = load_markdown_documents()
    chunks = split_documents(raw_documents)

    corpus_tokens = [tokenize_text(doc.page_content) for doc in chunks]

    bm25 = BM25Okapi(corpus_tokens)

    return chunks, bm25


def clear_bm25_cache():
    """
    如果修改了 data/raw 里的知识库，可以调用它清空缓存。
    当前开发阶段一般重启服务即可。
    """
    get_bm25_index.cache_clear()


def bm25_search(query: str, k: int = 8) -> list[dict]:
    """
    BM25 关键词检索。
    score 越大越相关。
    """
    chunks, bm25 = get_bm25_index()

    if not chunks:
        return []

    query_tokens = tokenize_text(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:k]

    results = []

    for rank, index in enumerate(ranked_indices, start=1):
        doc: Document = chunks[index]
        score = float(scores[index])

        # score 为 0 的结果通常没有关键词命中，可以保留，也可以过滤。
        # 这里先保留，方便调试。
        metadata = doc.metadata

        results.append(
            {
                "key": doc_key(metadata),
                "content": doc.page_content,
                "source": metadata.get("source", "unknown"),
                "filename": metadata.get("filename", "unknown"),
                "topic": metadata.get("topic", "unknown"),
                "section": metadata.get("section", "unknown"),
                "chunk_id": metadata.get("chunk_id"),
                "bm25_score": score,
                "bm25_rank": rank,
                "terms": extract_terms(doc.page_content),
            }
        )

    return results