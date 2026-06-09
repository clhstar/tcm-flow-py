import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import MiniMaxEmbeddings

load_dotenv()

INDEX_DIR = Path("data/index/chroma")
COLLECTION_NAME = "tcm_knowledge"


def get_embeddings() -> MiniMaxEmbeddings:
    """
    使用 MiniMax Embeddings 作为向量化模型。

    需要在 .env 中配置：
    MINIMAX_API_KEY=xxx
    MINIMAX_GROUP_ID=xxx

    可选：
    MINIMAX_EMBEDDING_MODEL=embo-01
    """
    model = os.getenv("MINIMAX_EMBEDDING_MODEL")

    if model:
        return MiniMaxEmbeddings(model=model)

    return MiniMaxEmbeddings()


def get_vector_store() -> Chroma:
    """
    获取 Chroma 向量库实例。
    persist_directory 表示索引会持久化保存到 data/index/chroma。
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(INDEX_DIR),
    )