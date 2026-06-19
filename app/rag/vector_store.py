import json
from pathlib import Path

from app.rag.ancient_books.config import load_production_config
from app.rag.ancient_books.models import (
    BgeM3Encoder,
    BgeReranker,
    snapshot_files,
    snapshot_tree_sha256,
)
from app.rag.database.elasticsearch_index import ElasticsearchKeywordIndex
from app.rag.database.engine import DatabaseRetrievalEngine
from app.rag.database.repository import RagPostgresRepository


CONFIG_PATH = Path("app/rag/config/ancient_books.yaml")
MODELS_MANIFEST_PATH = Path("data/rag/ancient_books/models/manifest.json")
DATABASE_CORPUS_ID = "ancient-books-v1.0.0"

_database_engine = None


def _load_model_record(
    manifest: dict,
    *,
    role: str,
    expected: dict,
) -> tuple[Path, dict]:
    record = manifest.get(role)
    if not isinstance(record, dict):
        raise ValueError(f"model manifest missing {role}")
    if record.get("model") != expected["model"]:
        raise ValueError(f"{role} model does not match production config")
    if record.get("revision") != expected["revision"]:
        raise ValueError(f"{role} revision does not match production config")
    path = Path(record["local_path"]).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"missing local model snapshot: {path}")
    actual_hash = snapshot_tree_sha256(snapshot_files(path))
    if actual_hash != record.get("snapshot_tree_sha256"):
        raise ValueError(f"{role} local model snapshot hash does not match manifest")
    return path, record


def configure_database_engine(engine) -> None:
    global _database_engine
    _database_engine = engine


def clear_database_engine() -> None:
    global _database_engine
    _database_engine = None


def get_database_engine():
    if _database_engine is None:
        raise RuntimeError("Database RAG engine requires application startup wiring")
    return _database_engine


def build_database_engine(pool, settings):
    if pool is None:
        raise ValueError("Database RAG engine requires a database pool")
    if not settings.elasticsearch_url:
        raise ValueError("ELASTICSEARCH_URL is required for database RAG engine")

    config = load_production_config(CONFIG_PATH)
    model_manifest = json.loads(MODELS_MANIFEST_PATH.read_text(encoding="utf-8"))
    embedding_path, _ = _load_model_record(
        model_manifest,
        role="embedding",
        expected=config["models"]["embedding"],
    )
    reranker_path, _ = _load_model_record(
        model_manifest,
        role="reranker",
        expected=config["models"]["reranker"],
    )

    try:
        from elasticsearch import AsyncElasticsearch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "elasticsearch is required for database RAG engine"
        ) from exc

    return DatabaseRetrievalEngine(
        corpus_id=DATABASE_CORPUS_ID,
        repository=RagPostgresRepository(pool),
        keyword_index=ElasticsearchKeywordIndex(
            AsyncElasticsearch(settings.elasticsearch_url),
            settings.elasticsearch_rag_index_alias,
        ),
        encoder=BgeM3Encoder(embedding_path, config["models"]["embedding"]),
        reranker=BgeReranker(reranker_path, config["models"]["reranker"]),
        settings=config["retrieval"],
    )


def get_configured_retrieval_engine(settings=None):
    del settings
    return get_database_engine()
