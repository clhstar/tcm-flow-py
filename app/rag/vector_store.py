import json
from functools import lru_cache
from pathlib import Path

from app.rag.ancient_books.config import load_production_config
from app.rag.ancient_books.models import (
    BgeM3Encoder,
    BgeReranker,
    snapshot_files,
    snapshot_tree_sha256,
)
from app.rag.ancient_books.pipeline import sha256_file
from app.rag.ancient_books.runtime import ProductionRetrievalEngine, load_index


INDEX_DIR = Path("data/rag/ancient_books/index")
CORPUS_DIR = Path("data/rag/ancient_books/corpus")
CONFIG_PATH = Path("app/rag/config/ancient_books.yaml")
MODELS_MANIFEST_PATH = Path("data/rag/ancient_books/models/manifest.json")


class _UnavailableModel:
    def __init__(self, reason: str):
        self.reason = reason

    def encode(self, texts):
        raise RuntimeError(self.reason)

    def score(self, pairs):
        raise RuntimeError(self.reason)


def _load_model_record(
    manifest: dict,
    *,
    role: str,
    expected: dict,
) -> tuple[Path, dict]:
    record = manifest.get(role)
    if not isinstance(record, dict):
        raise ValueError(f"模型 Manifest 缺少 {role}")
    if record.get("model") != expected["model"]:
        raise ValueError(f"{role} 模型名与生产配置不一致")
    if record.get("revision") != expected["revision"]:
        raise ValueError(f"{role} 模型 revision 与生产配置不一致")
    path = Path(record["local_path"]).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"缺少本地模型快照: {path}")
    actual_hash = snapshot_tree_sha256(snapshot_files(path))
    if actual_hash != record.get("snapshot_tree_sha256"):
        raise ValueError(f"{role} 本地模型快照哈希不一致")
    return path, record


@lru_cache(maxsize=4)
def _get_production_engine_cached(
    index_dir: str,
    corpus_dir: str,
    config_path: str,
    models_manifest_path: str,
    index_manifest_sha256: str,
    corpus_manifest_sha256: str,
    models_manifest_sha256: str | None,
) -> ProductionRetrievalEngine:
    del index_manifest_sha256, corpus_manifest_sha256, models_manifest_sha256
    index = load_index(Path(index_dir), Path(corpus_dir))
    config = load_production_config(Path(config_path))

    try:
        model_manifest = json.loads(
            Path(models_manifest_path).read_text(encoding="utf-8")
        )
        embedding_path, _ = _load_model_record(
            model_manifest,
            role="embedding",
            expected=config["models"]["embedding"],
        )
        encoder = BgeM3Encoder(embedding_path, config["models"]["embedding"])
    except Exception as error:
        encoder = _UnavailableModel(f"Dense 模型不可用: {error}")

    try:
        if "model_manifest" not in locals():
            model_manifest = json.loads(
                Path(models_manifest_path).read_text(encoding="utf-8")
            )
        reranker_path, _ = _load_model_record(
            model_manifest,
            role="reranker",
            expected=config["models"]["reranker"],
        )
        reranker = BgeReranker(reranker_path, config["models"]["reranker"])
    except Exception as error:
        reranker = _UnavailableModel(f"Reranker 模型不可用: {error}")

    return ProductionRetrievalEngine(
        index=index,
        encoder=encoder,
        reranker=reranker,
        settings=config["retrieval"],
    )


def get_production_engine(
    *,
    index_dir: Path = INDEX_DIR,
    corpus_dir: Path = CORPUS_DIR,
    config_path: Path = CONFIG_PATH,
    models_manifest_path: Path = MODELS_MANIFEST_PATH,
) -> ProductionRetrievalEngine:
    index_manifest = index_dir / "manifest.json"
    corpus_manifest = corpus_dir / "manifest.json"
    models_hash = (
        sha256_file(models_manifest_path) if models_manifest_path.is_file() else None
    )
    return _get_production_engine_cached(
        str(index_dir.resolve()),
        str(corpus_dir.resolve()),
        str(config_path.resolve()),
        str(models_manifest_path.resolve()),
        sha256_file(index_manifest),
        sha256_file(corpus_manifest),
        models_hash,
    )


def clear_production_engine_cache() -> None:
    _get_production_engine_cached.cache_clear()


def get_database_engine():
    raise RuntimeError("Database RAG engine requires application startup wiring")


def get_configured_retrieval_engine(settings=None):
    if settings is None:
        from app.config import get_settings

        settings = get_settings()
    if settings.rag_engine == "database":
        return get_database_engine()
    return get_production_engine()
