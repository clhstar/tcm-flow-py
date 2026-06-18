import os
from dataclasses import dataclass
from functools import lru_cache


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class AppSettings:
    database_url: str | None
    postgres_pool_size: int
    checkpoint_backend: str
    rag_engine: str
    rag_fallback_file_engine: bool
    elasticsearch_url: str | None
    elasticsearch_rag_index_alias: str
    elasticsearch_analyzer: str

    @classmethod
    def from_env(cls) -> "AppSettings":
        checkpoint_backend = os.getenv("CHECKPOINT_BACKEND", "memory").strip().lower()
        rag_engine = os.getenv("RAG_ENGINE", "file").strip().lower()
        if checkpoint_backend not in {"memory", "postgres"}:
            raise ValueError("CHECKPOINT_BACKEND must be memory or postgres")
        if rag_engine not in {"file", "database"}:
            raise ValueError("RAG_ENGINE must be file or database")
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            postgres_pool_size=_int_env("POSTGRES_POOL_SIZE", 10),
            checkpoint_backend=checkpoint_backend,
            rag_engine=rag_engine,
            rag_fallback_file_engine=_bool_env("RAG_FALLBACK_FILE_ENGINE", True),
            elasticsearch_url=os.getenv("ELASTICSEARCH_URL"),
            elasticsearch_rag_index_alias=os.getenv(
                "ELASTICSEARCH_RAG_INDEX_ALIAS",
                "tcm_rag_chunks_current",
            ),
            elasticsearch_analyzer=os.getenv("ELASTICSEARCH_ANALYZER", "standard"),
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings.from_env()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
