import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


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
    elasticsearch_url: str | None
    elasticsearch_rag_index_alias: str
    elasticsearch_analyzer: str

    @classmethod
    def from_env(cls) -> "AppSettings":
        checkpoint_backend = os.getenv("CHECKPOINT_BACKEND", "memory").strip().lower()
        postgres_pool_size = _int_env("POSTGRES_POOL_SIZE", 10)
        if checkpoint_backend not in {"memory", "postgres"}:
            raise ValueError("CHECKPOINT_BACKEND must be memory or postgres")
        if postgres_pool_size <= 0:
            raise ValueError("POSTGRES_POOL_SIZE must be positive")
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            postgres_pool_size=postgres_pool_size,
            checkpoint_backend=checkpoint_backend,
            elasticsearch_url=os.getenv("ELASTICSEARCH_URL"),
            elasticsearch_rag_index_alias=os.getenv(
                "ELASTICSEARCH_RAG_INDEX_ALIAS",
                "tcm_rag_chunks_current",
            ),
            elasticsearch_analyzer=os.getenv("ELASTICSEARCH_ANALYZER", "standard"),
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    load_dotenv(override=False)
    return AppSettings.from_env()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
