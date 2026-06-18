from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver

from app.config import AppSettings


class AsyncPostgresSaver:
    @classmethod
    def from_conn_string(cls, database_url: str):
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver as Saver
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "langgraph-checkpoint-postgres is required for Postgres checkpointer"
            ) from exc
        return Saver.from_conn_string(database_url)


@lru_cache(maxsize=2)
def _memory_checkpointer():
    return InMemorySaver()


@lru_cache(maxsize=2)
def _postgres_checkpointer(database_url: str):
    return AsyncPostgresSaver.from_conn_string(database_url)


def get_checkpointer(settings: AppSettings):
    if settings.checkpoint_backend == "memory":
        return _memory_checkpointer()
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres checkpointer")
    return _postgres_checkpointer(settings.database_url)


def reset_checkpointer_cache() -> None:
    _memory_checkpointer.cache_clear()
    _postgres_checkpointer.cache_clear()
