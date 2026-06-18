from functools import lru_cache
from importlib import import_module
import sys

from langgraph.checkpoint.memory import InMemorySaver

from app.config import AppSettings


class _ManagedPostgresCheckpointer:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.context = None
        self.checkpointer = None

    def get(self):
        if self.checkpointer is None:
            saver_class = _import_postgres_saver()
            context = saver_class.from_conn_string(self.database_url)
            checkpointer = context.__enter__()
            try:
                checkpointer.setup()
            except Exception:
                context.__exit__(*sys.exc_info())
                raise
            self.context = context
            self.checkpointer = checkpointer
        return self.checkpointer

    def close(self) -> None:
        if self.context is not None:
            self.context.__exit__(None, None, None)
            self.context = None
            self.checkpointer = None


_postgres_checkpointers: dict[str, _ManagedPostgresCheckpointer] = {}


def _import_postgres_saver():
    try:
        module = import_module("langgraph.checkpoint.postgres")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("langgraph.checkpoint.postgres"):
            raise ModuleNotFoundError(
                "langgraph-checkpoint-postgres is required for Postgres checkpointer"
            ) from exc
        raise
    return module.PostgresSaver


@lru_cache(maxsize=2)
def _memory_checkpointer():
    return InMemorySaver()


def _postgres_checkpointer(database_url: str):
    handle = _postgres_checkpointers.get(database_url)
    if handle is None:
        handle = _ManagedPostgresCheckpointer(database_url)
        _postgres_checkpointers[database_url] = handle
    return handle.get()


def get_checkpointer(settings: AppSettings):
    if settings.checkpoint_backend == "memory":
        return _memory_checkpointer()
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres checkpointer")
    return _postgres_checkpointer(settings.database_url)


def reset_checkpointer_cache() -> None:
    _memory_checkpointer.cache_clear()
    for handle in list(_postgres_checkpointers.values()):
        handle.close()
    _postgres_checkpointers.clear()
