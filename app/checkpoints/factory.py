import inspect
import sys
from functools import lru_cache
from importlib import import_module

from langgraph.checkpoint.memory import InMemorySaver

from app.config import AppSettings


class _ManagedAsyncPostgresCheckpointer:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.context = None
        self.checkpointer = None

    async def get(self):
        if self.checkpointer is None:
            saver_class = _import_async_postgres_saver()
            context = saver_class.from_conn_string(self.database_url)
            checkpointer = await context.__aenter__()
            try:
                setup_result = checkpointer.setup()
                if inspect.isawaitable(setup_result):
                    await setup_result
            except Exception:
                await context.__aexit__(*sys.exc_info())
                raise
            self.context = context
            self.checkpointer = checkpointer
        return self.checkpointer

    async def close(self) -> None:
        if self.context is not None:
            await self.context.__aexit__(None, None, None)
            self.context = None
            self.checkpointer = None


_async_postgres_checkpointers: dict[str, _ManagedAsyncPostgresCheckpointer] = {}


def _import_async_postgres_saver():
    try:
        module = import_module("langgraph.checkpoint.postgres.aio")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("langgraph.checkpoint.postgres"):
            raise ModuleNotFoundError(
                "langgraph-checkpoint-postgres is required for Postgres checkpointer"
            ) from exc
        raise
    return module.AsyncPostgresSaver


@lru_cache(maxsize=2)
def _memory_checkpointer():
    return InMemorySaver()


async def _async_postgres_checkpointer(database_url: str):
    handle = _async_postgres_checkpointers.get(database_url)
    if handle is None:
        handle = _ManagedAsyncPostgresCheckpointer(database_url)
        _async_postgres_checkpointers[database_url] = handle
    return await handle.get()


def get_checkpointer(settings: AppSettings):
    if settings.checkpoint_backend == "memory":
        return _memory_checkpointer()
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres checkpointer")
    raise RuntimeError("Postgres checkpointer is async; use get_checkpointer_async")


async def get_checkpointer_async(settings: AppSettings):
    if settings.checkpoint_backend == "memory":
        return _memory_checkpointer()
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres checkpointer")
    return await _async_postgres_checkpointer(settings.database_url)


def reset_checkpointer_cache() -> None:
    if _async_postgres_checkpointers:
        raise RuntimeError("async Postgres checkpointers require async reset")
    _memory_checkpointer.cache_clear()


async def reset_checkpointer_cache_async() -> None:
    _memory_checkpointer.cache_clear()
    for handle in list(_async_postgres_checkpointers.values()):
        await handle.close()
    _async_postgres_checkpointers.clear()
