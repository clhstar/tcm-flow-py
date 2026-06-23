from contextlib import asynccontextmanager
from importlib import import_module
from typing import AsyncIterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Checkpointer

from app.config import AppSettings


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


@asynccontextmanager
async def make_checkpointer(settings: AppSettings) -> AsyncIterator[Checkpointer]:
    backend = settings.checkpoint_backend

    if backend == "memory":
        yield InMemorySaver()
        return

    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres checkpointer")

    AsyncPostgresSaver = _import_async_postgres_saver()

    async with AsyncPostgresSaver.from_conn_string(
        settings.database_url
    ) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
