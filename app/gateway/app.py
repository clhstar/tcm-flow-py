from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.checkpoints.factory import reset_checkpointer_cache_async
from app.config import get_settings
from app.db.migrations import run_schema_migrations
from app.db.pool import create_pool_from_settings
from app.gateway.routers import rag
from app.gateway.routers import thread_runs
from app.gateway.routers import threads
from app.rag.vector_store import (
    build_database_engine,
    clear_database_engine,
    configure_database_engine,
)
from app.runtime.state import configure_state, reset_state_to_memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    pool = await create_pool_from_settings(settings)
    async with pool.acquire() as connection:
        await run_schema_migrations(connection)
    app.state.postgres_pool = pool

    if settings.checkpoint_backend == "postgres":
        configure_state(pool=pool)

    configure_database_engine(build_database_engine(pool, settings))

    try:
        yield
    finally:
        clear_database_engine()
        await reset_checkpointer_cache_async()
        reset_state_to_memory()
        await pool.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="TCM-Flow",
        description="A DeerFlow-like Agentic RAG system for TCM QA",
        version="1.7.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(threads.router)
    app.include_router(thread_runs.router)
    app.include_router(rag.router)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": "1.7.0",
            "architecture": "deerflow-like",
        }

    return app


app = create_app()
