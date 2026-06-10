from fastapi import FastAPI

from app.gateway.routers import threads
from app.gateway.routers import thread_runs
from app.gateway.routers import rag


def create_app() -> FastAPI:
    app = FastAPI(
        title="TCM-Flow",
        description="A DeerFlow-like Agentic RAG system for TCM QA",
        version="0.9.0",
    )

    app.include_router(threads.router)
    app.include_router(thread_runs.router)
    app.include_router(rag.router)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": "0.9.0",
            "architecture": "deerflow-like",
        }

    return app


app = create_app()