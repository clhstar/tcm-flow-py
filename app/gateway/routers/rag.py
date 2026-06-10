from fastapi import APIRouter

from app.rag.retrieval_log import read_recent_logs

router = APIRouter(
    prefix="/api/rag",
    tags=["rag"],
)


@router.get("/logs")
async def get_rag_logs(limit: int = 20):
    return {
        "items": read_recent_logs(limit=limit)
    }