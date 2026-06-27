import logging

from fastapi import APIRouter, HTTPException

from app.runtime.state import state

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/threads",
    tags=["threads"],
)


@router.post("")
async def create_thread():
    thread = await state.thread_store.create()

    return {
        "thread_id": thread.thread_id,
        "created_at": thread.created_at,
    }


@router.get("")
async def list_threads():
    return await state.thread_store.list()


@router.get("/{thread_id}")
async def get_thread(thread_id: str):
    thread = await state.thread_store.get(thread_id)

    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    return thread


@router.get("/{thread_id}/history")
async def get_thread_history(thread_id: str):
    thread = await state.thread_store.get(thread_id)

    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    values = thread.values or {}
    logger.info(f"Thread history: {thread}")
    return {
        "thread_id": thread.thread_id,
        "status": thread.status,
        "conversation": values.get("conversation", []),
        "messages": values.get("messages", []),
    }
