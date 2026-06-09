from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.schemas import RunCreateRequest
from app.services import state, start_run

app = FastAPI(title="Mini DeerFlow")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/threads")
async def create_thread():
    thread = await state.thread_store.create()
    return {
        "thread_id": thread.thread_id,
        "created_at": thread.created_at,
    }


@app.get("/api/threads")
async def list_threads():
    return list(state.thread_store.threads.values())


@app.post("/api/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, body: RunCreateRequest):
    record = await start_run(body, thread_id)

    return StreamingResponse(
        state.bridge.subscribe(record.run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Run-Id": record.run_id,
        },
    )
    
@app.get("/api/threads/{thread_id}")
async def get_thread(thread_id: str):
    thread = await state.thread_store.get(thread_id)
    if thread is None:
        return {"error": "Thread not found"}

    return {
        "thread_id": thread.thread_id,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
        "status": thread.status,
        "values": thread.values,
    }
    
@app.get("/api/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    thread = await state.thread_store.get(thread_id)

    if thread is None:
        return {
            "error": "Thread not found"
        }

    return {
        "thread_id": thread.thread_id,
        "status": thread.status,
        "conversation": thread.values.get("conversation", []),
        "pending_clarification": thread.values.get("pending_clarification"),
        "messages": thread.values.get("messages", []),
    }