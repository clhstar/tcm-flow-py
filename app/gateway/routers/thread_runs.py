from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.runtime.services import start_run
from app.runtime.state import state
from app.schemas import RunCreateRequest

router = APIRouter(
    prefix="/api/threads",
    tags=["thread-runs"],
)


@router.post("/{thread_id}/runs/stream")
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