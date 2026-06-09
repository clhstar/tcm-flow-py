import asyncio

from fastapi import HTTPException

from app.schemas import RunCreateRequest
from app.store import ThreadStore, RunManager, RunRecord
from app.stream import StreamBridge
from app.worker import run_agent


class AppState:
    def __init__(self):
        self.thread_store = ThreadStore()
        self.run_manager = RunManager()
        self.bridge = StreamBridge()


state = AppState()


async def start_run(body: RunCreateRequest, thread_id: str) -> RunRecord:
    thread = await state.thread_store.get(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    record = await state.run_manager.create(
        thread_id=thread_id,
        assistant_id=body.assistant_id,
    )

    state.bridge.create(record.run_id)

    task = asyncio.create_task(
        run_agent(
            bridge=state.bridge,
            run_manager=state.run_manager,
            thread_store=state.thread_store,
            record=record,
            input_data=body.input.model_dump(),
            context=body.context,
        )
    )

    record.task = task
    return record