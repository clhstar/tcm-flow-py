import asyncio

from fastapi import HTTPException

from app.agents.registry import resolve_agent_factory
from app.runtime.runs.worker import run_agent
from app.runtime.state import state
from app.schemas import RunCreateRequest
from app.store.models import RunRecord


async def start_run(body: RunCreateRequest, thread_id: str) -> RunRecord:
    """
    创建并启动一次 Agent Run。

    - 校验 thread
    - 创建 run
    - 创建 stream bridge
    - 解析 agent_factory
    - 启动后台 worker
    """
    thread = await state.thread_store.get(thread_id)

    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    record = await state.run_manager.create(
        thread_id=thread_id,
        assistant_id=body.assistant_id,
    )

    state.bridge.create(record.run_id)

    try:
        agent_factory = resolve_agent_factory(body.assistant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    context = dict(body.context or {})
    context["stream_mode"] = list(body.stream_mode or [])

    task = asyncio.create_task(
        run_agent(
            bridge=state.bridge,
            run_manager=state.run_manager,
            thread_store=state.thread_store,
            record=record,
            agent_factory=agent_factory,
            input_data=body.input.model_dump(),
            context=context,
        )
    )

    record.task = task
    return record
