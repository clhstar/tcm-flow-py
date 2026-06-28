import asyncio
import logging
import traceback
from collections.abc import Callable
from typing import Any

from app.runtime.runs.context import (
    RunContext,
    build_runnable_config,
    build_runtime_context,
)
from app.runtime.runs.input import extract_user_text
from app.runtime.runs.projection import (
    RunCompletionProjection,
    checkpoint_message_count,
)
from app.runtime.runs.stream_adapter import LangGraphStreamAdapter
from app.runtime.stream import StreamBridge
from app.store.models import RunRecord
from app.store.run_manager import RunManager


logger = logging.getLogger(__name__)


async def run_agent(
    bridge: StreamBridge,
    run_manager: RunManager,
    record: RunRecord,
    *,
    ctx: RunContext,
    agent_factory: Callable[[dict[str, Any]], Any],
    graph_input: dict[str, Any],
    config: dict[str, Any],
    stream_modes: list[str] | None = None,
) -> None:
    run_id = record.run_id
    thread_id = record.thread_id
    thread_store = ctx.thread_store

    try:
        await run_manager.set_status(run_id, "running")
        await thread_store.update_status(thread_id, "running")

        await bridge.publish(
            run_id,
            "metadata",
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "assistant_id": record.assistant_id,
                "architecture": "tcm-flow",
            },
        )

        thread = await thread_store.get(thread_id)
        thread_values = dict(thread.values or {}) if thread is not None else {}

        runtime_context = build_runtime_context(record, ctx.agent_context)
        runnable_config = build_runnable_config(record, config, runtime_context)
        agent = agent_factory(runtime_context)
        message_start_index = await checkpoint_message_count(agent, runnable_config)
        emit_debug_events = bool(runtime_context.get("debug_events"))

        projection = RunCompletionProjection(
            record=record,
            thread_values=thread_values,
            user_text=extract_user_text(graph_input),
            emit_debug_events=emit_debug_events,
            message_start_index=message_start_index,
        )
        adapter = LangGraphStreamAdapter(
            bridge=bridge,
            run_id=run_id,
            emit_debug_events=emit_debug_events,
        )
        snapshot = await adapter.forward(
            agent,
            graph_input,
            runnable_config,
            stream_modes,
            projection.observe_values,
        )

        async def publish_update(payload: dict[str, Any]) -> None:
            await bridge.publish(run_id, "updates", payload)

        completion = await projection.complete(
            agent=agent,
            config=runnable_config,
            snapshot=snapshot,
            publish_update=publish_update,
        )
        await thread_store.update_values(
            thread_id,
            completion.thread_values,
            run_id=run_id,
        )
        await bridge.publish(run_id, "final", completion.final_payload)
        await run_manager.set_status(run_id, completion.run_status)
        await thread_store.update_status(thread_id, completion.thread_status)

    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        logger.exception("Run %s failed", run_id)
        await run_manager.set_status(run_id, "error", error=error)
        await thread_store.update_status(thread_id, "error")
        await bridge.publish(run_id, "error", {"message": error})

    finally:
        await bridge.publish_end(run_id)
        asyncio.create_task(bridge.cleanup(run_id, delay=60))
