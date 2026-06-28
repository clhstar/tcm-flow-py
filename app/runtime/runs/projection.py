from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from app.middlewares.clarification_controller import (
    extract_latest_clarification_question,
)
from app.middlewares.guardrail_middleware import apply_guardrails
from app.middlewares.trace_middleware import extract_trace_events_from_messages
from app.runtime.public_messages import (
    append_visible_messages,
    build_chat_response,
    extract_final_assistant_text,
    extract_latest_assistant_message,
    extract_pending_clarification,
)
from app.runtime.runs.stream_adapter import ProjectedEvents, StreamSnapshot
from app.runtime.serialization import serialize_message
from app.store.models import RunRecord


PublishUpdate = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class CompletionResult:
    run_status: str
    thread_status: str
    thread_values: dict[str, Any]
    final_payload: dict[str, Any]


async def checkpoint_message_count(agent: Any, config: dict[str, Any]) -> int:
    aget_state = getattr(agent, "aget_state", None)
    if aget_state is None:
        return 0
    snapshot = await aget_state(config)
    return len(snapshot.values.get("messages", []))


async def _replace_final_ai_message_in_checkpoint(
    *,
    agent: Any,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    final_text: str,
) -> list[dict[str, Any]]:
    target = next(
        (
            message
            for message in reversed(messages)
            if message.get("type") == "ai"
            and not message.get("tool_calls")
            and extract_final_assistant_text([message])
        ),
        None,
    )
    if target is None or extract_final_assistant_text([target]) == final_text:
        return messages

    message_id = target.get("id")
    if not message_id:
        raise RuntimeError("无法写回 Guardrail 答案：最终 AIMessage 缺少 id")

    await agent.aupdate_state(
        config,
        {"messages": [AIMessage(id=message_id, content=final_text)]},
    )
    snapshot = await agent.aget_state(config)
    return [
        serialize_message(message)
        for message in snapshot.values.get("messages", [])
    ]


class RunCompletionProjection:
    def __init__(
        self,
        *,
        record: RunRecord,
        thread_values: dict[str, Any],
        user_text: str,
        emit_debug_events: bool,
        message_start_index: int,
    ) -> None:
        self.record = record
        self.thread_values = deepcopy(thread_values)
        self.user_text = user_text
        self.emit_debug_events = emit_debug_events
        self.message_start_index = message_start_index
        self.current_agent_trace: list[dict[str, Any]] = []
        self.emitted_trace_keys: set[str] = set()

    def current_run_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return messages[self.message_start_index :]

    def _refresh_agent_trace(self, values: dict[str, Any]) -> None:
        raw_trace = values.get("agent_trace")
        if isinstance(raw_trace, list):
            self.current_agent_trace = [
                deepcopy(item) for item in raw_trace if isinstance(item, dict)
            ]

    async def observe_values(
        self,
        values: dict[str, Any],
    ) -> ProjectedEvents:
        self._refresh_agent_trace(values)

        raw_messages = values.get("messages", [])
        messages = (
            [serialize_message(message) for message in raw_messages]
            if isinstance(raw_messages, list)
            else []
        )

        if not self.emit_debug_events:
            return []

        return [
            ("updates", event)
            for event in extract_trace_events_from_messages(
                messages=self.current_run_messages(messages),
                emitted_keys=self.emitted_trace_keys,
            )
        ]

    async def complete(
        self,
        *,
        agent: Any,
        config: dict[str, Any],
        snapshot: StreamSnapshot,
        publish_update: PublishUpdate,
    ) -> CompletionResult:
        self._refresh_agent_trace(snapshot.latest_values)
        full_messages = [
            serialize_message(message) for message in snapshot.latest_messages
        ]
        current_messages = self.current_run_messages(full_messages)
        clarification = extract_latest_clarification_question(current_messages)

        if clarification:
            assistant_message = (
                extract_latest_assistant_message(current_messages)
                or clarification
            )
            conversation = append_visible_messages(
                self.thread_values,
                self.user_text,
                assistant_message,
                run_id=self.record.run_id,
                agent_trace=self.current_agent_trace,
            )
            return CompletionResult(
                run_status="waiting_clarification",
                thread_status="waiting",
                thread_values={
                    "messages": full_messages,
                    "conversation": conversation,
                },
                final_payload=build_chat_response(
                    thread_id=self.record.thread_id,
                    run_id=self.record.run_id,
                    status="need_clarification",
                    assistant_message=assistant_message,
                    pending_clarification=extract_pending_clarification(
                        current_messages
                    ),
                ),
            )

        original_final_text = extract_final_assistant_text(current_messages)
        if self.emit_debug_events:
            await publish_update(
                {
                    "type": "guardrail",
                    "status": "started",
                    "agent": "guardrail_middleware",
                    "summary": "正在进行术语一致性校验与答案安全检查。",
                }
            )

        guardrail_result = await apply_guardrails(
            final_text=original_final_text,
            messages=full_messages,
        )

        if self.emit_debug_events:
            await publish_update(
                {
                    "type": "guardrail",
                    "status": "completed",
                    "agent": "guardrail_middleware",
                    "summary": "术语一致性校验完成。",
                    "validation": guardrail_result.get("validation"),
                    "rewritten": guardrail_result.get("rewritten"),
                }
            )

        final_text = guardrail_result["final_text"]
        if final_text != original_final_text:
            full_messages = await _replace_final_ai_message_in_checkpoint(
                agent=agent,
                config=config,
                messages=full_messages,
                final_text=final_text,
            )

        conversation = append_visible_messages(
            self.thread_values,
            self.user_text,
            final_text,
            run_id=self.record.run_id,
            agent_trace=self.current_agent_trace,
        )
        return CompletionResult(
            run_status="success",
            thread_status="idle",
            thread_values={
                "messages": full_messages,
                "conversation": conversation,
            },
            final_payload=build_chat_response(
                thread_id=self.record.thread_id,
                run_id=self.record.run_id,
                status="completed",
                assistant_message=final_text,
            ),
        )
