from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.runtime.serialization import serialize
from app.runtime.stream import StreamBridge


ProjectedEvents = list[tuple[str, Any]]
ValuesObserver = Callable[
    [dict[str, Any]],
    ProjectedEvents | Awaitable[ProjectedEvents],
]


@dataclass(frozen=True)
class StreamSnapshot:
    latest_values: dict[str, Any] = field(default_factory=dict)
    latest_messages: list[Any] = field(default_factory=list)


def normalize_stream_modes(modes: Any) -> list[str]:
    if modes is None:
        return ["messages"]
    if isinstance(modes, str):
        return [modes]
    if isinstance(modes, (list, tuple, set)):
        return [str(mode) for mode in modes]
    return ["messages"]


def ensure_internal_stream_modes(modes: list[str]) -> list[str]:
    result: list[str] = []
    for mode in ("messages", "values", *modes):
        if mode not in result:
            result.append(mode)
    return result


def split_stream_item(item: Any) -> tuple[str, Any]:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[0], item[1]
    return "values", item


class LangGraphStreamAdapter:
    def __init__(
        self,
        bridge: StreamBridge,
        run_id: str,
        emit_debug_events: bool,
    ):
        self.bridge = bridge
        self.run_id = run_id
        self.emit_debug_events = emit_debug_events

    async def forward(
        self,
        agent: Any,
        graph_input: Any,
        config: dict[str, Any],
        requested_modes: Any,
        values_observer: ValuesObserver | None = None,
    ) -> StreamSnapshot:
        normalized_modes = normalize_stream_modes(requested_modes)
        internal_modes = ensure_internal_stream_modes(normalized_modes)
        publish_values = "values" in normalized_modes or self.emit_debug_events

        latest_values: dict[str, Any] = {}
        latest_messages: list[Any] = []

        async for stream_item in agent.astream(
            graph_input,
            config=config,
            stream_mode=internal_modes,
        ):
            stream_event, chunk = split_stream_item(stream_item)

            if stream_event == "messages":
                await self.bridge.publish(
                    self.run_id,
                    "messages",
                    serialize(chunk, mode="messages"),
                )
                continue

            if stream_event != "values":
                payload = serialize(chunk, mode=stream_event)
                if stream_event != "updates":
                    payload = {
                        "stream_event": stream_event,
                        "data": payload,
                    }
                await self.bridge.publish(self.run_id, "updates", payload)
                continue

            serialized_values = serialize(chunk, mode="values")
            if isinstance(serialized_values, dict):
                latest_values = serialized_values
                messages = serialized_values.get("messages", [])
                latest_messages = list(messages) if isinstance(messages, list) else []

            if publish_values:
                await self.bridge.publish(
                    self.run_id,
                    "values",
                    serialized_values,
                )

            if values_observer is None or not isinstance(serialized_values, dict):
                continue

            projected_events = values_observer(serialized_values)
            if inspect.isawaitable(projected_events):
                projected_events = await projected_events
            for event, data in projected_events:
                await self.bridge.publish(self.run_id, event, data)

        return StreamSnapshot(
            latest_values=latest_values,
            latest_messages=latest_messages,
        )
