import unittest
from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk

from app.runtime.runs.stream_adapter import (
    LangGraphStreamAdapter,
    StreamSnapshot,
    ensure_internal_stream_modes,
    normalize_stream_modes,
)
from app.runtime.stream import StreamBridge


class FakeStreamingAgent:
    def __init__(self, chunks: list[Any]):
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def astream(self, graph_input, *, config, stream_mode):
        self.calls.append(
            {
                "graph_input": graph_input,
                "config": config,
                "stream_mode": stream_mode,
            }
        )
        for chunk in self.chunks:
            yield chunk


class RuntimeStreamAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def forward_chunks(
        self,
        chunks: list[Any],
        *,
        requested_modes: Any = "messages",
        emit_debug_events: bool = False,
        values_observer=None,
    ):
        run_id = "run-1"
        bridge = StreamBridge()
        bridge.create(run_id)
        agent = FakeStreamingAgent(chunks)
        adapter = LangGraphStreamAdapter(
            bridge=bridge,
            run_id=run_id,
            emit_debug_events=emit_debug_events,
        )
        snapshot = await adapter.forward(
            agent=agent,
            graph_input={"messages": [{"role": "user", "content": "hello"}]},
            config={"configurable": {"thread_id": "thread-1"}},
            requested_modes=requested_modes,
            values_observer=values_observer,
        )

        events = []
        queue = bridge.queues[run_id]
        while not queue.empty():
            events.append(queue.get_nowait())
        return snapshot, events, agent

    def test_mode_normalization_and_internal_ordering_are_deterministic(self):
        self.assertEqual(normalize_stream_modes(None), ["messages"])
        self.assertEqual(normalize_stream_modes(object()), ["messages"])
        self.assertEqual(normalize_stream_modes("tasks"), ["tasks"])
        self.assertEqual(normalize_stream_modes(["tasks", 7]), ["tasks", "7"])
        self.assertEqual(normalize_stream_modes(("updates",)), ["updates"])
        self.assertEqual(normalize_stream_modes({"values"}), ["values"])
        self.assertEqual(normalize_stream_modes([]), [])
        self.assertEqual(
            ensure_internal_stream_modes(
                ["values", "tasks", "messages", "updates", "tasks"]
            ),
            ["messages", "values", "tasks", "updates"],
        )

    def test_set_mode_normalization_uses_stable_sorted_string_order(self):
        class DeliberatelyUnsortedSet(set):
            def __iter__(self):
                return iter(("updates", "tasks", "messages"))

        self.assertEqual(
            normalize_stream_modes(
                DeliberatelyUnsortedSet({"messages", "tasks", "updates"})
            ),
            ["messages", "tasks", "updates"],
        )

    def test_stream_snapshot_is_frozen_and_uses_independent_defaults(self):
        first = StreamSnapshot()
        second = StreamSnapshot()

        self.assertIsNot(first.latest_values, second.latest_values)
        self.assertIsNot(first.latest_messages, second.latest_messages)
        with self.assertRaises(FrozenInstanceError):
            first.latest_values = {"status": "changed"}

    async def test_multi_mode_stream_maps_and_serializes_events_exactly(self):
        snapshot, events, agent = await self.forward_chunks(
            [
                (
                    "messages",
                    (
                        AIMessageChunk(content="hello", id="chunk-1"),
                        {"langgraph_node": "agent"},
                    ),
                ),
                ("values", {"status": "working"}),
                ("tasks", {"id": "task-1"}),
                ("updates", {"agent": {"status": "complete"}}),
            ],
            requested_modes=["messages", "tasks", "updates"],
        )

        self.assertEqual(
            events,
            [
                (
                    "messages",
                    [
                        {
                            "type": "AIMessageChunk",
                            "content": "hello",
                            "id": "chunk-1",
                            "additional_kwargs": {},
                            "response_metadata": {},
                            "tool_calls": [],
                            "tool_call_chunks": [],
                            "invalid_tool_calls": [],
                        },
                        {"langgraph_node": "agent"},
                    ],
                ),
                (
                    "updates",
                    {"stream_event": "tasks", "data": {"id": "task-1"}},
                ),
                ("updates", {"agent": {"status": "complete"}}),
            ],
        )
        self.assertEqual(snapshot.latest_values, {"status": "working"})
        self.assertEqual(
            agent.calls,
            [
                {
                    "graph_input": {
                        "messages": [{"role": "user", "content": "hello"}]
                    },
                    "config": {"configurable": {"thread_id": "thread-1"}},
                    "stream_mode": ["messages", "values", "tasks", "updates"],
                }
            ],
        )

    async def test_requested_values_precede_observer_events_and_fill_snapshot(self):
        observed_values = []

        def observer(values):
            observed_values.append(values)
            return [("final", {"status": values["status"]})]

        snapshot, events, _ = await self.forward_chunks(
            [
                (
                    "values",
                    {
                        "status": "ready",
                        "messages": [AIMessage(content="done", id="message-1")],
                    },
                )
            ],
            requested_modes=["values"],
            values_observer=observer,
        )

        expected_values = {
            "status": "ready",
            "messages": [
                {
                    "type": "ai",
                    "content": "done",
                    "id": "message-1",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                }
            ],
        }
        self.assertEqual(observed_values, [expected_values])
        self.assertEqual(
            events,
            [
                ("values", expected_values),
                ("final", {"status": "ready"}),
            ],
        )
        self.assertEqual(snapshot.latest_values, expected_values)
        self.assertEqual(snapshot.latest_messages, expected_values["messages"])

    async def test_observer_and_snapshot_mutations_do_not_alias_published_values(self):
        def observer(values):
            values["status"] = "observer-mutated"
            values["messages"][0]["content"] = "observer-mutated"
            return []

        snapshot, events, _ = await self.forward_chunks(
            [
                (
                    "values",
                    {
                        "status": "original",
                        "messages": [{"type": "ai", "content": "original"}],
                    },
                )
            ],
            requested_modes="values",
            values_observer=observer,
        )
        original_values = {
            "status": "original",
            "messages": [{"type": "ai", "content": "original"}],
        }

        self.assertEqual(snapshot.latest_values, original_values)
        self.assertEqual(snapshot.latest_messages, original_values["messages"])
        self.assertEqual(events, [("values", original_values)])

        snapshot.latest_values["status"] = "snapshot-mutated"
        snapshot.latest_messages[0]["content"] = "snapshot-mutated"

        self.assertEqual(events, [("values", original_values)])

    async def test_messages_only_keeps_raw_values_fallback_without_publishing_it(self):
        snapshot, events, agent = await self.forward_chunks(
            [
                {
                    "status": "complete",
                    "messages": [{"type": "ai", "content": "raw fallback"}],
                }
            ],
            requested_modes="messages",
        )

        self.assertEqual(events, [])
        self.assertEqual(snapshot.latest_values["status"], "complete")
        self.assertEqual(
            snapshot.latest_messages,
            [{"type": "ai", "content": "raw fallback"}],
        )
        self.assertEqual(agent.calls[0]["stream_mode"], ["messages", "values"])

    async def test_debug_events_publish_unrequested_values(self):
        _, events, _ = await self.forward_chunks(
            [("values", {"status": "debug"})],
            requested_modes="messages",
            emit_debug_events=True,
        )

        self.assertEqual(events, [("values", {"status": "debug"})])

    async def test_async_observer_is_awaited_and_non_dict_values_are_skipped(self):
        observed_values = []

        async def observer(values):
            observed_values.append(values)
            return [("clarification", {"question": values["question"]})]

        snapshot, events, _ = await self.forward_chunks(
            [
                ("values", {"question": "How long?"}),
                ("values", "not a state mapping"),
            ],
            requested_modes="values",
            values_observer=observer,
        )

        self.assertEqual(observed_values, [{"question": "How long?"}])
        self.assertEqual(
            events,
            [
                ("values", {"question": "How long?"}),
                ("clarification", {"question": "How long?"}),
                ("values", "not a state mapping"),
            ],
        )
        self.assertEqual(snapshot.latest_values, {"question": "How long?"})


class StreamBridgeCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_uses_default_delay_before_removing_queue(self):
        bridge = StreamBridge()
        bridge.create("default-delay-run")

        with patch(
            "app.runtime.stream.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            await bridge.cleanup("default-delay-run")

        sleep.assert_awaited_once_with(60)
        self.assertNotIn("default-delay-run", bridge.queues)

    async def test_cleanup_uses_explicit_positive_delay(self):
        bridge = StreamBridge()
        bridge.create("positive-delay-run")

        with patch(
            "app.runtime.stream.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            await bridge.cleanup("positive-delay-run", delay=2.5)

        sleep.assert_awaited_once_with(2.5)
        self.assertNotIn("positive-delay-run", bridge.queues)

    async def test_cleanup_skips_sleep_for_non_positive_delays_idempotently(self):
        bridge = StreamBridge()

        for delay in (0, -1):
            with self.subTest(delay=delay):
                run_id = f"non-positive-delay-{delay}"
                bridge.create(run_id)

                with patch(
                    "app.runtime.stream.asyncio.sleep",
                    new_callable=AsyncMock,
                ) as sleep:
                    await bridge.cleanup(run_id, delay=delay)
                    await bridge.cleanup(run_id, delay=delay)

                sleep.assert_not_awaited()
                self.assertNotIn(run_id, bridge.queues)

    async def test_cleanup_removes_queue_idempotently_without_changing_end_contract(self):
        bridge = StreamBridge()
        bridge.create("ended-run")
        await bridge.publish_end("ended-run")

        events = [event async for event in bridge.subscribe("ended-run")]

        self.assertEqual(
            events,
            ['event: end\ndata: {"status": "done"}\n\n'],
        )

        bridge.create("cleanup-run")
        await bridge.cleanup("cleanup-run", delay=0)
        await bridge.cleanup("cleanup-run", delay=0)

        self.assertNotIn("cleanup-run", bridge.queues)


if __name__ == "__main__":
    unittest.main()
