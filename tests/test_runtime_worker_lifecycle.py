import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.runtime.runs.context import RunContext
from app.runtime.runs.worker import run_agent
from app.runtime.stream import StreamBridge
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


class CancelledAgent:
    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": []})

    async def astream(self, graph_input, *, config, stream_mode):
        if False:
            yield None
        raise asyncio.CancelledError


class SuccessfulAgent:
    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": []})

    async def astream(self, graph_input, *, config, stream_mode):
        yield (
            "values",
            {
                "messages": [
                    HumanMessage(content="question", id="human-1"),
                    AIMessage(content="answer", id="ai-1"),
                ]
            },
        )


class CancelOnSuccessRunManager(RunManager):
    async def set_status(
        self,
        run_id: str,
        status: str,
        error: str | None = None,
    ):
        if status == "success":
            task = asyncio.current_task()
            if task is not None:
                task.cancel()
            await asyncio.sleep(0)
        await super().set_status(run_id, status, error=error)


class FailingCancellationRunManager(RunManager):
    async def set_status(
        self,
        run_id: str,
        status: str,
        error: str | None = None,
    ):
        if status == "cancelled":
            raise RuntimeError("run cancellation status failed")
        await super().set_status(run_id, status, error=error)


class FailingCancellationThreadStore(ThreadStore):
    async def update_status(self, thread_id: str, status: str):
        if status == "idle":
            raise RuntimeError("thread cancellation status failed")
        await super().update_status(thread_id, status)


class RecordingBridge(StreamBridge):
    def __init__(self):
        super().__init__()
        self.cleanup_calls: list[tuple[str, float]] = []

    async def cleanup(self, run_id: str, delay: float = 60):
        self.cleanup_calls.append((run_id, delay))


class RuntimeWorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def drain_events(self, bridge: RecordingBridge, run_id: str):
        queued_events = []
        while not bridge.queues[run_id].empty():
            queued_events.append(await bridge.queues[run_id].get())
        return queued_events

    async def test_cancelled_run_is_finalized_without_error_or_final_event(self):
        bridge = RecordingBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with self.assertRaises(asyncio.CancelledError):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                record=run,
                ctx=RunContext(thread_store=thread_store),
                agent_factory=lambda context: CancelledAgent(),
                graph_input={"messages": []},
                config={},
                stream_modes=["messages"],
            )

        stored_run = await run_manager.get(run.run_id)
        stored_thread = await thread_store.get(thread.thread_id)
        self.assertIsNotNone(stored_run)
        self.assertIsNotNone(stored_thread)
        self.assertEqual(stored_run.status, "cancelled")
        self.assertEqual(stored_thread.status, "idle")

        await asyncio.sleep(0)
        self.assertEqual(bridge.cleanup_calls, [(run.run_id, 60)])

        queued_events = await self.drain_events(bridge, run.run_id)
        event_names = [event for event, _ in queued_events]
        self.assertNotIn("error", event_names)
        self.assertNotIn("final", event_names)
        self.assertLess(event_names.index("metadata"), event_names.index("end"))
        self.assertEqual(queued_events[-1], ("end", {"status": "done"}))

    async def test_cancellation_during_success_status_prevents_final_event(self):
        bridge = RecordingBridge()
        run_manager = CancelOnSuccessRunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "answer",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": [],
                }
            ),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await run_agent(
                    bridge=bridge,
                    run_manager=run_manager,
                    record=run,
                    ctx=RunContext(thread_store=thread_store),
                    agent_factory=lambda context: SuccessfulAgent(),
                    graph_input={
                        "messages": [{"type": "human", "content": "question"}]
                    },
                    config={},
                    stream_modes=["messages"],
                )

        stored_run = await run_manager.get(run.run_id)
        stored_thread = await thread_store.get(thread.thread_id)
        self.assertIsNotNone(stored_run)
        self.assertIsNotNone(stored_thread)
        self.assertEqual(stored_run.status, "cancelled")
        self.assertEqual(stored_thread.status, "idle")
        self.assertEqual(
            stored_thread.values["conversation"][-1]["content"],
            "answer",
        )

        await asyncio.sleep(0)
        self.assertEqual(bridge.cleanup_calls, [(run.run_id, 60)])
        queued_events = await self.drain_events(bridge, run.run_id)
        event_names = [event for event, _ in queued_events]
        self.assertNotIn("error", event_names)
        self.assertNotIn("final", event_names)
        self.assertEqual(queued_events[-1], ("end", {"status": "done"}))

    async def test_run_status_failure_does_not_mask_cancellation(self):
        bridge = RecordingBridge()
        run_manager = FailingCancellationRunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with self.assertLogs("app.runtime.runs.worker", level="ERROR") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await run_agent(
                    bridge=bridge,
                    run_manager=run_manager,
                    record=run,
                    ctx=RunContext(thread_store=thread_store),
                    agent_factory=lambda context: CancelledAgent(),
                    graph_input={"messages": []},
                    config={},
                )

        stored_thread = await thread_store.get(thread.thread_id)
        self.assertIsNotNone(stored_thread)
        self.assertEqual(stored_thread.status, "idle")
        self.assertIn("failed to mark run", "\n".join(logs.output).lower())

        await asyncio.sleep(0)
        self.assertEqual(bridge.cleanup_calls, [(run.run_id, 60)])
        queued_events = await self.drain_events(bridge, run.run_id)
        event_names = [event for event, _ in queued_events]
        self.assertNotIn("error", event_names)
        self.assertNotIn("final", event_names)
        self.assertEqual(queued_events[-1], ("end", {"status": "done"}))

    async def test_thread_status_failure_does_not_mask_cancellation(self):
        bridge = RecordingBridge()
        run_manager = RunManager()
        thread_store = FailingCancellationThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with self.assertLogs("app.runtime.runs.worker", level="ERROR") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await run_agent(
                    bridge=bridge,
                    run_manager=run_manager,
                    record=run,
                    ctx=RunContext(thread_store=thread_store),
                    agent_factory=lambda context: CancelledAgent(),
                    graph_input={"messages": []},
                    config={},
                )

        stored_run = await run_manager.get(run.run_id)
        self.assertIsNotNone(stored_run)
        self.assertEqual(stored_run.status, "cancelled")
        self.assertIn("failed to reset thread", "\n".join(logs.output).lower())

        await asyncio.sleep(0)
        self.assertEqual(bridge.cleanup_calls, [(run.run_id, 60)])
        queued_events = await self.drain_events(bridge, run.run_id)
        event_names = [event for event, _ in queued_events]
        self.assertNotIn("error", event_names)
        self.assertNotIn("final", event_names)
        self.assertEqual(queued_events[-1], ("end", {"status": "done"}))


if __name__ == "__main__":
    unittest.main()
