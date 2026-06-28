import asyncio
from types import SimpleNamespace
import unittest

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


class RecordingBridge(StreamBridge):
    def __init__(self):
        super().__init__()
        self.cleanup_calls: list[tuple[str, float]] = []

    async def cleanup(self, run_id: str, delay: float = 60):
        self.cleanup_calls.append((run_id, delay))


class RuntimeWorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
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

        queued_events = []
        while not bridge.queues[run.run_id].empty():
            queued_events.append(await bridge.queues[run.run_id].get())
        event_names = [event for event, _ in queued_events]
        self.assertNotIn("error", event_names)
        self.assertNotIn("final", event_names)
        self.assertLess(event_names.index("metadata"), event_names.index("end"))
        self.assertEqual(queued_events[-1], ("end", {"status": "done"}))


if __name__ == "__main__":
    unittest.main()
