import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage

from app.runtime.runs.worker import run_agent
from app.runtime.stream import StreamBridge
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


class FakeAsyncAgent:
    async def astream(self, input_data, *, config, stream_mode):
        yield {"messages": [AIMessage(content="done")]}


class AsyncAgentFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def drain_events(self, bridge: StreamBridge, run_id: str) -> list[str]:
        events = []
        async for event in bridge.subscribe(run_id):
            events.append(event)
        return events

    async def test_run_agent_awaits_async_agent_factory(self):
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        async def async_factory(context):
            return FakeAsyncAgent()

        with patch(
            "app.runtime.runs.worker.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "done",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": [],
                }
            ),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                thread_store=thread_store,
                record=run,
                agent_factory=async_factory,
                input_data={"messages": [{"type": "human", "content": "hello"}]},
                context={},
            )

        events = await self.drain_events(bridge, run.run_id)

        self.assertEqual((await run_manager.get(run.run_id)).status, "success")
        self.assertTrue(any("event: final" in event for event in events))


if __name__ == "__main__":
    unittest.main()
