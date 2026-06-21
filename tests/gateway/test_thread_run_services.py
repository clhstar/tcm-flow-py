import importlib
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from app.runtime.state import reset_state_to_memory, state
from app.schemas import RunCreateRequest


class ThreadRunServicesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state_to_memory()

    def load_services_with_fake_registry(self):
        fake_registry = ModuleType("app.agents.registry")
        fake_registry.resolve_agent_factory = lambda assistant_id: (
            lambda context: object()
        )
        with patch.dict(sys.modules, {"app.agents.registry": fake_registry}):
            return importlib.import_module("app.runtime.services")

    async def test_start_run_forwards_requested_stream_mode_to_worker_context(self):
        captured_context = {}
        services = self.load_services_with_fake_registry()
        thread = await state.thread_store.create()
        body = RunCreateRequest(
            assistant_id="lead_agent",
            input={
                "messages": [
                    {
                        "type": "human",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ]
            },
            stream_mode=["messages"],
            context={"debug_events": True},
        )

        async def fake_run_agent(*, context, **kwargs):
            captured_context.update(context)

        with (
            patch(
                "app.runtime.services.resolve_agent_factory",
                return_value=lambda context: object(),
            ),
            patch("app.runtime.services.run_agent", new=fake_run_agent),
        ):
            record = await services.start_run(body, thread.thread_id)
            await record.task

        self.assertEqual(captured_context["stream_mode"], ["messages"])
        self.assertTrue(captured_context["debug_events"])

    async def test_start_run_defaults_to_messages_stream_mode(self):
        captured_context = {}
        services = self.load_services_with_fake_registry()
        thread = await state.thread_store.create()
        body = RunCreateRequest(
            assistant_id="lead_agent",
            input={
                "messages": [
                    {
                        "type": "human",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ]
            },
        )

        async def fake_run_agent(*, context, **kwargs):
            captured_context.update(context)

        with (
            patch(
                "app.runtime.services.resolve_agent_factory",
                return_value=lambda context: object(),
            ),
            patch("app.runtime.services.run_agent", new=fake_run_agent),
        ):
            record = await services.start_run(body, thread.thread_id)
            await record.task

        self.assertEqual(captured_context["stream_mode"], ["messages"])


if __name__ == "__main__":
    unittest.main()
