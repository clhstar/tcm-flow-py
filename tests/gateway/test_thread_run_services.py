import importlib
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from app.runtime.runs.context import RunContext
from app.runtime.state import reset_state_to_memory, state
from app.schemas import RunCreateRequest


class ThreadRunServicesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state_to_memory()

    def load_services_with_fake_registry(self):
        existing = sys.modules.get("app.runtime.services")
        if existing is not None:
            return existing
        fake_registry = ModuleType("app.agents.registry")
        fake_registry.resolve_agent_factory = lambda assistant_id: (
            lambda context: object()
        )
        with patch.dict(sys.modules, {"app.agents.registry": fake_registry}):
            services = importlib.import_module("app.runtime.services")
        sys.modules["app.runtime.services"] = services
        return services

    async def test_start_run_separates_normalized_input_context_config_and_stream_modes(self):
        captured = {}
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
            stream_mode=["messages", "values"],
            context={"debug_events": True},
            config={
                "metadata": {"source": "service-test"},
                "custom": {"preserved": True},
            },
        )

        async def fake_run_agent(**kwargs):
            captured.update(kwargs)

        with (
            patch(
                "app.runtime.services.resolve_agent_factory",
                return_value=lambda context: object(),
            ),
            patch("app.runtime.services.run_agent", new=fake_run_agent),
        ):
            record = await services.start_run(body, thread.thread_id)
            await record.task

        self.assertEqual(
            captured["graph_input"],
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertIsInstance(captured["ctx"], RunContext)
        self.assertIs(captured["ctx"].thread_store, state.thread_store)
        self.assertEqual(captured["ctx"].agent_context, {"debug_events": True})
        self.assertNotIn("stream_mode", captured["ctx"].agent_context)
        self.assertEqual(
            captured["config"],
            {
                "metadata": {"source": "service-test"},
                "custom": {"preserved": True},
            },
        )
        self.assertEqual(captured["stream_modes"], ["messages", "values"])

    async def test_start_run_defaults_to_messages_stream_mode(self):
        captured = {}
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

        async def fake_run_agent(**kwargs):
            captured.update(kwargs)

        with (
            patch(
                "app.runtime.services.resolve_agent_factory",
                return_value=lambda context: object(),
            ),
            patch("app.runtime.services.run_agent", new=fake_run_agent),
        ):
            record = await services.start_run(body, thread.thread_id)
            await record.task

        self.assertEqual(captured["stream_modes"], ["messages"])
        self.assertEqual(captured["config"], {})
        self.assertEqual(captured["ctx"].agent_context, {})


if __name__ == "__main__":
    unittest.main()
