import inspect
import unittest
from unittest.mock import patch

from app.agents.lead_agent.agent import build_lead_agent, make_lead_agent
from app.agents.lead_agent.state import LeadAgentState
from app.config import AppSettings
from app.runtime import state as runtime_state


class LeadAgentFactoryTests(unittest.TestCase):
    def make_settings(self):
        return AppSettings(
            database_url=None,
            postgres_pool_size=10,
            checkpoint_backend="memory",
            elasticsearch_url=None,
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
            openai_model="settings-model",
            openai_base_url="https://settings.example/v1",
            openai_api_key="settings-key",
        )

    def test_build_lead_agent_uses_explicit_checkpointer_settings_and_state_schema(self):
        checkpointer = object()
        created = object()

        with patch(
            "app.agents.lead_agent.agent.get_settings",
            return_value=self.make_settings(),
        ):
            with patch(
                "app.agents.lead_agent.agent.ChatOpenAI",
                return_value=object(),
            ) as chat_openai:
                with patch(
                    "app.agents.lead_agent.agent.get_available_tools",
                    return_value=[],
                ):
                    with patch(
                        "app.agents.lead_agent.agent.create_agent",
                        return_value=created,
                    ) as create_agent:
                        agent = build_lead_agent(
                            {"temperature": 0.1},
                            checkpointer=checkpointer,
                        )

        self.assertIs(agent, created)
        self.assertEqual(chat_openai.call_args.kwargs["model"], "settings-model")
        self.assertEqual(
            chat_openai.call_args.kwargs["base_url"],
            "https://settings.example/v1",
        )
        self.assertEqual(chat_openai.call_args.kwargs["api_key"], "settings-key")
        self.assertEqual(chat_openai.call_args.kwargs["temperature"], 0.1)
        self.assertIs(chat_openai.call_args.kwargs["streaming"], True)
        self.assertIs(create_agent.call_args.kwargs["checkpointer"], checkpointer)
        self.assertIs(create_agent.call_args.kwargs["state_schema"], LeadAgentState)

    def test_context_model_overrides_settings_model(self):
        with patch(
            "app.agents.lead_agent.agent.get_settings",
            return_value=self.make_settings(),
        ):
            with patch(
                "app.agents.lead_agent.agent.ChatOpenAI",
                return_value=object(),
            ) as chat_openai:
                with patch(
                    "app.agents.lead_agent.agent.get_available_tools",
                    return_value=[],
                ):
                    with patch(
                        "app.agents.lead_agent.agent.create_agent",
                        return_value=object(),
                    ):
                        build_lead_agent(
                            {"model_name": "context-model"},
                            checkpointer=object(),
                        )

        self.assertEqual(chat_openai.call_args.kwargs["model"], "context-model")

    def test_make_lead_agent_sync_uses_shared_runtime_checkpointer(self):
        checkpointer = object()
        created = object()
        original_checkpointer = runtime_state.state.checkpointer
        runtime_state.state.checkpointer = checkpointer

        try:
            with patch(
                "app.agents.lead_agent.agent.build_lead_agent",
                return_value=created,
            ) as build_agent:
                agent = make_lead_agent({})
        finally:
            runtime_state.state.checkpointer = original_checkpointer

        self.assertIs(agent, created)
        self.assertFalse(inspect.isawaitable(agent))
        self.assertIs(build_agent.call_args.kwargs["checkpointer"], checkpointer)


if __name__ == "__main__":
    unittest.main()
