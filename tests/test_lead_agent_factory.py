import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.lead_agent.agent import make_lead_agent


class LeadAgentFactoryTests(unittest.TestCase):
    def test_memory_lead_agent_uses_configured_checkpointer(self):
        settings = SimpleNamespace(checkpoint_backend="memory")
        checkpointer = object()
        created = object()

        with patch("app.agents.lead_agent.agent.get_settings", return_value=settings):
            with patch(
                "app.agents.lead_agent.agent.get_checkpointer",
                return_value=checkpointer,
            ) as get_checkpointer:
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
                            agent = make_lead_agent({})

        self.assertIs(agent, created)
        get_checkpointer.assert_called_once_with(settings)
        self.assertIs(create_agent.call_args.kwargs["checkpointer"], checkpointer)
        self.assertIs(chat_openai.call_args.kwargs["streaming"], True)


if __name__ == "__main__":
    unittest.main()
