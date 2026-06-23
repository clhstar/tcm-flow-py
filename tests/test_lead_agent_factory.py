import unittest
from unittest.mock import patch

from app.agents.lead_agent.agent import make_lead_agent
from app.runtime import state as runtime_state


class LeadAgentFactoryTests(unittest.TestCase):
    def test_lead_agent_uses_shared_runtime_checkpointer(self):
        checkpointer = object()
        created = object()
        original_checkpointer = runtime_state.state.checkpointer
        runtime_state.state.checkpointer = checkpointer

        try:
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
        finally:
            runtime_state.state.checkpointer = original_checkpointer

        self.assertIs(agent, created)
        self.assertIs(create_agent.call_args.kwargs["checkpointer"], checkpointer)
        self.assertIs(chat_openai.call_args.kwargs["streaming"], True)


if __name__ == "__main__":
    unittest.main()
