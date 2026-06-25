import unittest

from app.agents.workflow_agent.components.base import StructuredWorkflowComponent
from app.agents.workflow_agent.models import InquiryState


class FakeStructuredRunnable:
    def __init__(self):
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return InquiryState(
            chief_complaint="stomach distension",
            information_sufficiency="sufficient",
        )


class FakeStructuredModel:
    def __init__(self):
        self.structured = FakeStructuredRunnable()

    def with_structured_output(self, schema=None, *, method, strict=None):
        return self.structured


class TestWorkflowComponent(StructuredWorkflowComponent[InquiryState]):
    schema = InquiryState
    system_prompt = "Collect inquiry facts."


class WorkflowAgentComponentBoundaryTests(unittest.TestCase):
    def test_agent_components_live_in_dedicated_modules(self):
        from app.agents.workflow_agent.components.answer import AnswerAgent
        from app.agents.workflow_agent.components.evidence import EvidenceAgent
        from app.agents.workflow_agent.components.inquiry import InquiryAgent
        from app.agents.workflow_agent.components.safety import SafetyAgent
        from app.agents.workflow_agent.components.syndrome import SyndromeAgent

        self.assertEqual(
            InquiryAgent.__module__,
            "app.agents.workflow_agent.components.inquiry",
        )
        self.assertEqual(
            EvidenceAgent.__module__,
            "app.agents.workflow_agent.components.evidence",
        )
        self.assertEqual(
            SyndromeAgent.__module__,
            "app.agents.workflow_agent.components.syndrome",
        )
        self.assertEqual(
            AnswerAgent.__module__,
            "app.agents.workflow_agent.components.answer",
        )
        self.assertEqual(
            SafetyAgent.__module__,
            "app.agents.workflow_agent.components.safety",
        )


class WorkflowAgentComponentPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_component_prompt_contains_json_keyword(self):
        model = FakeStructuredModel()
        component = TestWorkflowComponent(model)

        await component.invoke_structured({"field": "value"})

        content = "\n".join(message["content"] for message in model.structured.messages)
        self.assertIn("json", content.lower())


if __name__ == "__main__":
    unittest.main()
