import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.agents.workflow_agent.llm import build_workflow_model, structured_model
from app.agents.workflow_agent.models import InquiryState
from app.agents.workflow_agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    INQUIRY_SYSTEM_PROMPT,
    SAFETY_SYSTEM_PROMPT,
    SYNDROME_SYSTEM_PROMPT,
)


class WorkflowAgentLLMTests(unittest.TestCase):
    def test_structured_model_uses_json_schema_strict_format_for_openai_models(self):
        settings = SimpleNamespace(
            openai_model="gpt-4.1-mini",
            openai_base_url=None,
        )
        model = Mock()
        with patch("app.agents.workflow_agent.llm.get_settings", return_value=settings):
            structured_model(model, InquiryState)

        model.with_structured_output.assert_called_once_with(
            InquiryState,
            method="json_schema",
            strict=True,
        )

    def test_structured_model_uses_json_mode_for_deepseek_backends(self):
        settings = SimpleNamespace(
            openai_model="deepseek-chat",
            openai_base_url="https://api.deepseek.com/v1",
        )
        model = Mock()
        with patch("app.agents.workflow_agent.llm.get_settings", return_value=settings):
            structured_model(model, InquiryState)

        model.with_structured_output.assert_called_once_with(
            method="json_mode",
        )

    def test_structured_model_prefers_model_instance_over_default_settings(self):
        settings = SimpleNamespace(
            openai_model="deepseek-chat",
            openai_base_url=None,
        )
        model = Mock()
        model.model_name = "gpt-4.1-mini"
        model.base_url = None
        with patch("app.agents.workflow_agent.llm.get_settings", return_value=settings):
            structured_model(model, InquiryState)

        model.with_structured_output.assert_called_once_with(
            InquiryState,
            method="json_schema",
            strict=True,
        )

    def test_build_workflow_model_uses_settings_and_context(self):
        settings = SimpleNamespace(
            openai_model="settings-model",
            openai_base_url="https://example.test/v1",
            openai_api_key="test-key",
        )

        with (
            patch("app.agents.workflow_agent.llm.get_settings", return_value=settings),
            patch("app.agents.workflow_agent.llm.ChatOpenAI") as chat_openai,
        ):
            build_workflow_model(
                {
                    "model_name": "context-model",
                    "temperature": 0.1,
                    "streaming": True,
                }
            )

        chat_openai.assert_called_once_with(
            model="context-model",
            base_url="https://example.test/v1",
            api_key="test-key",
            temperature=0.1,
            streaming=True,
        )

    def test_prompts_preserve_agent_boundaries(self):
        self.assertIn("只负责整理问诊信息", INQUIRY_SYSTEM_PROMPT)
        self.assertIn("possible_patterns 只能使用输入 allowed_terms", SYNDROME_SYSTEM_PROMPT)
        self.assertIn("不新增术语", ANSWER_SYSTEM_PROMPT)
        self.assertIn("检查初稿是否直接诊断", SAFETY_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
