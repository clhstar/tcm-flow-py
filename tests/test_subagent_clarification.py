import json
import unittest
from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from app.middlewares.clarification_controller import (
    extract_latest_clarification_question,
)
from app.middlewares.clarification_middleware import ClarificationMiddleware
from app.runtime.runs.worker import run_agent
from app.runtime.stream import StreamBridge
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore
from app.subagents.dynamic_subagent import (
    DynamicSubAgent,
    SubAgentOutput,
    parse_subagent_output,
)
from app.subagents.prompts import DYNAMIC_SUBAGENT_SYSTEM_PROMPT


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return self


@tool("task")
def clarification_task(description: str) -> str:
    """Return a structured subagent result that requires clarification."""
    return json.dumps(
        {
            "type": "subagent_result",
            "needs_clarification": True,
            "clarification_questions": [
                "腰痛持续多久了？",
                "是否伴有腿麻或无力？",
            ],
            "content": "缺少关键信息。",
        },
        ensure_ascii=False,
    )


@tool("task")
def completed_task(description: str) -> str:
    """Return a structured subagent result that can be synthesized."""
    return json.dumps(
        {
            "type": "subagent_result",
            "needs_clarification": False,
            "clarification_questions": [],
            "content": "子任务已经完成，可以继续综合。",
        },
        ensure_ascii=False,
    )


class SubAgentOutputTests(unittest.TestCase):
    def test_schema_requires_all_protocol_fields(self):
        schema = SubAgentOutput.model_json_schema()

        self.assertCountEqual(
            schema["required"],
            ["status", "clarification_questions", "content"],
        )

    def test_parse_structured_missing_information_result(self):
        result = parse_subagent_output(
            json.dumps(
                {
                    "status": "needs_clarification",
                    "clarification_questions": [
                        "腰痛持续多久了？",
                        "是否伴有腿麻或无力？",
                    ],
                    "content": "缺少关键信息，暂不能继续分析。",
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(
            result.clarification_questions,
            ["腰痛持续多久了？", "是否伴有腿麻或无力？"],
        )

    def test_subagent_prompt_forbids_question_numbering(self):
        self.assertIn(
            "clarification_questions 数组中的每个元素只写问题正文",
            DYNAMIC_SUBAGENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "不要添加序号",
            DYNAMIC_SUBAGENT_SYSTEM_PROMPT,
        )

    def test_rejects_free_text_output(self):
        with self.assertRaisesRegex(ValueError, "必须是合法的 JSON 对象"):
            parse_subagent_output("缺少关键信息，请补充腰痛持续时间。")

    def test_rejects_unknown_status(self):
        with self.assertRaisesRegex(ValueError, "status 只能是"):
            parse_subagent_output(
                json.dumps(
                    {
                        "status": "pending",
                        "clarification_questions": [],
                        "content": "等待更多信息。",
                    },
                    ensure_ascii=False,
                )
            )

    def test_rejects_missing_clarification_questions(self):
        with self.assertRaisesRegex(
            ValueError,
            "needs_clarification 必须包含 1 到 3 个澄清问题",
        ):
            parse_subagent_output(
                json.dumps(
                    {
                        "status": "needs_clarification",
                        "clarification_questions": [],
                        "content": "缺少关键信息。",
                    },
                    ensure_ascii=False,
                )
            )

    def test_rejects_more_than_three_clarification_questions(self):
        with self.assertRaisesRegex(
            ValueError,
            "needs_clarification 必须包含 1 到 3 个澄清问题",
        ):
            parse_subagent_output(
                json.dumps(
                    {
                        "status": "needs_clarification",
                        "clarification_questions": [
                            "问题1？",
                            "问题2？",
                            "问题3？",
                            "问题4？",
                        ],
                        "content": "缺少关键信息。",
                    },
                    ensure_ascii=False,
                )
            )

    def test_rejects_questions_for_completed_result(self):
        with self.assertRaisesRegex(
            ValueError,
            "completed 状态不能包含澄清问题",
        ):
            parse_subagent_output(
                json.dumps(
                    {
                        "status": "completed",
                        "clarification_questions": ["还需要补充吗？"],
                        "content": "子任务已经完成。",
                    },
                    ensure_ascii=False,
                )
            )

    def test_task_result_requests_clarification(self):
        content = json.dumps(
            {
                "type": "subagent_result",
                "needs_clarification": True,
                "clarification_questions": [
                    "腰痛持续多久了？",
                    "是否伴有腿麻或无力？",
                ],
                "content": "缺少关键信息。",
            },
            ensure_ascii=False,
        )

        question = extract_latest_clarification_question(
            [
                {
                    "type": "tool",
                    "name": "task",
                    "content": content,
                }
            ]
        )

        self.assertIn("1. 腰痛持续多久了？", question)
        self.assertIn("2. 是否伴有腿麻或无力？", question)


class DynamicSubAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_uses_deepseek_json_mode(self):
        model = Mock()
        structured_model = AsyncMock()
        structured_model.ainvoke.return_value = SubAgentOutput(
            status="completed",
            clarification_questions=[],
            content="子任务结果",
        )
        model.with_structured_output.return_value = structured_model
        agent = DynamicSubAgent()

        with patch.object(agent, "get_model", return_value=model):
            result = await agent.run(
                description="整理已有信息",
                expected_output="输出分析要点",
            )

        model.with_structured_output.assert_called_once_with(
            SubAgentOutput,
            method="json_mode",
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.content, "子任务结果")
        self.assertEqual(result.task_description, "整理已有信息")


class SubAgentClarificationRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_result_stops_run_before_next_model_call(self):
        model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "task-call-1",
                            "name": "task",
                            "args": {"description": "整理缺失信息"},
                        }
                    ],
                ),
                AIMessage(content="这条最终回答不应该被执行。"),
            ]
        )
        agent = create_agent(
            model=model,
            tools=[clarification_task],
            middleware=[ClarificationMiddleware()],
            checkpointer=InMemorySaver(),
        )
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        await run_agent(
            bridge=bridge,
            run_manager=run_manager,
            thread_store=thread_store,
            record=run,
            agent_factory=lambda context: agent,
            input_data={
                "messages": [{"type": "human", "content": "请帮我分析腰痛"}]
            },
            context={},
        )

        stored_thread = await thread_store.get(thread.thread_id)
        messages = stored_thread.values["messages"]
        snapshot = await agent.aget_state(
            {"configurable": {"thread_id": thread.thread_id}}
        )

        self.assertEqual(
            (await run_manager.get(run.run_id)).status,
            "waiting_clarification",
        )
        self.assertEqual(messages[-1]["name"], "task")
        self.assertNotIn(
            "这条最终回答不应该被执行。",
            [message.get("content") for message in messages],
        )
        self.assertIn(
            "腰痛持续多久了？",
            stored_thread.values["conversation"][-1]["content"],
        )
        self.assertEqual(snapshot.next, ())

    async def test_completed_task_continues_to_lead_agent(self):
        model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "task-call-2",
                            "name": "task",
                            "args": {"description": "整理已有信息"},
                        }
                    ],
                ),
                AIMessage(content="Lead Agent 已完成综合回答。"),
            ]
        )
        agent = create_agent(
            model=model,
            tools=[completed_task],
            middleware=[ClarificationMiddleware()],
            checkpointer=InMemorySaver(),
        )
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        await run_agent(
            bridge=bridge,
            run_manager=run_manager,
            thread_store=thread_store,
            record=run,
            agent_factory=lambda context: agent,
            input_data={
                "messages": [{"type": "human", "content": "请整理已有信息"}]
            },
            context={},
        )

        stored_thread = await thread_store.get(thread.thread_id)

        self.assertEqual((await run_manager.get(run.run_id)).status, "success")
        self.assertEqual(
            stored_thread.values["conversation"][-1]["content"],
            "Lead Agent 已完成综合回答。",
        )


if __name__ == "__main__":
    unittest.main()
