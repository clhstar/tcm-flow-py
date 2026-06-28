import unittest
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from app.middlewares.clarification_middleware import ClarificationMiddleware
from app.middlewares.clarification_controller import normalize_question_items
from app.agents.lead_agent.prompt import SYSTEM_PROMPT
from app.runtime.runs.context import RunContext
from app.runtime.runs.input import normalize_graph_input
from app.runtime.runs.worker import run_agent
from app.runtime.serialization import serialize_message
from app.runtime.stream import StreamBridge
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


@tool("ask_clarification", return_direct=True)
def ask_clarification(questions: list[str]) -> str:
    """Return formatted clarification questions for tests."""
    lines = ["请补充："]
    lines.extend(
        f"{index}. {question}" for index, question in enumerate(questions, start=1)
    )
    return "\n".join(lines)


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return self


class ClarificationMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.middleware = ClarificationMiddleware()

    def make_request(self, questions: Any):
        return SimpleNamespace(
            tool_call={
                "id": "call-1",
                "name": "ask_clarification",
                "args": {"questions": questions},
            }
        )

    def test_rejects_empty_questions(self):
        with self.assertRaisesRegex(ValueError, "澄清问题不能为空"):
            self.middleware.handle(self.make_request([]))

    def test_rejects_more_than_three_questions(self):
        with self.assertRaisesRegex(ValueError, "澄清问题不能多于3个"):
            self.middleware.handle(
                self.make_request(
                    [
                        "这种情况持续多久了？",
                        "是否伴有腿麻？",
                        "是否有大小便异常？",
                        "近期是否受过外伤？",
                    ]
                )
            )

    def test_formats_question_bodies_with_display_numbering(self):
        command = self.middleware.handle(
            self.make_request(
                [
                    "这种情况持续多久了？",
                    "是否伴有腿麻？",
                    "是否有大小便异常？",
                ]
            )
        )

        message = command.update["messages"][0]

        self.assertIn("1. 这种情况持续多久了？", message.content)
        self.assertIn("2. 是否伴有腿麻？", message.content)
        self.assertIn("3. 是否有大小便异常？", message.content)

    def test_lead_agent_prompt_forbids_question_numbering(self):
        self.assertIn(
            "questions 数组中的每个元素只写问题正文",
            SYSTEM_PROMPT,
        )
        self.assertIn("不要添加序号", SYSTEM_PROMPT)

    def test_program_does_not_strip_numbering_from_question_body(self):
        self.assertEqual(
            normalize_question_items(["1. 这种情况持续多久了？"]),
            ["1. 这种情况持续多久了？"],
        )

    def test_message_serialization_preserves_tool_linkage(self):
        message = ToolMessage(
            id="clarification:call-1",
            content="请补充持续时间。",
            name="ask_clarification",
            tool_call_id="call-1",
        )

        payload = serialize_message(message)

        self.assertEqual(payload["id"], "clarification:call-1")
        self.assertEqual(payload["tool_call_id"], "call-1")
        self.assertEqual(payload["name"], "ask_clarification")


class ClarificationRunTests(unittest.IsolatedAsyncioTestCase):
    async def drain_events(self, bridge: StreamBridge, run_id: str) -> list[str]:
        events = []
        async for event in bridge.subscribe(run_id):
            events.append(event)
        return events

    async def test_follow_up_in_same_thread_preserves_clarification_history(self):
        model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "ask_clarification",
                            "args": {"questions": ["这种情况持续多久了？"]},
                        }
                    ],
                ),
                AIMessage(content="收到，你已经持续两周。"),
            ]
        )
        agent = create_agent(
            model=model,
            tools=[ask_clarification],
            middleware=[ClarificationMiddleware()],
            checkpointer=InMemorySaver(),
        )
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()

        first_run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(first_run.run_id)
        await run_agent(
            bridge=bridge,
            run_manager=run_manager,
            record=first_run,
            ctx=RunContext(thread_store=thread_store),
            agent_factory=lambda context: agent,
            graph_input=normalize_graph_input(
                {"messages": [{"type": "human", "content": "我最近胃胀"}]}
            ),
            config={},
        )
        first_events = await self.drain_events(bridge, first_run.run_id)

        self.assertEqual(
            (await run_manager.get(first_run.run_id)).status,
            "waiting_clarification",
        )
        self.assertFalse(
            any("event: clarification" in event for event in first_events)
        )
        self.assertTrue(any("event: final" in event for event in first_events))

        second_run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(second_run.run_id)
        await run_agent(
            bridge=bridge,
            run_manager=run_manager,
            record=second_run,
            ctx=RunContext(thread_store=thread_store),
            agent_factory=lambda context: agent,
            graph_input=normalize_graph_input(
                {"messages": [{"type": "human", "content": "已经两周了"}]}
            ),
            config={},
        )
        second_events = await self.drain_events(bridge, second_run.run_id)
        stored_thread = await thread_store.get(thread.thread_id)
        snapshot = await agent.aget_state(
            {"configurable": {"thread_id": thread.thread_id}}
        )
        messages = [
            serialize_message(message) for message in snapshot.values["messages"]
        ]

        self.assertEqual((await run_manager.get(second_run.run_id)).status, "success")
        self.assertTrue(any("event: final" in event for event in second_events))
        self.assertEqual(stored_thread.values["messages"], messages)
        self.assertEqual(
            [(message["type"], message.get("name")) for message in messages],
            [
                ("human", None),
                ("ai", None),
                ("tool", "ask_clarification"),
                ("human", None),
                ("ai", None),
            ],
        )
        self.assertEqual(messages[2]["id"], "clarification:call-1")
        self.assertEqual(messages[2]["tool_call_id"], "call-1")

    async def test_guardrail_rewrite_replaces_checkpoint_history(self):
        model = ToolCallingFakeModel(
            responses=[AIMessage(content="原始且不应保留的病机答案")]
        )
        agent = create_agent(
            model=model,
            tools=[],
            checkpointer=InMemorySaver(),
        )
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        guardrail_result = {
            "final_text": "经过 Guardrail 的安全答案",
            "validation": {"passed": True},
            "validation_before_rewrite": {"passed": False},
            "rewritten": True,
            "allowed_terms": [],
        }

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(return_value=guardrail_result),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                record=run,
                ctx=RunContext(thread_store=thread_store),
                agent_factory=lambda context: agent,
                graph_input=normalize_graph_input(
                    {"messages": [{"type": "human", "content": "请帮我分析"}]}
                ),
                config={},
            )

        snapshot = await agent.aget_state(
            {"configurable": {"thread_id": thread.thread_id}}
        )
        checkpoint_messages = snapshot.values["messages"]
        stored_thread = await thread_store.get(thread.thread_id)

        self.assertEqual(
            checkpoint_messages[-1].content,
            "经过 Guardrail 的安全答案",
        )
        self.assertEqual(
            stored_thread.values["messages"],
            [serialize_message(message) for message in checkpoint_messages],
        )
        self.assertNotIn(
            "原始且不应保留的病机答案",
            [message.content for message in checkpoint_messages],
        )


if __name__ == "__main__":
    unittest.main()
