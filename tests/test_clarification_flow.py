import unittest
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from app.middlewares.clarification_middleware import ClarificationMiddleware
from app.runtime.runs.worker import message_to_dict, run_agent
from app.runtime.stream import StreamBridge
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore
from app.tools.builtins.clarification_tool import ask_clarification


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

    def make_request(self, question: str):
        return SimpleNamespace(
            tool_call={
                "id": "call-1",
                "name": "ask_clarification",
                "args": {"question": question},
            }
        )

    def test_rejects_empty_question(self):
        with self.assertRaisesRegex(ValueError, "question cannot be empty"):
            self.middleware.handle(self.make_request("   "))

    def test_limits_clarification_to_three_questions(self):
        command = self.middleware.handle(
            self.make_request(
                "这种情况持续多久了？"
                "大便情况怎么样？"
                "是否伴有反酸或烧心？"
                "是否有明显腹痛？"
            )
        )

        message = command.update["messages"][0]

        self.assertIn("1. 这种情况持续多久了？", message.content)
        self.assertIn("2. 大便情况如何？", message.content)
        self.assertIn("3. 是否伴有反酸或烧心？", message.content)
        self.assertNotIn("4.", message.content)
        self.assertNotIn("是否有明显腹痛？", message.content)

    def test_message_serialization_preserves_tool_linkage(self):
        message = ToolMessage(
            id="clarification:call-1",
            content="请补充持续时间。",
            name="ask_clarification",
            tool_call_id="call-1",
        )

        payload = message_to_dict(message)

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
                            "args": {"question": "这种情况持续多久了？"},
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
            thread_store=thread_store,
            record=first_run,
            agent_factory=lambda context: agent,
            input_data={
                "messages": [{"type": "human", "content": "我最近胃胀"}]
            },
            context={},
        )
        first_events = await self.drain_events(bridge, first_run.run_id)

        self.assertEqual(
            (await run_manager.get(first_run.run_id)).status,
            "waiting_clarification",
        )
        self.assertTrue(any("event: clarification" in event for event in first_events))

        second_run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(second_run.run_id)
        await run_agent(
            bridge=bridge,
            run_manager=run_manager,
            thread_store=thread_store,
            record=second_run,
            agent_factory=lambda context: agent,
            input_data={
                "messages": [{"type": "human", "content": "已经两周了"}]
            },
            context={},
        )
        second_events = await self.drain_events(bridge, second_run.run_id)
        stored_thread = await thread_store.get(thread.thread_id)
        messages = stored_thread.values["messages"]

        self.assertEqual((await run_manager.get(second_run.run_id)).status, "success")
        self.assertTrue(any("event: final" in event for event in second_events))
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


if __name__ == "__main__":
    unittest.main()
