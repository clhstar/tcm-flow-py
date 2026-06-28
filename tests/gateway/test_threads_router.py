import json
import unittest
from collections.abc import Callable, Sequence
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from unittest.mock import AsyncMock, patch

from app.gateway.routers import threads
from app.middlewares.clarification_middleware import ClarificationMiddleware
from app.runtime.public_messages import extract_pending_clarification
from app.runtime.runs.context import RunContext
from app.runtime.runs.input import normalize_graph_input
from app.runtime.runs.worker import run_agent
from app.runtime.stream import StreamBridge
from app.runtime.state import reset_state_to_memory, state
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return self


@tool("ask_clarification", return_direct=True)
def ask_clarification(questions: list[str]) -> str:
    """Return formatted clarification questions for tests."""
    lines = ["Please provide:"]
    lines.extend(
        f"{index}. {question}" for index, question in enumerate(questions, start=1)
    )
    return "\n".join(lines)


class ThreadsRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        reset_state_to_memory()
        app = FastAPI()
        app.include_router(threads.router)
        self.client = TestClient(app)

    async def test_history_preserves_complete_message_chain(self):
        thread = await state.thread_store.create()
        messages = [
            {
                "type": "human",
                "content": "I wake up with a headache",
                "id": "human-1",
            },
            {
                "type": "ai",
                "content": "",
                "id": "ai-tool",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "retrieve_tcm_knowledge",
                        "args": {"query": "morning headache"},
                    }
                ],
            },
            {
                "type": "tool",
                "content": "retrieval result",
                "id": "tool-1",
                "name": "retrieve_tcm_knowledge",
                "tool_call_id": "call-1",
            },
            {
                "type": "ai",
                "content": "The symptom needs further evaluation.",
                "id": "ai-final",
            },
        ]
        await state.thread_store.update_values(
            thread.thread_id,
            {
                "messages": messages,
                "last_validation": {"passed": True},
                "last_allowed_terms": ["headache"],
                "last_rewritten": False,
                "last_agent_trace": [{"agent": "SafetyAgent"}],
            },
        )

        response = self.client.get(f"/api/threads/{thread.thread_id}/history")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["messages"], messages)
        self.assertTrue(
            {
                "last_validation",
                "last_allowed_terms",
                "last_rewritten",
                "last_agent_trace",
            }.isdisjoint(body)
        )

    async def test_history_preserves_enriched_conversation_and_raw_messages(self):
        thread = await state.thread_store.create()
        conversation = [
            {"role": "user", "content": "最近头痛"},
            {
                "role": "assistant",
                "content": "请补充持续时间。",
                "run_id": "run-1",
                "agent_trace": [{"agent": "InquiryAgent"}],
            },
        ]
        messages = [
            {"type": "human", "content": "最近头痛", "id": "human-1"},
            {
                "type": "ai",
                "content": "请补充持续时间。",
                "id": "ai-1",
            },
        ]
        await state.thread_store.update_values(
            thread.thread_id,
            {"conversation": conversation, "messages": messages},
        )

        response = self.client.get(f"/api/threads/{thread.thread_id}/history")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["conversation"], conversation)
        self.assertEqual(body["messages"], messages)


class PublicMessageProjectionTests(unittest.TestCase):
    def test_task_tool_payload_is_not_used_for_pending_clarification(self):
        messages = [
            {
                "type": "tool",
                "name": "task",
                "content": json.dumps(
                    {
                        "needs_clarification": True,
                        "clarification_questions": [
                            "How long has the headache lasted?"
                        ],
                    }
                ),
            }
        ]

        self.assertIsNone(extract_pending_clarification(messages))


class ThreadRunStreamMessageTests(unittest.IsolatedAsyncioTestCase):
    async def drain_events(self, bridge: StreamBridge, run_id: str) -> list[str]:
        events = []
        async for event in bridge.subscribe(run_id):
            events.append(event)
        return events

    def parse_events(self, events: list[str]) -> list[tuple[str, Any]]:
        parsed = []
        for raw_event in events:
            event_name = ""
            data = ""
            for line in raw_event.splitlines():
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data = line[len("data:"):].strip()
            if event_name and data:
                parsed.append((event_name, json.loads(data)))
        return parsed

    async def test_clarification_result_is_published_as_final_not_custom_event(self):
        model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="Thanks, I need a few details first:",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "ask_clarification",
                            "args": {
                                "questions": [
                                    "How long has the headache lasted?",
                                    "Where is the headache located?",
                                    "Any nausea or dizziness?",
                                ]
                            },
                        }
                    ],
                )
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
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        await run_agent(
            bridge=bridge,
            run_manager=run_manager,
            record=run,
            ctx=RunContext(thread_store=thread_store),
            agent_factory=lambda context: agent,
            graph_input=normalize_graph_input({
                "messages": [
                    {"type": "human", "content": "I wake up with a headache"}
                ]
            }),
            config={},
        )

        events = await self.drain_events(bridge, run.run_id)
        parsed_events = self.parse_events(events)
        event_names = [event for event, _ in parsed_events]
        values_events = [data for event, data in parsed_events if event == "values"]
        final_events = [data for event, data in parsed_events if event == "final"]

        self.assertNotIn("values", event_names)
        self.assertNotIn("clarification", event_names)
        self.assertNotIn("agent_step", event_names)
        self.assertTrue(final_events)
        self.assertEqual(final_events[-1]["status"], "need_clarification")
        self.assertEqual(
            final_events[-1]["assistant_message"],
            (
                "Thanks, I need a few details first:\n"
                "1. How long has the headache lasted?\n"
                "2. Where is the headache located?\n"
                "3. Any nausea or dizziness?"
            ),
        )
        self.assertEqual(
            final_events[-1]["pending_clarification"],
            [
                "How long has the headache lasted?",
                "Where is the headache located?",
                "Any nausea or dizziness?",
            ],
        )
        self.assertNotIn("messages", final_events[-1])
        self.assertNotIn("tool_calls", final_events[-1])

    async def test_requested_values_publish_full_state_snapshot(self):
        class FakeAgent:
            async def astream(self, input_data, *, config, stream_mode):
                self.seen_stream_mode = stream_mode
                yield (
                    "values",
                    {
                        "messages": [
                            HumanMessage(content="current user", id="human-new"),
                            AIMessage(
                                content="",
                                id="ai-tool",
                                tool_calls=[
                                    {
                                        "id": "call-1",
                                        "name": "retrieve_tcm_knowledge",
                                        "args": {"query": "stomach pain"},
                                    }
                                ],
                                response_metadata={"finish_reason": "tool_calls"},
                            ),
                            ToolMessage(
                                content="retrieval result",
                                id="tool-1",
                                name="retrieve_tcm_knowledge",
                                tool_call_id="call-1",
                            ),
                            AIMessage(content="current answer", id="ai-final"),
                        ],
                        "usage_metadata": {"total_tokens": 42},
                        "__pregel_internal": "hidden",
                    },
                )

        fake_agent = FakeAgent()
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "current answer",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": [],
                }
            ),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                record=run,
                ctx=RunContext(thread_store=thread_store),
                agent_factory=lambda context: fake_agent,
                graph_input=normalize_graph_input({
                    "messages": [{"type": "human", "content": "current user"}]
                }),
                config={},
                stream_modes=["messages", "values"],
            )

        parsed_events = self.parse_events(await self.drain_events(bridge, run.run_id))
        values_events = [data for event, data in parsed_events if event == "values"]

        self.assertEqual(fake_agent.seen_stream_mode, ["messages", "values"])
        self.assertTrue(values_events)
        values_payload = values_events[-1]
        self.assertIn("messages", values_payload)
        self.assertEqual(values_payload["usage_metadata"], {"total_tokens": 42})
        self.assertNotIn("__pregel_internal", values_payload)
        self.assertEqual(values_payload["messages"][1]["type"], "ai")
        self.assertEqual(
            values_payload["messages"][1]["tool_calls"][0]["name"],
            "retrieve_tcm_knowledge",
        )
        self.assertEqual(values_payload["messages"][2]["type"], "tool")
        self.assertEqual(values_payload["messages"][2]["tool_call_id"], "call-1")

    async def test_values_only_request_still_subscribes_messages_internally(self):
        class FakeAgent:
            async def astream(self, input_data, *, config, stream_mode):
                self.seen_stream_mode = stream_mode
                yield (
                    "values",
                    {
                        "messages": [
                            HumanMessage(content="current user", id="human-new"),
                            AIMessage(content="current answer", id="ai-final"),
                        ]
                    },
                )

        fake_agent = FakeAgent()
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "current answer",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": [],
                }
            ),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                record=run,
                ctx=RunContext(thread_store=thread_store),
                agent_factory=lambda context: fake_agent,
                graph_input=normalize_graph_input({
                    "messages": [{"type": "human", "content": "current user"}]
                }),
                config={},
                stream_modes=["values"],
            )

        self.assertEqual(fake_agent.seen_stream_mode, ["messages", "values"])

    async def test_debug_trace_events_are_published_as_updates(self):
        class FakeAgent:
            async def astream(self, input_data, *, config, stream_mode):
                self.seen_stream_mode = stream_mode
                yield (
                    "values",
                    {
                        "messages": [
                            HumanMessage(content="current user", id="human-new"),
                            AIMessage(
                                content="",
                                id="ai-tool",
                                tool_calls=[
                                    {
                                        "id": "call-1",
                                        "name": "retrieve_tcm_knowledge",
                                        "args": {"query": "stomach pain"},
                                    }
                                ],
                            ),
                            ToolMessage(
                                content="检索模式：hybrid\n原始检索问题：stomach pain",
                                id="tool-1",
                                name="retrieve_tcm_knowledge",
                                tool_call_id="call-1",
                            ),
                            AIMessage(content="current answer", id="ai-final"),
                        ]
                    },
                )

        fake_agent = FakeAgent()
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "current answer",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": [],
                }
            ),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                record=run,
                ctx=RunContext(
                    thread_store=thread_store,
                    agent_context={"debug_events": True},
                ),
                agent_factory=lambda context: fake_agent,
                graph_input=normalize_graph_input({
                    "messages": [{"type": "human", "content": "current user"}]
                }),
                config={},
                stream_modes=["messages"],
            )

        parsed_events = self.parse_events(await self.drain_events(bridge, run.run_id))
        event_names = [event for event, _ in parsed_events]
        allowed_events = {
            "metadata",
            "messages",
            "updates",
            "values",
            "final",
            "error",
            "end",
        }

        self.assertEqual(fake_agent.seen_stream_mode, ["messages", "values"])
        self.assertLessEqual(set(event_names), allowed_events)
        self.assertIn("updates", event_names)
        self.assertIn("values", event_names)
        self.assertNotIn("agent_step", event_names)
        self.assertNotIn("tool_result", event_names)
        self.assertNotIn("clarification", event_names)

    async def test_final_event_returns_only_current_business_response(self):
        class FakeAgent:
            async def astream(self, input_data, *, config, stream_mode):
                yield {
                    "messages": [
                        HumanMessage(content="previous user", id="human-old"),
                        AIMessage(content="previous answer", id="ai-old"),
                        HumanMessage(content="current user", id="human-new"),
                        AIMessage(content="current answer", id="ai-new"),
                    ]
                }

        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        await thread_store.update_values(
            thread.thread_id,
            {
                "conversation": [
                    {"role": "user", "content": "previous user"},
                    {"role": "assistant", "content": "previous answer"},
                ],
            },
        )
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "current answer",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": [],
                }
            ),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                record=run,
                ctx=RunContext(thread_store=thread_store),
                agent_factory=lambda context: FakeAgent(),
                graph_input=normalize_graph_input({
                    "messages": [{"type": "human", "content": "current user"}]
                }),
                config={},
            )

        parsed_events = self.parse_events(await self.drain_events(bridge, run.run_id))
        final_events = [data for event, data in parsed_events if event == "final"]
        stored_thread = await thread_store.get(thread.thread_id)

        self.assertTrue(final_events)
        final_payload = final_events[-1]
        self.assertEqual(
            final_payload,
            {
                "thread_id": thread.thread_id,
                "run_id": run.run_id,
                "status": "completed",
                "assistant_message": "current answer",
                "pending_clarification": None,
                "references": [],
            },
        )
        self.assertNotIn("previous answer", json.dumps(final_payload))
        self.assertNotIn("messages", final_payload)
        self.assertNotIn("conversation", final_payload)
        self.assertNotIn("validation", final_payload)
        self.assertNotIn("agent_trace", final_payload)
        self.assertNotIn("agent_step", [event for event, _ in parsed_events])
        self.assertEqual(
            [message.get("id") for message in stored_thread.values["messages"]],
            ["human-old", "ai-old", "human-new", "ai-new"],
        )
        self.assertEqual(
            stored_thread.values["conversation"][-2:],
            [
                {"role": "user", "content": "current user"},
                {
                    "role": "assistant",
                    "content": "current answer",
                    "run_id": run.run_id,
                },
            ],
        )
        self.assertTrue(
            {
                "last_validation",
                "last_allowed_terms",
                "last_rewritten",
                "last_agent_trace",
            }.isdisjoint(stored_thread.values)
        )

    async def test_requested_messages_stream_emits_deerflow_message_chunks(self):
        class FakeAgent:
            def __init__(self):
                self.seen_stream_mode = None

            async def astream(self, input_data, *, config, stream_mode):
                self.seen_stream_mode = stream_mode
                messages = [
                    HumanMessage(content="current user", id="human-new"),
                    AIMessage(content="你好", id="ai-new"),
                ]

                if isinstance(stream_mode, list) and "messages" in stream_mode:
                    yield (
                        "messages",
                        (
                            AIMessageChunk(content="你", id="chunk-1"),
                            {
                                "langgraph_node": "model",
                                "thread_id": config["configurable"]["thread_id"],
                            },
                        ),
                    )
                    yield ("values", {"messages": messages})
                    return

                yield {"messages": messages}

        fake_agent = FakeAgent()
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "你好",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": [],
                }
            ),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                record=run,
                ctx=RunContext(thread_store=thread_store),
                agent_factory=lambda context: fake_agent,
                graph_input=normalize_graph_input({
                    "messages": [{"type": "human", "content": "current user"}]
                }),
                config={},
                stream_modes=["messages"],
            )

        parsed_events = self.parse_events(await self.drain_events(bridge, run.run_id))
        message_events = [
            data for event, data in parsed_events if event == "messages"
        ]
        final_events = [data for event, data in parsed_events if event == "final"]

        self.assertIsInstance(fake_agent.seen_stream_mode, list)
        self.assertIn("messages", fake_agent.seen_stream_mode)
        self.assertIn("values", fake_agent.seen_stream_mode)
        self.assertTrue(message_events)

        chunk_payload, metadata = message_events[0]
        self.assertEqual(chunk_payload["type"], "AIMessageChunk")
        self.assertEqual(chunk_payload["content"], "你")
        self.assertEqual(metadata["langgraph_node"], "model")
        self.assertEqual(metadata["thread_id"], thread.thread_id)
        self.assertEqual(final_events[-1]["assistant_message"], "你好")

    async def test_full_ai_messages_are_forwarded_without_synthetic_chunking(self):
        class FakeAgent:
            async def astream(self, input_data, *, config, stream_mode):
                yield (
                    "messages",
                    (
                        AIMessage(
                            content="streaming fallback response",
                            id="ai-full",
                        ),
                        {
                            "langgraph_node": "model",
                            "thread_id": config["configurable"]["thread_id"],
                        },
                    ),
                )
                yield (
                    "values",
                    {
                        "messages": [
                            HumanMessage(content="current user", id="human-new"),
                            AIMessage(
                                content="streaming fallback response",
                                id="ai-full",
                            ),
                        ]
                    },
                )

        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "streaming fallback response",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": [],
                }
            ),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                record=run,
                ctx=RunContext(thread_store=thread_store),
                agent_factory=lambda context: FakeAgent(),
                graph_input=normalize_graph_input({
                    "messages": [{"type": "human", "content": "current user"}]
                }),
                config={},
                stream_modes=["messages"],
            )

        parsed_events = self.parse_events(await self.drain_events(bridge, run.run_id))
        message_events = [
            data for event, data in parsed_events if event == "messages"
        ]

        self.assertEqual(len(message_events), 1)
        self.assertEqual(message_events[0][0]["type"], "ai")
        self.assertEqual(
            message_events[0][0]["content"],
            "streaming fallback response",
        )


if __name__ == "__main__":
    unittest.main()
