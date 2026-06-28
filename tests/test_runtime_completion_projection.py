import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.runtime.public_messages import (
    append_visible_messages,
    extract_final_assistant_text,
)
from app.runtime.runs.projection import (
    RunCompletionProjection,
    checkpoint_message_count,
)
from app.runtime.runs.stream_adapter import StreamSnapshot
from app.runtime.serialization import serialize_message
from app.store.models import RunRecord


class CheckpointAgent:
    def __init__(self, messages):
        self.messages = list(messages)
        self.get_calls = []
        self.update_calls = []

    async def aget_state(self, config):
        self.get_calls.append(config)
        return SimpleNamespace(values={"messages": list(self.messages)})

    async def aupdate_state(self, config, values):
        replacement = values["messages"][0]
        self.update_calls.append((config, replacement))
        for index, message in enumerate(self.messages):
            message_id = (
                message.get("id")
                if isinstance(message, dict)
                else getattr(message, "id", None)
            )
            if message_id == replacement.id:
                self.messages[index] = replacement
                break


def serialize_messages(messages):
    return [serialize_message(message) for message in messages]


def guardrail_result(final_text, *, rewritten=False):
    return {
        "final_text": final_text,
        "validation": {"passed": True},
        "validation_before_rewrite": {"passed": not rewritten},
        "rewritten": rewritten,
        "allowed_terms": [],
    }


class RuntimeCompletionProjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_checkpoint_message_count_handles_checkpoint_and_stateless_agents(self):
        config = {"configurable": {"thread_id": "thread-1"}}
        agent = CheckpointAgent(
            [
                HumanMessage(content="question", id="human-1"),
                AIMessage(content="answer", id="ai-1"),
            ]
        )

        self.assertEqual(await checkpoint_message_count(agent, config), 2)
        self.assertEqual(await checkpoint_message_count(object(), config), 0)
        self.assertEqual(agent.get_calls, [config])

    async def test_clarification_uses_only_current_run_and_skips_guardrail(self):
        old_messages = [
            AIMessage(
                content="Old clarification:",
                id="ai-old-tool",
                tool_calls=[
                    {
                        "id": "call-old",
                        "name": "ask_clarification",
                        "args": {"questions": ["Old question?"]},
                    }
                ],
            ),
            {
                "type": "tool",
                "content": "Old question?",
                "name": "ask_clarification",
                "tool_call_id": "call-old",
                "id": "clarification:call-old",
            },
        ]
        current_messages = [
            HumanMessage(content="current question", id="human-new"),
            AIMessage(
                content="Please clarify:",
                id="ai-new-tool",
                tool_calls=[
                    {
                        "id": "call-new",
                        "name": "ask_clarification",
                        "args": {"questions": ["How long?"]},
                    }
                ],
            ),
            {
                "type": "tool",
                "content": "Please clarify: How long?",
                "name": "ask_clarification",
                "tool_call_id": "call-new",
                "id": "clarification:call-new",
            },
        ]
        serialized = serialize_messages(old_messages + current_messages)
        existing_conversation = [{"role": "assistant", "content": "Earlier"}]
        projection = RunCompletionProjection(
            record=RunRecord("run-1", "thread-1", "lead_agent"),
            thread_values={"conversation": existing_conversation},
            user_text="current question",
            emit_debug_events=False,
            message_start_index=len(old_messages),
        )
        publish_update = AsyncMock()

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(side_effect=AssertionError("guardrail must be skipped")),
        ):
            result = await projection.complete(
                agent=CheckpointAgent(old_messages + current_messages),
                config={"configurable": {"thread_id": "thread-1"}},
                snapshot=StreamSnapshot(
                    latest_values={
                        "messages": serialized,
                        "agent_trace": [{"agent": "InquiryAgent"}],
                    },
                    latest_messages=serialized,
                ),
                publish_update=publish_update,
            )

        expected_assistant = "Please clarify:\n1. How long?"
        self.assertEqual(result.run_status, "waiting_clarification")
        self.assertEqual(result.thread_status, "waiting")
        self.assertEqual(
            result.final_payload,
            {
                "thread_id": "thread-1",
                "run_id": "run-1",
                "status": "need_clarification",
                "assistant_message": expected_assistant,
                "pending_clarification": ["How long?"],
                "references": [],
            },
        )
        self.assertEqual(
            result.thread_values["conversation"],
            [
                {"role": "assistant", "content": "Earlier"},
                {"role": "user", "content": "current question"},
                {
                    "role": "assistant",
                    "content": expected_assistant,
                    "run_id": "run-1",
                    "agent_trace": [{"agent": "InquiryAgent"}],
                },
            ],
        )
        publish_update.assert_not_awaited()

    async def test_old_clarification_with_current_final_completes_normally(self):
        old_messages = [
            AIMessage(
                content="Old clarification:",
                id="ai-old-tool",
                tool_calls=[
                    {
                        "id": "call-old",
                        "name": "ask_clarification",
                        "args": {"questions": ["Old question?"]},
                    }
                ],
            ),
            {
                "type": "tool",
                "content": "Old question?",
                "name": "ask_clarification",
                "tool_call_id": "call-old",
            },
        ]
        current_messages = [
            HumanMessage(content="new details", id="human-new"),
            AIMessage(content="Current answer", id="ai-new"),
        ]
        full_messages = serialize_messages(old_messages + current_messages)
        projection = self._projection(message_start_index=len(old_messages))
        guardrail = AsyncMock(return_value=guardrail_result("Current answer"))

        with patch("app.runtime.runs.projection.apply_guardrails", new=guardrail):
            result = await projection.complete(
                agent=CheckpointAgent(old_messages + current_messages),
                config={},
                snapshot=StreamSnapshot(
                    latest_values={"messages": full_messages},
                    latest_messages=full_messages,
                ),
                publish_update=AsyncMock(),
            )

        self.assertEqual(result.final_payload["status"], "completed")
        self.assertEqual(result.final_payload["assistant_message"], "Current answer")
        self.assertIsNone(result.final_payload["pending_clarification"])
        guardrail.assert_awaited_once()

    async def test_completion_extracts_current_answer_but_guardrails_full_history(self):
        messages = [
            HumanMessage(content="old", id="human-old"),
            AIMessage(content="Old answer", id="ai-old"),
            HumanMessage(content="current", id="human-current"),
            AIMessage(content="Current answer", id="ai-current"),
        ]
        serialized = serialize_messages(messages)
        projection = self._projection(message_start_index=2)
        guardrail = AsyncMock(return_value=guardrail_result("Current answer"))

        with patch("app.runtime.runs.projection.apply_guardrails", new=guardrail):
            result = await projection.complete(
                agent=CheckpointAgent(messages),
                config={},
                snapshot=StreamSnapshot(
                    latest_values={"messages": serialized},
                    latest_messages=serialized,
                ),
                publish_update=AsyncMock(),
            )

        guardrail.assert_awaited_once_with(
            final_text="Current answer",
            messages=serialized,
        )
        self.assertEqual(result.final_payload["assistant_message"], "Current answer")

    async def test_guardrail_rewrite_updates_same_checkpoint_message_and_rereads_it(self):
        messages = [
            HumanMessage(content="question", id="human-1"),
            AIMessage(content="unsafe original", id="ai-final"),
        ]
        serialized = serialize_messages(messages)
        agent = CheckpointAgent(messages)
        projection = self._projection(emit_debug_events=True)
        trace = [{"agent": "SafetyAgent"}]

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value=guardrail_result("safe rewrite", rewritten=True)
            ),
        ):
            result = await projection.complete(
                agent=agent,
                config={"configurable": {"thread_id": "thread-1"}},
                snapshot=StreamSnapshot(
                    latest_values={"messages": serialized, "agent_trace": trace},
                    latest_messages=serialized,
                ),
                publish_update=AsyncMock(),
            )

        self.assertEqual(len(agent.update_calls), 1)
        self.assertEqual(agent.update_calls[0][1].id, "ai-final")
        self.assertEqual(agent.update_calls[0][1].content, "safe rewrite")
        self.assertEqual(len(agent.get_calls), 1)
        self.assertEqual(result.thread_values["messages"][-1]["id"], "ai-final")
        self.assertEqual(
            result.thread_values["messages"][-1]["content"], "safe rewrite"
        )
        self.assertNotIn(
            "unsafe original",
            [message.get("content") for message in result.thread_values["messages"]],
        )
        self.assertEqual(
            result.thread_values["conversation"][-1],
            {
                "role": "assistant",
                "content": "safe rewrite",
                "run_id": "run-1",
                "agent_trace": trace,
            },
        )

    async def test_guardrail_rewrite_requires_final_ai_message_id(self):
        messages = [
            serialize_message(HumanMessage(content="question", id="human-1")),
            {"type": "ai", "content": "unsafe original"},
        ]
        agent = CheckpointAgent(messages)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value=guardrail_result("safe rewrite", rewritten=True)
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "AIMessage.*id"):
                await self._projection().complete(
                    agent=agent,
                    config={},
                    snapshot=StreamSnapshot(
                        latest_values={"messages": messages},
                        latest_messages=messages,
                    ),
                    publish_update=AsyncMock(),
                )

        self.assertEqual(agent.update_calls, [])

    async def test_debug_observer_scopes_deduplicates_and_defensively_copies(self):
        old_call = {
            "type": "ai",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-old",
                    "name": "old_tool",
                    "args": {"query": "old"},
                }
            ],
        }
        current_call = {
            "type": "ai",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-current",
                    "name": "current_tool",
                    "args": {"query": "current"},
                }
            ],
        }
        thread_values = {"conversation": [{"role": "assistant", "content": "old"}]}
        trace = [{"agent": "ResearchAgent", "action": "search"}]
        values = {
            "messages": [old_call, serialize_message(HumanMessage(content="new")), current_call],
            "agent_trace": trace,
        }
        projection = RunCompletionProjection(
            record=RunRecord("run-1", "thread-1", "lead_agent"),
            thread_values=thread_values,
            user_text="new",
            emit_debug_events=True,
            message_start_index=1,
        )

        first = await projection.observe_values(values)
        second = await projection.observe_values(values)
        trace[0]["agent"] = "mutated"
        thread_values["conversation"].append(
            {"role": "assistant", "content": "mutated"}
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0][0], "updates")
        self.assertEqual(first[0][1]["tool"], "current_tool")
        self.assertNotIn("old_tool", repr(first))
        self.assertEqual(second, [])
        self.assertEqual(
            projection.current_agent_trace,
            [{"agent": "ResearchAgent", "action": "search"}],
        )
        self.assertEqual(
            projection.thread_values,
            {"conversation": [{"role": "assistant", "content": "old"}]},
        )

    async def test_debug_guardrail_updates_are_ordered_and_preserve_payload(self):
        messages = [AIMessage(content="answer", id="ai-final")]
        serialized = serialize_messages(messages)
        publish_update = AsyncMock()
        guardrail = AsyncMock(return_value=guardrail_result("answer", rewritten=True))

        with patch("app.runtime.runs.projection.apply_guardrails", new=guardrail):
            await self._projection(emit_debug_events=True).complete(
                agent=CheckpointAgent(messages),
                config={},
                snapshot=StreamSnapshot(
                    latest_values={"messages": serialized},
                    latest_messages=serialized,
                ),
                publish_update=publish_update,
            )

        self.assertEqual(
            publish_update.await_args_list,
            [
                call(
                    {
                        "type": "guardrail",
                        "status": "started",
                        "agent": "guardrail_middleware",
                        "summary": "正在进行术语一致性校验与答案安全检查。",
                    }
                ),
                call(
                    {
                        "type": "guardrail",
                        "status": "completed",
                        "agent": "guardrail_middleware",
                        "summary": "术语一致性校验完成。",
                        "validation": {"passed": True},
                        "rewritten": True,
                    }
                ),
            ],
        )

    async def test_disabled_debug_path_emits_no_guardrail_updates(self):
        messages = [AIMessage(content="answer", id="ai-final")]
        serialized = serialize_messages(messages)
        publish_update = AsyncMock()

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(return_value=guardrail_result("answer")),
        ):
            await self._projection(emit_debug_events=False).complete(
                agent=CheckpointAgent(messages),
                config={},
                snapshot=StreamSnapshot(
                    latest_values={"messages": serialized},
                    latest_messages=serialized,
                ),
                publish_update=publish_update,
            )

        publish_update.assert_not_awaited()

    async def test_final_payload_excludes_internal_state(self):
        messages = [AIMessage(content="answer", id="ai-final")]
        serialized = serialize_messages(messages)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(return_value=guardrail_result("answer")),
        ):
            result = await self._projection().complete(
                agent=CheckpointAgent(messages),
                config={},
                snapshot=StreamSnapshot(
                    latest_values={
                        "messages": serialized,
                        "validation": {"passed": True},
                        "agent_trace": [{"agent": "lead_agent"}],
                    },
                    latest_messages=serialized,
                ),
                publish_update=AsyncMock(),
            )

        self.assertEqual(
            set(result.final_payload),
            {
                "thread_id",
                "run_id",
                "status",
                "assistant_message",
                "pending_clarification",
                "references",
            },
        )
        self.assertTrue(
            {"messages", "validation", "agent_trace"}.isdisjoint(result.final_payload)
        )

    def _projection(
        self,
        *,
        emit_debug_events=False,
        message_start_index=0,
    ):
        return RunCompletionProjection(
            record=RunRecord("run-1", "thread-1", "lead_agent"),
            thread_values={"conversation": []},
            user_text="question",
            emit_debug_events=emit_debug_events,
            message_start_index=message_start_index,
        )


class PublicMessageProjectionTests(unittest.TestCase):
    def test_append_visible_messages_preserves_inputs_and_copies_trace(self):
        conversation = [{"role": "assistant", "content": "existing"}]
        thread_values = {"conversation": conversation, "other": "value"}
        trace = [{"agent": "SafetyAgent", "action": "validate"}]
        original_thread_values = deepcopy(thread_values)
        original_trace = deepcopy(trace)

        result = append_visible_messages(
            thread_values,
            "question",
            "answer",
            run_id="run-1",
            agent_trace=trace,
        )

        self.assertEqual(thread_values, original_thread_values)
        self.assertEqual(conversation, original_thread_values["conversation"])
        self.assertEqual(trace, original_trace)
        self.assertEqual(
            result,
            [
                {"role": "assistant", "content": "existing"},
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "run_id": "run-1",
                    "agent_trace": original_trace,
                },
            ],
        )
        self.assertIsNot(result[-1]["agent_trace"], trace)
        self.assertIsNot(result[-1]["agent_trace"][0], trace[0])

    def test_extract_final_assistant_text_handles_list_content_and_skips_tool_calls(self):
        messages = [
            {"type": "ai", "content": "older answer"},
            {
                "type": "ai",
                "content": "hidden tool preamble",
                "tool_calls": [{"name": "search", "args": {}}],
            },
            {"type": "ai", "content": "   "},
            {
                "type": "ai",
                "content": [
                    {"type": "text", "text": "visible "},
                    {"type": "text", "text": "answer"},
                ],
            },
        ]

        self.assertEqual(extract_final_assistant_text(messages), "visible answer")


if __name__ == "__main__":
    unittest.main()
