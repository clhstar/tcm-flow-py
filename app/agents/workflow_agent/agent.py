from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage

from app.agents.workflow_agent.llm import build_workflow_model
from app.agents.workflow_agent.workflow import TCMWorkflow
from app.runtime import state as runtime_state
from app.runtime.serialization import serialize_message


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return _extract_text(content.get("text"))
        if "content" in content:
            return _extract_text(content.get("content"))
        return ""
    if isinstance(content, list):
        return "\n".join(
            text for text in (_extract_text(block).strip() for block in content) if text
        )
    return str(content)


def _latest_user_text(input_data: dict[str, Any]) -> str:
    for message in reversed(input_data.get("messages", [])):
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("type")
        if role in {"user", "human"}:
            return _extract_text(message.get("content", "")).strip()
    return ""


class WorkflowAgent:
    def __init__(
        self,
        *,
        workflow: TCMWorkflow | None = None,
        thread_store: Any | None = None,
    ) -> None:
        if workflow is None:
            raise ValueError(
                "WorkflowAgent requires an explicit TCMWorkflow. "
                "Use make_workflow_agent() for runtime construction."
            )
        self.workflow = workflow
        self.thread_store = thread_store or runtime_state.state.thread_store

    def _thread_id(self, config: dict[str, Any]) -> str:
        return str(config["configurable"]["thread_id"])

    async def _read_values(self, thread_id: str) -> dict[str, Any]:
        thread = await self.thread_store.get(thread_id)
        return dict(thread.values) if thread else {}

    async def _read_messages(self, thread_id: str) -> list[dict[str, Any]]:
        values = await self._read_values(thread_id)
        messages = values.get("messages") or []
        return [message for message in messages if isinstance(message, dict)]

    async def _read_conversation(self, thread_id: str) -> list[dict[str, Any]]:
        values = await self._read_values(thread_id)
        conversation = values.get("conversation") or []
        return [item for item in conversation if isinstance(item, dict)]

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        graph = getattr(self.workflow, "graph", None)
        if graph is not None and hasattr(graph, "aget_state"):
            try:
                return await graph.aget_state(config)
            except ValueError as exc:
                if "No checkpointer set" not in str(exc):
                    raise

        return SimpleNamespace(
            values={"messages": await self._read_messages(self._thread_id(config))},
            next=(),
        )

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        graph = getattr(self.workflow, "graph", None)
        if graph is not None and hasattr(graph, "aupdate_state"):
            try:
                await graph.aupdate_state(config, values)
                return
            except ValueError as exc:
                if "No checkpointer set" not in str(exc):
                    raise

        thread_id = self._thread_id(config)
        messages = list(await self._read_messages(thread_id))

        for incoming in values.get("messages", []):
            serialized = serialize_message(incoming)
            message_id = serialized.get("id")
            if not message_id:
                messages.append(serialized)
                continue

            for index in range(len(messages) - 1, -1, -1):
                if messages[index].get("id") == message_id:
                    messages[index] = serialized
                    break
            else:
                messages.append(serialized)

        await self.thread_store.update_values(thread_id, {"messages": messages})

    async def astream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        stream_mode: Any,
    ):
        thread_id = self._thread_id(config)
        previous_messages = await self._read_messages(thread_id)
        conversation = await self._read_conversation(thread_id)
        user_text = _latest_user_text(input_data)

        workflow_result = await self.workflow.run(
            user_text=user_text,
            conversation=conversation,
            config=config,
        )

        current_messages = [
            HumanMessage(content=user_text),
            *workflow_result.messages,
        ]
        serialized_messages = [
            *previous_messages,
            *(serialize_message(message) for message in current_messages),
        ]

        await self.thread_store.update_values(
            thread_id,
            {
                "messages": serialized_messages,
                "last_agent_trace": workflow_result.agent_trace,
            },
        )

        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode or [])
        if "messages" in modes and workflow_result.final_text:
            yield (
                "messages",
                (
                    AIMessageChunk(content=workflow_result.final_text),
                    {"thread_id": thread_id, "agent": "workflow_agent"},
                ),
            )

        yield (
            "values",
            {
                "messages": serialized_messages,
                "workflow_trace": workflow_result.agent_trace,
            },
        )


def make_workflow_agent(context: dict[str, Any] | None = None) -> WorkflowAgent:
    model = build_workflow_model(context)
    return WorkflowAgent(
        workflow=TCMWorkflow(
            model=model,
            checkpointer=runtime_state.state.checkpointer,
        )
    )
