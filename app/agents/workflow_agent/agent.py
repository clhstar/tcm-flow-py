from __future__ import annotations

from typing import Any

from app.agents.workflow_agent.llm import build_workflow_model
from app.agents.workflow_agent.workflow import TCMWorkflow
from app.runtime import state as runtime_state


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

    async def _read_conversation(self, thread_id: str) -> list[dict[str, Any]]:
        values = await self._read_values(thread_id)
        conversation = values.get("conversation") or []
        return [item for item in conversation if isinstance(item, dict)]

    async def aget_state(self, config: dict[str, Any]) -> Any:
        return await self.workflow.graph.aget_state(config)

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        await self.workflow.graph.aupdate_state(config, values)

    async def astream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any],
        stream_mode: Any,
    ):
        thread_id = self._thread_id(config)
        conversation = await self._read_conversation(thread_id)
        user_text = _latest_user_text(input_data)

        async for stream_event, chunk in self.workflow.astream(
            user_text=user_text,
            conversation=conversation,
            config=config,
            stream_mode=stream_mode,
        ):
            yield stream_event, chunk


def make_workflow_agent(context: dict[str, Any] | None = None) -> WorkflowAgent:
    model = build_workflow_model(context)
    return WorkflowAgent(
        workflow=TCMWorkflow(
            model=model,
            checkpointer=runtime_state.state.checkpointer,
        )
    )
