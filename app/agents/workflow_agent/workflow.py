from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage

from app.agents.workflow_agent.components.intent import IntentAgent
from app.agents.workflow_agent.components.answer import AnswerAgent
from app.agents.workflow_agent.components.evidence import EvidenceAgent
from app.agents.workflow_agent.components.inquiry import InquiryAgent
from app.agents.workflow_agent.components.safety import SafetyAgent
from app.agents.workflow_agent.components.syndrome import SyndromeAgent
from app.agents.workflow_agent.graph import build_workflow_graph


@dataclass
class WorkflowRunResult:
    messages: list[BaseMessage]
    final_text: str
    needs_clarification: bool
    agent_trace: list[dict]


class TCMWorkflow:
    def __init__(
        self,
        *,
        checkpointer: Any,
        model: Any | None = None,
        intent_agent: IntentAgent | None = None,
        inquiry_agent: InquiryAgent | None = None,
        evidence_agent: EvidenceAgent | None = None,
        syndrome_agent: SyndromeAgent | None = None,
        answer_agent: AnswerAgent | None = None,
        safety_agent: SafetyAgent | None = None,
    ) -> None:
        if checkpointer is None:
            raise ValueError("TCMWorkflow requires a LangGraph checkpointer.")
        if model is None and (
            intent_agent is None
            or inquiry_agent is None
            or syndrome_agent is None
            or answer_agent is None
            or safety_agent is None
        ):
            raise ValueError(
                "TCMWorkflow requires a ChatOpenAI-compatible model when LLM agents "
                "are not supplied explicitly."
            )
        self.intent_agent = intent_agent or IntentAgent(model)
        self.inquiry_agent = inquiry_agent or InquiryAgent(model)
        self.evidence_agent = evidence_agent or EvidenceAgent()
        self.syndrome_agent = syndrome_agent or SyndromeAgent(model)
        self.answer_agent = answer_agent or AnswerAgent(model)
        self.safety_agent = safety_agent or SafetyAgent(model)
        self.checkpointer = checkpointer
        self.graph = build_workflow_graph(
            intent_agent=self.intent_agent,
            inquiry_agent=self.inquiry_agent,
            evidence_agent=self.evidence_agent,
            syndrome_agent=self.syndrome_agent,
            answer_agent=self.answer_agent,
            safety_agent=self.safety_agent,
            checkpointer=checkpointer,
        )

    async def _checkpoint_offsets(
        self,
        config: dict[str, Any],
    ) -> tuple[int, int]:
        snapshot = await self.graph.aget_state(config)
        values = dict(getattr(snapshot, "values", {}) or {})
        return (
            len(values.get("messages", []) or []),
            len(values.get("agent_trace", []) or []),
        )

    def _run_id(self, config: dict[str, Any]) -> str:
        configurable = config.get("configurable") or {}
        return str(configurable.get("run_id") or uuid4().hex)

    def _initial_state(
        self,
        *,
        user_text: str,
        conversation: Sequence[object] | None = None,
        run_id: str,
        human_message_id: str,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "user_text": user_text,
            "conversation": [
                item for item in conversation or [] if isinstance(item, dict)
            ],
            "messages": [
                HumanMessage(content=user_text, id=human_message_id),
            ],
            "intent": {},
            "inquiry": {},
            "evidence": {},
            "syndrome": {},
            "answer": {},
            "safety": {},
            "needs_clarification": False,
            "final_text": "",
            "agent_trace": [],
        }

    def _run_result_from_state(
        self,
        result: dict[str, Any],
        *,
        message_start: int,
        trace_start: int,
        human_message_id: str,
    ) -> WorkflowRunResult:
        current_messages = list(result.get("messages", []))[message_start:]
        workflow_messages = [
            message
            for message in current_messages
            if getattr(message, "id", None) != human_message_id
        ]
        current_trace = list(result.get("agent_trace", []))[trace_start:]
        return WorkflowRunResult(
            messages=workflow_messages,
            final_text=str(result.get("final_text", "")),
            needs_clarification=bool(result.get("needs_clarification")),
            agent_trace=current_trace,
        )

    def _normalize_stream_mode(self, stream_mode: Any) -> list[str]:
        if stream_mode is None:
            return ["messages", "values"]
        if isinstance(stream_mode, str):
            return [stream_mode]
        return [str(mode) for mode in stream_mode]

    def _split_stream_chunk(self, chunk: Any) -> tuple[str, Any]:
        if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
            return chunk
        return "values", chunk

    async def astream(
        self,
        *,
        user_text: str,
        config: dict[str, Any],
        conversation: Sequence[object] | None = None,
        stream_mode: Any = None,
    ):
        _, trace_start = await self._checkpoint_offsets(config)
        run_id = self._run_id(config)
        human_message_id = f"workflow-{run_id}-human-1"
        input_state = self._initial_state(
            user_text=user_text,
            conversation=conversation,
            run_id=run_id,
            human_message_id=human_message_id,
        )

        async for chunk in self.graph.astream(
            input_state,
            config=config,
            stream_mode=self._normalize_stream_mode(stream_mode),
        ):
            stream_event, payload = self._split_stream_chunk(chunk)
            if stream_event == "values" and isinstance(payload, dict):
                payload = dict(payload)
                payload["agent_trace"] = list(payload.get("agent_trace", []))[
                    trace_start:
                ]
            yield stream_event, payload

    async def run(
        self,
        user_text: str,
        config: dict[str, Any],
        conversation: Sequence[object] | None = None,
    ) -> WorkflowRunResult:
        message_start, trace_start = await self._checkpoint_offsets(config)
        run_id = self._run_id(config)
        human_message_id = f"workflow-{run_id}-human-1"
        input_state = self._initial_state(
            user_text=user_text,
            conversation=conversation,
            run_id=run_id,
            human_message_id=human_message_id,
        )
        result: dict[str, Any] = {}

        async for chunk in self.graph.astream(
            input_state,
            config=config,
            stream_mode=["values"],
        ):
            stream_event, payload = self._split_stream_chunk(chunk)
            if stream_event == "values" and isinstance(payload, dict):
                result = payload

        return self._run_result_from_state(
            result,
            message_start=message_start,
            trace_start=trace_start,
            human_message_id=human_message_id,
        )


__all__ = [
    "AnswerAgent",
    "EvidenceAgent",
    "InquiryAgent",
    "IntentAgent",
    "SafetyAgent",
    "SyndromeAgent",
    "TCMWorkflow",
    "WorkflowRunResult",
]
