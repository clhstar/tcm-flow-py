from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage

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
        model: Any | None = None,
        inquiry_agent: InquiryAgent | None = None,
        evidence_agent: EvidenceAgent | None = None,
        syndrome_agent: SyndromeAgent | None = None,
        answer_agent: AnswerAgent | None = None,
        safety_agent: SafetyAgent | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        if (
            model is None
            and (
                inquiry_agent is None
                or syndrome_agent is None
                or answer_agent is None
                or safety_agent is None
            )
        ):
            raise ValueError(
                "TCMWorkflow requires a ChatOpenAI-compatible model when LLM agents "
                "are not supplied explicitly."
            )

        self.inquiry_agent = inquiry_agent or InquiryAgent(model)
        self.evidence_agent = evidence_agent or EvidenceAgent()
        self.syndrome_agent = syndrome_agent or SyndromeAgent(model)
        self.answer_agent = answer_agent or AnswerAgent(model)
        self.safety_agent = safety_agent or SafetyAgent(model)
        self.checkpointer = checkpointer
        self.graph = build_workflow_graph(
            inquiry_agent=self.inquiry_agent,
            evidence_agent=self.evidence_agent,
            syndrome_agent=self.syndrome_agent,
            answer_agent=self.answer_agent,
            safety_agent=self.safety_agent,
            checkpointer=checkpointer,
        )

    async def _checkpoint_offsets(
        self,
        config: dict[str, Any] | None,
    ) -> tuple[int, int]:
        if self.checkpointer is None or config is None:
            return 0, 0

        snapshot = await self.graph.aget_state(config)
        values = dict(getattr(snapshot, "values", {}) or {})
        return (
            len(values.get("messages", []) or []),
            len(values.get("agent_trace", []) or []),
        )

    def _run_id(self, config: dict[str, Any] | None) -> str:
        configurable = (config or {}).get("configurable") or {}
        return str(configurable.get("run_id") or uuid4().hex)

    async def run(
        self,
        user_text: str,
        conversation: Sequence[object] | None = None,
        config: dict[str, Any] | None = None,
    ) -> WorkflowRunResult:
        message_start, trace_start = await self._checkpoint_offsets(config)
        run_id = self._run_id(config)
        human_message_id = f"workflow-{run_id}-human-1"

        result = await self.graph.ainvoke(
            {
                "run_id": run_id,
                "user_text": user_text,
                "conversation": [
                    item for item in conversation or [] if isinstance(item, dict)
                ],
                "messages": [
                    HumanMessage(content=user_text, id=human_message_id),
                ],
                "agent_trace": [],
            },
            config=config,
        )

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


__all__ = [
    "AnswerAgent",
    "EvidenceAgent",
    "InquiryAgent",
    "SafetyAgent",
    "SyndromeAgent",
    "TCMWorkflow",
    "WorkflowRunResult",
]
