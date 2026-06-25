from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agents.workflow_agent.components.intent import IntentAgent
from app.agents.workflow_agent.components.answer import AnswerAgent
from app.agents.workflow_agent.components.evidence import EvidenceAgent
from app.agents.workflow_agent.components.inquiry import InquiryAgent
from app.agents.workflow_agent.components.safety import SafetyAgent
from app.agents.workflow_agent.components.syndrome import SyndromeAgent
from app.agents.workflow_agent.models import (
    AnswerDraft,
    EvidenceResult,
    InquiryState,
    IntentState,
    SafetyReview,
    SyndromeAnalysis,
)
from app.agents.workflow_agent.state import WorkflowState
from app.middlewares.clarification_controller import format_clarification_questions


def _as_model(value: Any, schema: type[Any]) -> Any:
    if isinstance(value, schema):
        return value
    return schema.model_validate(value)


def _state_model(
    state: WorkflowState,
    key: str,
    schema: type[Any],
) -> Any:
    value = state.get(key)
    if value is None:
        return schema()
    return _as_model(value, schema)


def _intent(state: WorkflowState) -> IntentState:
    return _state_model(state, "intent", IntentState)


def _inquiry(state: WorkflowState) -> InquiryState:
    return _state_model(state, "inquiry", InquiryState)


def _evidence(state: WorkflowState) -> EvidenceResult:
    return _as_model(state["evidence"], EvidenceResult)


def _syndrome(state: WorkflowState) -> SyndromeAnalysis:
    return _state_model(state, "syndrome", SyndromeAnalysis)


def _answer(state: WorkflowState) -> AnswerDraft:
    return _as_model(state["answer"], AnswerDraft)


def _safety(state: WorkflowState) -> SafetyReview:
    return _as_model(state["safety"], SafetyReview)


def _state_value(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json")


def route_after_inquiry(state: WorkflowState) -> Literal["clarification", "evidence"]:
    inquiry = _inquiry(state)
    return "clarification" if inquiry.should_pause_for_clarification else "evidence"


def route_after_intent(
    state: WorkflowState,
) -> Literal["direct_response", "inquiry", "evidence"]:
    intent = _intent(state)

    if intent.route_hint == "direct_response":
        return "direct_response"

    if intent.route_hint == "evidence":
        return "evidence"

    return "inquiry"


def route_after_evidence(
    state: WorkflowState,
) -> Literal["syndrome", "answer_draft"]:
    intent = _intent(state)

    if intent.is_personal_health_query:
        return "syndrome"

    return "answer_draft"


def route_after_initial_safety(
    state: WorkflowState,
) -> Literal["answer_rewrite", "finalize"]:
    safety = _safety(state)
    return "answer_rewrite" if safety.rewrite_required else "finalize"


def route_after_rewrite_safety(
    state: WorkflowState,
) -> Literal["safe_fallback", "finalize"]:
    safety = _safety(state)
    return "safe_fallback" if safety.rewrite_required else "finalize"


def _message_id(state: WorkflowState, suffix: str) -> str:
    run_id = str(state.get("run_id") or "workflow")
    return f"workflow-{run_id}-{suffix}"


def build_workflow_graph(
    *,
    intent_agent: IntentAgent,
    inquiry_agent: InquiryAgent,
    evidence_agent: EvidenceAgent,
    syndrome_agent: SyndromeAgent,
    answer_agent: AnswerAgent,
    safety_agent: SafetyAgent,
    checkpointer: Any | None = None,
):
    async def intent_node(state: WorkflowState) -> dict[str, Any]:
        intent = await intent_agent.assess(
            user_text=state["user_text"],
            conversation=state.get("conversation", []),
        )

        return {
            "intent": _state_value(intent),
            "agent_trace": [
                {
                    "agent": "IntentAgent",
                    "model_call": True,
                    "primary_intent": intent.primary_intent,
                    "secondary_intents": intent.secondary_intents,
                    "confidence": intent.confidence,
                    "route_hint": intent.route_hint,
                    "requires_retrieval": intent.requires_retrieval,
                    "should_enter_inquiry": intent.should_enter_inquiry,
                    "has_risk_signal": intent.has_risk_signal,
                    "rule_signals": intent.rule_signals,
                }
            ],
        }

    async def direct_response_node(state: WorkflowState) -> dict[str, Any]:
        intent = _intent(state)

        if intent.direct_response:
            final_text = intent.direct_response
        elif intent.primary_intent == "off_topic":
            final_text = "这个问题可能不属于中医健康咨询范围。你可以描述中医知识、古籍条文、方剂知识或具体不适，我再帮你分析。"
        else:
            final_text = "你好，我可以帮你做中医知识解释、症状信息整理和基于证据的健康咨询。你可以直接描述想了解的问题。"

        return {
            "needs_clarification": False,
            "final_text": final_text,
            "messages": [
                AIMessage(
                    content=final_text,
                    id=_message_id(state, "intent-direct-ai-1"),
                )
            ],
            "agent_trace": [
                {
                    "agent": "IntentAgent",
                    "stage": "direct_response",
                    "primary_intent": intent.primary_intent,
                }
            ],
        }

    async def inquiry_node(state: WorkflowState) -> dict[str, Any]:
        inquiry = await inquiry_agent.assess(
            user_text=state["user_text"],
            conversation=state.get("conversation", []),
            intent=state.get("intent"),
        )
        return {
            "inquiry": _state_value(inquiry),
            "agent_trace": [
                {
                    "agent": "InquiryAgent",
                    "model_call": True,
                    "information_sufficiency": inquiry.information_sufficiency,
                    "should_pause_for_clarification": inquiry.should_pause_for_clarification,
                }
            ],
        }

    async def clarification_node(state: WorkflowState) -> dict[str, Any]:
        inquiry = _inquiry(state)
        tool_call_id = _message_id(state, "clarification-1")
        clarification_text = format_clarification_questions(
            inquiry.clarification_questions
        )
        return {
            "needs_clarification": True,
            "final_text": clarification_text,
            "messages": [
                AIMessage(
                    content="",
                    id=_message_id(state, "clarification-ai-1"),
                    tool_calls=[
                        {
                            "id": tool_call_id,
                            "name": "ask_clarification",
                            "args": {"questions": inquiry.clarification_questions},
                        }
                    ],
                ),
                ToolMessage(
                    id=f"clarification:{tool_call_id}",
                    name="ask_clarification",
                    tool_call_id=tool_call_id,
                    content=clarification_text,
                ),
            ],
        }

    async def evidence_node(state: WorkflowState) -> dict[str, Any]:
        inquiry = _inquiry(state)
        evidence = await evidence_agent.retrieve(
            user_text=state["user_text"],
            inquiry=inquiry,
        )
        tool_call_id = _message_id(state, "retrieval-1")
        return {
            "evidence": _state_value(evidence),
            "messages": [
                AIMessage(
                    content="",
                    id=_message_id(state, "retrieval-ai-1"),
                    tool_calls=[
                        {
                            "id": tool_call_id,
                            "name": "retrieve_tcm_knowledge",
                            "args": {"query": state["user_text"], "mode": "hybrid"},
                        }
                    ],
                ),
                ToolMessage(
                    id=f"tool:{tool_call_id}",
                    name="retrieve_tcm_knowledge",
                    tool_call_id=tool_call_id,
                    content=evidence.raw_tool_content,
                ),
            ],
            "agent_trace": [
                {
                    "agent": "EvidenceAgent",
                    "retrieval_status": evidence.retrieval_status,
                    "retrieval_mode": evidence.retrieval_mode,
                    "evidence_count": len(evidence.evidence),
                }
            ],
        }

    async def syndrome_node(state: WorkflowState) -> dict[str, Any]:
        inquiry = _inquiry(state)
        evidence = _evidence(state)
        syndrome = await syndrome_agent.analyze(
            user_text=state["user_text"],
            inquiry=inquiry,
            evidence=evidence,
        )
        return {
            "syndrome": _state_value(syndrome),
            "agent_trace": [
                {
                    "agent": "SyndromeAgent",
                    "model_call": True,
                    "possible_patterns": [
                        pattern.term for pattern in syndrome.possible_patterns
                    ],
                }
            ],
        }

    async def answer_draft_node(state: WorkflowState) -> dict[str, Any]:
        inquiry = _inquiry(state)
        evidence = _evidence(state)
        syndrome = _syndrome(state)
        answer = await answer_agent.compose(
            user_text=state["user_text"],
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome,
        )
        return {
            "answer": _state_value(answer),
            "agent_trace": [
                {"agent": "AnswerAgent", "stage": "draft", "model_call": True}
            ],
        }

    async def safety_initial_node(state: WorkflowState) -> dict[str, Any]:
        answer = _answer(state)
        inquiry = _inquiry(state)
        evidence = _evidence(state)
        syndrome = _syndrome(state)
        safety = await safety_agent.review(
            draft_answer=answer.draft_answer,
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome,
        )
        return {
            "safety": _state_value(safety),
            "agent_trace": [
                {
                    "agent": "SafetyAgent",
                    "stage": "initial",
                    "model_call": True,
                    "final_safety_level": safety.final_safety_level,
                    "rewrite_required": safety.rewrite_required,
                }
            ],
        }

    async def answer_rewrite_node(state: WorkflowState) -> dict[str, Any]:
        inquiry = _inquiry(state)
        evidence = _evidence(state)
        syndrome = _syndrome(state)
        safety = _safety(state)
        answer = await answer_agent.compose(
            user_text=state["user_text"],
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome,
            safety_review=safety,
        )
        return {
            "answer": _state_value(answer),
            "agent_trace": [
                {"agent": "AnswerAgent", "stage": "rewrite", "model_call": True}
            ],
        }

    async def safety_rewrite_node(state: WorkflowState) -> dict[str, Any]:
        answer = _answer(state)
        inquiry = _inquiry(state)
        evidence = _evidence(state)
        syndrome = _syndrome(state)
        safety = await safety_agent.review(
            draft_answer=answer.draft_answer,
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome,
        )
        return {
            "safety": _state_value(safety),
            "agent_trace": [
                {
                    "agent": "SafetyAgent",
                    "stage": "rewrite",
                    "model_call": True,
                    "final_safety_level": safety.final_safety_level,
                    "rewrite_required": safety.rewrite_required,
                }
            ],
        }

    async def safe_fallback_node(state: WorkflowState) -> dict[str, Any]:
        inquiry = _inquiry(state)
        evidence = _evidence(state)
        syndrome = _syndrome(state)
        answer = answer_agent.safe_fallback(inquiry)
        safety = await safety_agent.review(
            draft_answer=answer.draft_answer,
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome,
        )
        return {
            "answer": _state_value(answer),
            "safety": _state_value(safety),
            "agent_trace": [
                {"agent": "AnswerAgent", "stage": "safe_fallback"},
                {
                    "agent": "SafetyAgent",
                    "stage": "safe_fallback",
                    "model_call": True,
                    "final_safety_level": safety.final_safety_level,
                    "rewrite_required": safety.rewrite_required,
                },
            ],
        }

    async def finalize_node(state: WorkflowState) -> dict[str, Any]:
        answer = _answer(state)
        return {
            "needs_clarification": False,
            "final_text": answer.draft_answer,
            "messages": [
                AIMessage(
                    content=answer.draft_answer,
                    id=_message_id(state, "final-ai-1"),
                )
            ],
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("intent", intent_node)
    graph.add_node("direct_response", direct_response_node)
    graph.add_node("inquiry", inquiry_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("syndrome", syndrome_node)
    graph.add_node("answer_draft", answer_draft_node)
    graph.add_node("safety_initial", safety_initial_node)
    graph.add_node("answer_rewrite", answer_rewrite_node)
    graph.add_node("safety_rewrite", safety_rewrite_node)
    graph.add_node("safe_fallback", safe_fallback_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "intent")
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "direct_response": "direct_response",
            "inquiry": "inquiry",
            "evidence": "evidence",
        },
    )

    graph.add_edge("direct_response", END)

    graph.add_conditional_edges(
        "inquiry",
        route_after_inquiry,
        {"clarification": "clarification", "evidence": "evidence"},
    )
    graph.add_edge("clarification", END)

    graph.add_conditional_edges(
        "evidence",
        route_after_evidence,
        {
            "syndrome": "syndrome",
            "answer_draft": "answer_draft",
        },
    )

    graph.add_edge("syndrome", "answer_draft")
    graph.add_edge("answer_draft", "safety_initial")
    graph.add_conditional_edges(
        "safety_initial",
        route_after_initial_safety,
        {"answer_rewrite": "answer_rewrite", "finalize": "finalize"},
    )
    graph.add_edge("answer_rewrite", "safety_rewrite")
    graph.add_conditional_edges(
        "safety_rewrite",
        route_after_rewrite_safety,
        {"safe_fallback": "safe_fallback", "finalize": "finalize"},
    )
    graph.add_edge("safe_fallback", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
