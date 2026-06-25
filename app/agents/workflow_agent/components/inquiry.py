from __future__ import annotations

from collections.abc import Sequence

from app.agents.workflow_agent.components.base import StructuredWorkflowComponent
from app.agents.workflow_agent.models import InquiryState
from app.agents.workflow_agent.prompts import INQUIRY_SYSTEM_PROMPT

CLARIFICATION_MARKERS = (
    "\u8865\u5145",
    "\u8bf7\u8865\u5145",
    "\u8bf7\u5148",
    "\u5173\u952e\u4fe1\u606f",
)


def _content_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "content" in value:
            return _content_text(value.get("content"))
        if "text" in value:
            return _content_text(value.get("text"))
        return ""
    if isinstance(value, list | tuple):
        return " ".join(_content_text(item) for item in value if item)
    return str(value).strip()


def conversation_text(conversation: Sequence[object] | None) -> str:
    return "\n".join(
        text for text in (_content_text(item) for item in conversation or []) if text
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _has_known_fact(inquiry: InquiryState) -> bool:
    facts = inquiry.known_facts
    return bool(
        facts.duration.strip()
        or facts.triggers
        or facts.associated_symptoms
        or facts.risk_flags
    )


def _intent_value(intent: object | None, key: str) -> object | None:
    if intent is None:
        return None
    if isinstance(intent, dict):
        return intent.get(key)
    return getattr(intent, key, None)


def _should_continue_despite_pause(
    inquiry: InquiryState,
    *,
    user_text: str,
    visible_conversation: str,
    intent: object | None = None,
) -> bool:
    if not inquiry.should_pause_for_clarification:
        return False
    if not inquiry.chief_complaint:
        return False

    if inquiry.known_facts.risk_flags:
        return True

    if _intent_value(intent, "primary_intent") == "cause_explanation":
        return True

    return _contains_any(
        visible_conversation, CLARIFICATION_MARKERS
    ) and _has_known_fact(inquiry)


def _apply_pause_policy(
    inquiry: InquiryState,
    *,
    user_text: str,
    visible_conversation: str,
    intent: object | None = None,
) -> InquiryState:
    if not _should_continue_despite_pause(
        inquiry,
        user_text=user_text,
        visible_conversation=visible_conversation,
        intent=intent,
    ):
        return inquiry

    return inquiry.model_copy(
        update={
            "information_sufficiency": "sufficient",
            "should_pause_for_clarification": False,
        }
    )


class InquiryAgent(StructuredWorkflowComponent[InquiryState]):
    schema = InquiryState
    system_prompt = INQUIRY_SYSTEM_PROMPT

    async def assess(
        self,
        user_text: str,
        conversation: Sequence[object] | None = None,
        intent: object | None = None,
    ) -> InquiryState:
        visible_conversation = conversation_text(conversation)
        inquiry = await self.invoke_structured(
            {
                "user_question": user_text,
                "visible_conversation": visible_conversation,
                "output_contract": {
                    "max_clarification_questions": 3,
                    "pause_only_when_information_is_severely_insufficient": True,
                    "do_not_answer": True,
                },
            }
        )
        return _apply_pause_policy(
            inquiry,
            user_text=user_text,
            visible_conversation=visible_conversation,
            intent=intent,
        )
