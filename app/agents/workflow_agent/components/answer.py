from __future__ import annotations

from app.agents.workflow_agent.components.base import StructuredWorkflowComponent
from app.agents.workflow_agent.models import (
    AnswerDraft,
    EvidenceResult,
    InquiryState,
    SafetyReview,
    SyndromeAnalysis,
)
from app.agents.workflow_agent.prompts import ANSWER_SYSTEM_PROMPT


class AnswerAgent(StructuredWorkflowComponent[AnswerDraft]):
    schema = AnswerDraft
    system_prompt = ANSWER_SYSTEM_PROMPT

    async def compose(
        self,
        user_text: str,
        inquiry: InquiryState,
        evidence: EvidenceResult,
        syndrome: SyndromeAnalysis,
        safety_review: SafetyReview | None = None,
    ) -> AnswerDraft:
        return await self.invoke_structured(
            {
                "user_question": user_text,
                "inquiry_state": inquiry.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
                "syndrome_analysis": syndrome.model_dump(mode="json"),
                "safety_review": safety_review.model_dump(mode="json")
                if safety_review
                else None,
                "output_contract": {
                    "do_not_add_terms": True,
                    "do_not_add_evidence": True,
                    "do_not_add_diagnosis": True,
                    "do_not_add_prescriptions_or_dosages": True,
                },
            }
        )

    def safe_fallback(self, inquiry: InquiryState) -> AnswerDraft:
        complaint = inquiry.chief_complaint or "当前不适"
        missing = "、".join(inquiry.missing_info[:3])
        if missing:
            detail = f"建议先补充{missing}等信息。"
        else:
            detail = "建议先补充持续时间、诱因和伴随症状等信息。"
        risk_text = ""
        if inquiry.known_facts.risk_flags:
            risk_text = (
                "你提到的"
                + "、".join(inquiry.known_facts.risk_flags)
                + "属于需要重视的风险信号，建议及时线下就医评估。"
            )
        return AnswerDraft(
            draft_answer=(
                f"关于{complaint}，当前信息不足，不能给出诊断或用药建议。"
                f"{detail}{risk_text}"
            )
        )
