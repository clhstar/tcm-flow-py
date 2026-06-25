from __future__ import annotations

from app.agents.workflow_agent.components.base import StructuredWorkflowComponent
from app.agents.workflow_agent.models import (
    EvidenceResult,
    InquiryState,
    SafetyReview,
    SyndromeAnalysis,
)
from app.agents.workflow_agent.prompts import SAFETY_SYSTEM_PROMPT


class SafetyAgent(StructuredWorkflowComponent[SafetyReview]):
    schema = SafetyReview
    system_prompt = SAFETY_SYSTEM_PROMPT

    async def review(
        self,
        draft_answer: str,
        inquiry: InquiryState,
        evidence: EvidenceResult,
        syndrome: SyndromeAnalysis,
    ) -> SafetyReview:
        return await self.invoke_structured(
            {
                "draft_answer": draft_answer,
                "inquiry_state": inquiry.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
                "syndrome_analysis": syndrome.model_dump(mode="json"),
                "output_contract": {
                    "check_risk_flags": True,
                    "check_direct_diagnosis": True,
                    "check_prescription": True,
                    "check_dosage": True,
                    "require_offline_medical_advice_for_risk_flags": True,
                },
            }
        )
