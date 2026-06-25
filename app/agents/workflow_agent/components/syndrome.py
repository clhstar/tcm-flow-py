from __future__ import annotations

import re

from app.agents.workflow_agent.components.base import StructuredWorkflowComponent
from app.agents.workflow_agent.models import (
    EvidenceResult,
    InquiryState,
    SyndromeAnalysis,
    filter_allowed_patterns,
)
from app.agents.workflow_agent.prompts import SYNDROME_SYSTEM_PROMPT


class SyndromeAgent(StructuredWorkflowComponent[SyndromeAnalysis]):
    schema = SyndromeAnalysis
    system_prompt = SYNDROME_SYSTEM_PROMPT

    async def analyze(
        self,
        user_text: str,
        inquiry: InquiryState,
        evidence: EvidenceResult,
    ) -> SyndromeAnalysis:
        analysis = await self.invoke_structured(
            {
                "user_question": user_text,
                "inquiry_state": inquiry.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
                "allowed_terms": evidence.allowed_terms,
                "output_contract": {
                    "do_not_diagnose": True,
                    "possible_patterns_only": True,
                    "use_only_allowed_terms": True,
                    "use_only_evidence_ids": ["E1", "E2", "E3", "E4", "E5"],
                },
            }
        )
        return self._filter_to_evidence_contract(analysis, evidence)

    def _filter_to_evidence_contract(
        self,
        analysis: SyndromeAnalysis,
        evidence: EvidenceResult,
    ) -> SyndromeAnalysis:
        allowed_ids = {
            item.citation_id or item.id
            for item in evidence.evidence
            if re.fullmatch(r"E[1-5]", item.citation_id or item.id)
        }
        filtered = filter_allowed_patterns(analysis, evidence.allowed_terms)
        for pattern in filtered.possible_patterns:
            pattern.supporting_evidence = [
                evidence_id
                for evidence_id in pattern.supporting_evidence
                if evidence_id in allowed_ids
            ]
        return filtered
