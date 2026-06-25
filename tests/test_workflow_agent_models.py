import unittest
from pydantic import ValidationError

from app.agents.workflow_agent.models import (
    AnswerDraft,
    EvidenceItem,
    EvidenceResult,
    InquiryState,
    KnownFacts,
    PatternCandidate,
    SafetyReview,
    SyndromeAnalysis,
    filter_allowed_patterns,
)


class WorkflowAgentModelTests(unittest.TestCase):
    def test_inquiry_state_caps_and_normalizes_clarification_questions(self):
        state = InquiryState(
            chief_complaint="胃胀",
            known_facts=KnownFacts(triggers=["油腻后加重"]),
            missing_info=["持续时间", "大便情况", "反酸烧心", "食欲"],
            information_sufficiency="insufficient",
            clarification_questions=[
                "胃胀持续多久了？",
                "大便情况如何？",
                "是否伴有反酸、烧心或腹痛？",
                "食欲怎么样？",
            ],
            should_pause_for_clarification=True,
        )

        self.assertEqual(
            state.clarification_questions,
            [
                "胃胀持续多久了？",
                "大便情况如何？",
                "是否伴有反酸、烧心或腹痛？",
            ],
        )

    def test_inquiry_pause_requires_questions(self):
        with self.assertRaisesRegex(ValidationError, "clarification"):
            InquiryState(
                chief_complaint="胃胀",
                information_sufficiency="insufficient",
                clarification_questions=[],
                should_pause_for_clarification=True,
            )

    def test_completed_syndrome_analysis_filters_unauthorized_terms(self):
        analysis = SyndromeAnalysis(
            possible_patterns=[
                PatternCandidate(
                    term="食滞",
                    supporting_evidence=["E1"],
                    confidence="medium",
                    reason="油腻后胃胀加重，并伴有嗳气。",
                ),
                PatternCandidate(
                    term="脾胃虚弱",
                    supporting_evidence=["E2"],
                    confidence="low",
                    reason="该术语不在本次证据允许范围内。",
                ),
            ],
            not_enough_for_diagnosis=True,
            need_more_info=["舌象", "大便"],
        )

        filtered = filter_allowed_patterns(analysis, ["食滞"])

        self.assertEqual(len(filtered.possible_patterns), 1)
        self.assertEqual(filtered.possible_patterns[0].term, "食滞")

    def test_safety_review_marks_rewrite_required_for_unsafe_content(self):
        review = SafetyReview(
            has_risk_flags=True,
            risk_flags=["胸痛"],
            contains_diagnosis=True,
            contains_prescription=True,
            contains_dosage=True,
            needs_offline_medical_advice=True,
            final_safety_level="high",
            rewrite_required=True,
            rewrite_instructions=[
                "删除直接诊断表达。",
                "删除方药和剂量表达。",
                "加入线下就医提醒。",
            ],
        )

        self.assertTrue(review.rewrite_required)
        self.assertEqual(review.final_safety_level, "high")

    def test_evidence_result_keeps_raw_tool_content_for_guardrails(self):
        result = EvidenceResult(
            retrieval_status="ok",
            retrieval_mode="hybrid_parent",
            degraded=False,
            evidence=[
                EvidenceItem(
                    id="E1",
                    citation_id="E1",
                    role="syndrome_pattern",
                    text="因食而胀。",
                    source="《景岳全书》 卷一 / 胃脘",
                )
            ],
            allowed_terms=["食滞"],
            raw_tool_content="允许使用的专业术语：\n- 食滞",
        )

        self.assertEqual(result.evidence[0].id, "E1")
        self.assertIn("食滞", result.raw_tool_content)

    def test_answer_draft_strips_outer_whitespace(self):
        draft = AnswerDraft(draft_answer="  当前只能做谨慎分析。  ")

        self.assertEqual(draft.draft_answer, "当前只能做谨慎分析。")


if __name__ == "__main__":
    unittest.main()
