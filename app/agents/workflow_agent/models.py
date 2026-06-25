from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.middlewares.clarification_controller import normalize_question_items


InformationSufficiency = Literal["sufficient", "insufficient"]
Confidence = Literal["low", "medium", "high"]
SafetyLevel = Literal["low", "medium", "high"]


class KnownFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: str = ""
    triggers: list[str] = Field(default_factory=list)
    associated_symptoms: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class InquiryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chief_complaint: str = ""
    known_facts: KnownFacts = Field(default_factory=KnownFacts)
    missing_info: list[str] = Field(default_factory=list)
    information_sufficiency: InformationSufficiency = "insufficient"
    clarification_questions: list[str] = Field(default_factory=list)
    should_pause_for_clarification: bool = False

    @field_validator("chief_complaint")
    @classmethod
    def normalize_chief_complaint(cls, value: str) -> str:
        return value.strip()

    @field_validator("missing_info")
    @classmethod
    def normalize_missing_info(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=max(len(value), 1))

    @field_validator("clarification_questions")
    @classmethod
    def normalize_questions(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=3)

    @model_validator(mode="after")
    def validate_pause_contract(self) -> "InquiryState":
        if self.should_pause_for_clarification and not self.clarification_questions:
            raise ValueError("clarification questions are required when pausing")
        if (
            self.information_sufficiency == "sufficient"
            and self.should_pause_for_clarification
        ):
            raise ValueError("sufficient inquiry state cannot pause for clarification")
        return self


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    citation_id: str
    role: str = ""
    text: str
    source: str = ""

    @field_validator("id", "citation_id", "role", "text", "source")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class EvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_status: str = "insufficient_evidence"
    retrieval_mode: str = "hybrid_parent"
    degraded: bool = False
    evidence: list[EvidenceItem] = Field(default_factory=list)
    allowed_terms: list[str] = Field(default_factory=list)
    raw_tool_content: str = ""

    @field_validator("allowed_terms")
    @classmethod
    def normalize_allowed_terms(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=max(len(value), 1))


class PatternCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"
    reason: str

    @field_validator("term", "reason")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class SyndromeAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    possible_patterns: list[PatternCandidate] = Field(default_factory=list)
    not_enough_for_diagnosis: bool = True
    need_more_info: list[str] = Field(default_factory=list)

    @field_validator("need_more_info")
    @classmethod
    def normalize_need_more_info(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=max(len(value), 1))


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_answer: str

    @field_validator("draft_answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        return value.strip()


class SafetyReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_risk_flags: bool = False
    risk_flags: list[str] = Field(default_factory=list)
    contains_diagnosis: bool = False
    contains_prescription: bool = False
    contains_dosage: bool = False
    needs_offline_medical_advice: bool = False
    final_safety_level: SafetyLevel = "low"
    rewrite_required: bool = False
    rewrite_instructions: list[str] = Field(default_factory=list)

    @field_validator("risk_flags", "rewrite_instructions")
    @classmethod
    def normalize_list(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=max(len(value), 1))


def filter_allowed_patterns(
    analysis: SyndromeAnalysis,
    allowed_terms: list[str],
) -> SyndromeAnalysis:
    allowed = set(allowed_terms)
    return SyndromeAnalysis(
        possible_patterns=[
            pattern
            for pattern in analysis.possible_patterns
            if pattern.term in allowed
        ],
        not_enough_for_diagnosis=analysis.not_enough_for_diagnosis,
        need_more_info=analysis.need_more_info,
    )
