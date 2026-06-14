from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


ContentType = Literal["clause", "formula", "ingredients", "preparation", "note"]
AnomalyStatus = Literal["open", "reviewed", "ignored"]
ChunkStrategy = Literal["c0", "c1", "c2", "c3", "c4"]
AuditSampleType = Literal["clause", "formula", "note_or_boundary"]
AuditStatus = Literal["pending", "pass", "fail"]
ReviewDecision = Literal[
    "correct",
    "boundary_error",
    "type_error",
    "parent_error",
    "text_error",
]
QuestionType = Literal[
    "single_clause_fact",
    "formula_composition_or_use",
    "source_location",
    "multi_evidence",
    "unanswerable",
]
PilotSplit = Literal["smoke", "pilot", "formal"]
PilotBookScope = Literal[
    "shang_han_lun",
    "jin_gui_yao_lue",
    "both",
]


def normalize_sha256(value: str) -> str:
    return value.upper()


class EvidenceUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    book_id: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    volume: str
    chapter_id: str = Field(min_length=1)
    chapter_title: str = Field(min_length=1)
    clause_id: str = Field(min_length=1)
    clause_number: int | None = Field(default=None, ge=1)
    content_type: ContentType
    parent_id: str = Field(min_length=1)
    original_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    notes: list[str]
    source_file: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    corpus_version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")

    @field_validator("source_hash")
    @classmethod
    def normalize_source_hash(cls, value: str) -> str:
        return normalize_sha256(value)


class ChunkUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    strategy: ChunkStrategy
    book_id: str = Field(min_length=1)
    chapter_id: str = Field(min_length=1)
    clause_id: str | None
    retrieval_parent_id: str | None
    source_evidence_ids: list[str] = Field(min_length=1)
    text: str = Field(min_length=1)
    context_text: str = Field(min_length=1)
    char_count: int = Field(ge=1)
    start_index: int | None = Field(default=None, ge=0)
    source_hash: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    corpus_version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")

    @field_validator("source_hash")
    @classmethod
    def normalize_source_hash(cls, value: str) -> str:
        return normalize_sha256(value)


class AuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str = Field(min_length=1)
    book_id: str = Field(min_length=1)
    sample_type: AuditSampleType
    chapter_id: str = Field(min_length=1)
    clause_id: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    original_text: str = Field(min_length=1)
    structured_summary: str = Field(min_length=1)
    status: AuditStatus = "pending"
    decision: ReviewDecision | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None
    comment: str = ""


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    strategy: ChunkStrategy
    rank: int = Field(ge=1)
    text: str = Field(min_length=1)
    context_text: str = Field(min_length=1)
    source_evidence_ids: list[str] = Field(min_length=1)
    clause_ids: list[str] = Field(min_length=1)
    retrieval_parent_id: str | None
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None


class PilotEvidenceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(min_length=1)
    split: Literal["pilot"]
    book_scope: Literal["shang_han_lun", "jin_gui_yao_lue"]
    question_type: QuestionType
    anchor_evidence_ids: list[str]
    anchor_clause_ids: list[str]
    selection_seed: int
    selection_reason: str = Field(min_length=1)
    absence_queries: list[str] = Field(default_factory=list)


class PilotQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    question_type: QuestionType
    book_scope: PilotBookScope
    answerable: bool
    reference_answer: str
    gold_evidence_ids: list[str]
    gold_clause_ids: list[str]
    graded_relevance: dict[str, int]
    support_spans: list[str]
    review_status: Literal["draft", "approved", "rejected"]
    split: PilotSplit | None = None
    evidence_group_id: str | None = None
    question_version: int = Field(default=1, ge=1)

    @field_validator("graded_relevance")
    @classmethod
    def validate_graded_relevance(
        cls,
        value: dict[str, int],
    ) -> dict[str, int]:
        invalid = {
            relevance for relevance in value.values() if relevance not in {1, 2}
        }
        if invalid:
            raise ValueError("graded_relevance 只允许 1 或 2")
        return value

    @model_validator(mode="after")
    def validate_answer_contract(self) -> "PilotQuestion":
        gold_fields = (
            self.gold_evidence_ids,
            self.gold_clause_ids,
            self.support_spans,
        )
        if self.answerable and any(not field for field in gold_fields):
            raise ValueError(
                "可回答问题必须包含 gold Evidence、gold clause 和 support span"
            )
        if not self.answerable and any(gold_fields):
            raise ValueError("无答案问题的 gold/support 字段必须为空")
        if not self.answerable and self.graded_relevance:
            raise ValueError("无答案问题的 graded_relevance 必须为空")
        if self.review_status == "approved" and not self.reference_answer.strip():
            raise ValueError("approved 问题必须填写参考答案或无答案")
        return self


class ParseAnomaly(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anomaly_id: str = Field(min_length=1)
    book_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    chapter_id: str = Field(min_length=1)
    chapter_title: str = Field(min_length=1)
    clause_id: str | None
    reason: str = Field(min_length=1)
    original_text: str = Field(min_length=1)
    status: AnomalyStatus = "open"


class ParseStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chapter_count: int = Field(ge=0)
    clause_count: int = Field(ge=0)
    formula_count: int = Field(ge=0)
    ingredients_count: int = Field(ge=0)
    preparation_count: int = Field(ge=0)
    note_count: int = Field(ge=0)
    anomaly_count: int = Field(ge=0)


class ParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_units: list[EvidenceUnit]
    anomalies: list[ParseAnomaly]
    statistics: ParseStatistics
