from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SourceType = Literal["ancient_book", "curated_markdown"]
EvidenceRole = Literal[
    "diagnostic_method",
    "symptom_feature",
    "syndrome_pattern",
    "pathogenesis",
    "differential",
    "case",
]
CitationId = Literal["E1", "E2", "E3", "E4", "E5"]
NonEmptyString = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceHashMixin(BaseModel):
    source_hash: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")

    @field_validator("source_hash")
    @classmethod
    def normalize_source_hash(cls, value: str) -> str:
        return value.upper()


class SelectedSection(SourceHashMixin, StrictModel):
    section_id: NonEmptyString
    source_type: SourceType
    book_id: NonEmptyString
    book_title: NonEmptyString
    source_file: NonEmptyString
    volume: str
    chapter: str
    section: NonEmptyString
    symptom_tags: list[NonEmptyString]
    original_text: NonEmptyString


class EvidenceParent(SourceHashMixin, StrictModel):
    parent_id: NonEmptyString
    source_type: SourceType
    book_id: NonEmptyString
    book_title: NonEmptyString
    source_file: NonEmptyString
    volume: str
    chapter: str
    section: NonEmptyString
    symptom_tags: list[NonEmptyString]
    evidence_role: EvidenceRole
    original_text: NonEmptyString
    normalized_text: NonEmptyString


class RetrievalChunk(StrictModel):
    chunk_id: NonEmptyString
    parent_id: NonEmptyString
    text: str = Field(min_length=1, max_length=300)
    source_type: SourceType
    symptom_tags: list[NonEmptyString]
    evidence_role: EvidenceRole


class RetrievalHit(StrictModel):
    citation_id: CitationId
    chunk_id: NonEmptyString
    parent_id: NonEmptyString
    matched_child: NonEmptyString
    content: NonEmptyString
    source_type: SourceType
    book_title: NonEmptyString
    source_file: NonEmptyString
    volume: str
    chapter: str
    section: NonEmptyString
    symptom_tags: list[NonEmptyString]
    evidence_role: EvidenceRole
    retrieval_sources: list[NonEmptyString] = Field(min_length=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    rrf_score: float | None = Field(default=None, ge=0)
    reranker_score: float | None = None
