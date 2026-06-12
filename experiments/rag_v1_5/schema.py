from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ContentType = Literal["clause", "formula", "ingredients", "preparation", "note"]
AnomalyStatus = Literal["open", "reviewed", "ignored"]


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
        return value.upper()


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
