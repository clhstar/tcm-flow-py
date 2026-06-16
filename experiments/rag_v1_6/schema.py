from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PublicTcmQgSplit = Literal["train_pool", "dev", "test"]
PublicTcmQgMethod = Literal["B0", "B4", "P", "P-no-parent"]
PublicTcmQgChunkStrategy = Literal["b4", "child"]


class PublicTcmQgDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    annotations: list[dict[str, str]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_annotations(self) -> "PublicTcmQgDocument":
        for annotation in self.annotations:
            if not annotation.get("Q", "").strip():
                raise ValueError("annotation.Q cannot be empty")
            if not annotation.get("A", "").strip():
                raise ValueError("annotation.A cannot be empty")
        return self


class PublicTcmQgQaPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qa_id: str = Field(min_length=1)
    source_doc_id: str = Field(min_length=1)
    split: PublicTcmQgSplit
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    answer_start: int = Field(ge=0)
    answer_end: int = Field(gt=0)
    review_status: Literal["approved"]
    question_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_answer_span(self) -> "PublicTcmQgQaPair":
        if self.answer_end <= self.answer_start:
            raise ValueError("answer_end must be greater than answer_start")
        if self.source_text[self.answer_start : self.answer_end] != self.answer:
            raise ValueError("answer span does not match source_text")
        return self


class PublicTcmQgChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    strategy: PublicTcmQgChunkStrategy
    source_doc_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    context_text: str = Field(min_length=1)
    start_index: int = Field(ge=0)
    char_count: int = Field(ge=1)
    context_start_index: int = Field(ge=0)
    context_char_count: int = Field(ge=1)
    source_qa_ids: list[str] = Field(default_factory=list)


class PublicTcmQgRetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    strategy: PublicTcmQgChunkStrategy
    rank: int = Field(ge=1)
    source_doc_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    context_text: str = Field(min_length=1)
    start_index: int = Field(ge=0)
    char_count: int = Field(ge=1)
    context_start_index: int = Field(ge=0)
    context_char_count: int = Field(ge=1)
    bm25_score: float
    overlap_score: float
    score: float


class PublicTcmQgAnswerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qa_id: str = Field(min_length=1)
    source_doc_id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    method: PublicTcmQgMethod
    repeat_index: int = Field(ge=0)
    answer: str = Field(min_length=1)
    abstain: bool
    citations: list[str]
    retrieval_supported: bool
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    model_name: str = Field(min_length=1)
