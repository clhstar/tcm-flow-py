import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


EXPECTED_BOOK_IDS = frozenset(
    {
        "jing_yue_quan_shu",
        "yi_men_fa_lv",
        "zheng_yin_mai_zhi",
        "lei_zheng_zhi_cai",
        "zheng_zhi_hui_bu",
        "jin_gui_yao_lue",
        "huang_di_nei_jing_su_wen",
    }
)

NonEmptyString = Annotated[str, Field(min_length=1)]
AliasList = Annotated[list[NonEmptyString], Field(min_length=1)]
COMMIT_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BookConfig(StrictConfigModel):
    book_id: NonEmptyString
    title: NonEmptyString
    source_file: NonEmptyString
    symptom_scan: bool
    method_sections: list[NonEmptyString] = Field(default_factory=list)
    fixed_sections: list[NonEmptyString] = Field(default_factory=list)


class EmbeddingConfig(StrictConfigModel):
    model: NonEmptyString
    revision: NonEmptyString
    device: Literal["cuda"]
    use_fp16: Literal[True]
    batch_size: Literal[4]
    max_length: Literal[1024]


class RerankerConfig(StrictConfigModel):
    model: NonEmptyString
    revision: NonEmptyString
    device: Literal["cuda"]
    use_fp16: Literal[True]
    batch_size: Literal[2]
    max_length: Literal[1024]
    normalize_score: Literal[True]


class ModelsConfig(StrictConfigModel):
    embedding: EmbeddingConfig
    reranker: RerankerConfig


class RetrievalConfig(StrictConfigModel):
    bm25_top_k: Literal[20]
    dense_top_k: Literal[20]
    rrf_k: Literal[60]
    reranker_candidate_k: Literal[40]
    final_top_k: Literal[5]


class ProductionConfig(StrictConfigModel):
    version: Literal["v1.0.0"]
    source_encoding: Literal["cp936"]
    symptoms: dict[NonEmptyString, AliasList] = Field(min_length=10, max_length=10)
    exclude_title_patterns: list[NonEmptyString] = Field(min_length=1)
    books: list[BookConfig] = Field(min_length=7, max_length=7)
    models: ModelsConfig
    retrieval: RetrievalConfig

def _validate_books(raw_books: object) -> None:
    if not isinstance(raw_books, list):
        raise ValueError("production config books must be a list of exactly 7 books")
    if len(raw_books) != 7:
        raise ValueError(
            f"production config must contain exactly 7 books; found {len(raw_books)}"
        )

    book_ids = [
        book.get("book_id") if isinstance(book, dict) else None for book in raw_books
    ]
    if len(book_ids) != len(set(book_ids)):
        raise ValueError("production config contains duplicate book IDs")

    actual_book_ids = set(book_ids)
    if actual_book_ids != EXPECTED_BOOK_IDS:
        missing = sorted(EXPECTED_BOOK_IDS - actual_book_ids)
        unexpected = sorted(actual_book_ids - EXPECTED_BOOK_IDS, key=str)
        raise ValueError(
            "production config book IDs must exactly match EXPECTED_BOOK_IDS; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_model_revisions(raw_models: object) -> None:
    if not isinstance(raw_models, dict):
        raise ValueError("production config models must be a mapping")

    for model_kind in ("embedding", "reranker"):
        model = raw_models.get(model_kind)
        revision = model.get("revision") if isinstance(model, dict) else None
        if not isinstance(revision, str) or not COMMIT_HASH_PATTERN.fullmatch(revision):
            raise ValueError(
                f"models.{model_kind}.revision must be a 40-character hexadecimal "
                "commit hash"
            )


def load_production_config(path: Path) -> dict[str, Any]:
    """Load and validate a UTF-8 production ancient-book configuration."""

    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("production config root must be a YAML mapping")

    _validate_books(raw_config.get("books"))
    _validate_model_revisions(raw_config.get("models"))
    return ProductionConfig.model_validate(raw_config).model_dump()
