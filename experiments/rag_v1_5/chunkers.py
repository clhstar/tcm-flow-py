"""Deterministic chunking strategies for the V1.5 retrieval experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from experiments.rag_v1_5.schema import ChunkUnit, EvidenceUnit


ChunkConfig = dict[str, Any]


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    start: int
    end: int


def load_evidence(path: Path) -> list[EvidenceUnit]:
    units = [
        EvidenceUnit.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(
        units,
        key=lambda unit: (
            unit.book_id,
            unit.chapter_id,
            unit.clause_number or 0,
            unit.evidence_id,
        ),
    )


def load_chunk_config(path: Path) -> ChunkConfig:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Chunk config must be a mapping")
    return config


def build_chunks(
    units: list[EvidenceUnit],
    strategy: str,
    config: ChunkConfig,
) -> list[ChunkUnit]:
    if strategy in {"c0", "c1"}:
        return _build_character_chunks(units, strategy, config)
    if strategy == "c2":
        return _build_clause_chunks(units, config)
    raise ValueError(f"Unsupported chunk strategy: {strategy}")


def _build_character_chunks(
    units: list[EvidenceUnit],
    strategy: str,
    config: ChunkConfig,
) -> list[ChunkUnit]:
    strategy_config = config["strategies"][strategy]
    separators = _with_character_fallback(
        config["shared"]["separators"]
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=strategy_config["chunk_size"],
        chunk_overlap=strategy_config["chunk_overlap"],
        separators=separators,
        add_start_index=True,
    )
    chapters: dict[tuple[str, str], list[EvidenceUnit]] = {}
    for unit in units:
        if unit.content_type != "clause":
            continue
        chapters.setdefault((unit.book_id, unit.chapter_id), []).append(unit)

    chunks: list[ChunkUnit] = []
    for chapter_units in chapters.values():
        chapter_text, spans = _join_evidence_text(chapter_units)
        units_by_id = {unit.evidence_id: unit for unit in chapter_units}
        for document in splitter.create_documents([chapter_text]):
            text = document.page_content
            start_index = document.metadata["start_index"]
            end_index = start_index + len(text)
            source_evidence_ids = [
                span.evidence_id
                for span in spans
                if span.start < end_index and span.end > start_index
            ]
            source_unit = units_by_id[source_evidence_ids[0]]
            chunks.append(
                ChunkUnit(
                    chunk_id=(
                        f"{strategy}-{source_unit.chapter_id}-"
                        f"{start_index:06d}"
                    ),
                    strategy=strategy,
                    book_id=source_unit.book_id,
                    chapter_id=source_unit.chapter_id,
                    clause_id=None,
                    retrieval_parent_id=None,
                    source_evidence_ids=source_evidence_ids,
                    text=text,
                    context_text=text,
                    char_count=len(text),
                    start_index=start_index,
                    source_hash=source_unit.source_hash,
                    corpus_version=source_unit.corpus_version,
                )
            )
    return chunks


def _join_evidence_text(
    units: Iterable[EvidenceUnit],
) -> tuple[str, list[EvidenceSpan]]:
    parts: list[str] = []
    spans: list[EvidenceSpan] = []
    cursor = 0
    for unit in units:
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(unit.normalized_text)
        cursor += len(unit.normalized_text)
        spans.append(
            EvidenceSpan(
                evidence_id=unit.evidence_id,
                start=start,
                end=cursor,
            )
        )
    return "".join(parts), spans


def _build_clause_chunks(
    units: list[EvidenceUnit],
    config: ChunkConfig,
) -> list[ChunkUnit]:
    strategy_config = config["strategies"]["c2"]
    separators = _with_character_fallback(
        config["shared"]["separators"]
    )
    chunks: list[ChunkUnit] = []
    for clause in units:
        if clause.content_type != "clause":
            continue
        parts = _split_text(
            clause.normalized_text,
            max_length=strategy_config["max_length"],
            chunk_size=strategy_config["overflow_chunk_size"],
            chunk_overlap=strategy_config["overflow_overlap"],
            separators=separators,
        )
        for part_number, (text, start_index) in enumerate(parts, start=1):
            chunks.append(
                ChunkUnit(
                    chunk_id=(
                        f"c2-{clause.evidence_id}-{part_number:03d}"
                    ),
                    strategy="c2",
                    book_id=clause.book_id,
                    chapter_id=clause.chapter_id,
                    clause_id=clause.evidence_id,
                    retrieval_parent_id=clause.evidence_id,
                    source_evidence_ids=[clause.evidence_id],
                    text=text,
                    context_text=text,
                    char_count=len(text),
                    start_index=start_index,
                    source_hash=clause.source_hash,
                    corpus_version=clause.corpus_version,
                )
            )
    return chunks


def _split_text(
    text: str,
    *,
    max_length: int,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[tuple[str, int]]:
    if len(text) <= max_length:
        return [(text, 0)]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        add_start_index=True,
    )
    return [
        (document.page_content, document.metadata["start_index"])
        for document in splitter.create_documents([text])
    ]


def _with_character_fallback(separators: list[str]) -> list[str]:
    if "" in separators:
        return separators
    return [*separators, ""]
