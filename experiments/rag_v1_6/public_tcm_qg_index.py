from pathlib import Path

import yaml

from experiments.rag_v1_6.common import (
    VERSION,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    tokenize_text,
    utc_now,
    write_jsonl,
)
from experiments.rag_v1_6.schema import PublicTcmQgChunk, PublicTcmQgQaPair


def _window_starts(text_length: int, *, size: int, overlap: int) -> list[int]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("invalid chunk size/overlap")
    if text_length <= size:
        return [0]
    starts = []
    start = 0
    step = size - overlap
    while start < text_length:
        starts.append(start)
        if start + size >= text_length:
            break
        start += step
    return starts


def build_public_tcm_qg_chunks_from_rows(
    *,
    rows: list[dict],
    b4_chunk_size: int,
    b4_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> dict[str, list[dict]]:
    by_doc: dict[str, dict] = {}
    for row in rows:
        pair = PublicTcmQgQaPair.model_validate(row)
        document = by_doc.setdefault(
            pair.source_doc_id,
            {
                "source_doc_id": pair.source_doc_id,
                "source_text": pair.source_text,
                "qa_ids": [],
            },
        )
        if document["source_text"] != pair.source_text:
            raise ValueError(f"source text drift for {pair.source_doc_id}")
        document["qa_ids"].append(pair.qa_id)

    chunks = {"b4": [], "child": []}
    for source_doc_id in sorted(by_doc):
        document = by_doc[source_doc_id]
        text = document["source_text"]
        parent_id = f"tcmqg-{source_doc_id}"
        qa_ids = sorted(document["qa_ids"])
        for index, start in enumerate(
            _window_starts(
                len(text),
                size=b4_chunk_size,
                overlap=b4_chunk_overlap,
            )
        ):
            chunk_text = text[start : start + b4_chunk_size]
            chunk = PublicTcmQgChunk(
                chunk_id=f"{parent_id}-b4-{index:03d}",
                strategy="b4",
                source_doc_id=source_doc_id,
                parent_id=parent_id,
                text=chunk_text,
                context_text=chunk_text,
                start_index=start,
                char_count=len(chunk_text),
                context_start_index=start,
                context_char_count=len(chunk_text),
                source_qa_ids=qa_ids,
            )
            chunks["b4"].append(chunk.model_dump(mode="json"))
        for index, start in enumerate(
            _window_starts(
                len(text),
                size=child_chunk_size,
                overlap=child_chunk_overlap,
            )
        ):
            chunk_text = text[start : start + child_chunk_size]
            chunk = PublicTcmQgChunk(
                chunk_id=f"{parent_id}-child-{index:03d}",
                strategy="child",
                source_doc_id=source_doc_id,
                parent_id=parent_id,
                text=chunk_text,
                context_text=text,
                start_index=start,
                char_count=len(chunk_text),
                context_start_index=0,
                context_char_count=len(text),
                source_qa_ids=qa_ids,
            )
            chunks["child"].append(chunk.model_dump(mode="json"))
    return chunks


def build_public_tcm_qg_chunks(
    *,
    dataset_path: Path,
    config_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = read_jsonl(dataset_path, label="public TCM-QG dataset")
    chunks = build_public_tcm_qg_chunks_from_rows(
        rows=rows,
        b4_chunk_size=int(config["chunking"]["b4_chunk_size"]),
        b4_chunk_overlap=int(config["chunking"]["b4_chunk_overlap"]),
        child_chunk_size=int(config["chunking"]["child_chunk_size"]),
        child_chunk_overlap=int(config["chunking"]["child_chunk_overlap"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = {}
    for strategy, strategy_rows in chunks.items():
        chunk_path = output_dir / f"{strategy}.jsonl"
        write_jsonl(chunk_path, strategy_rows)
        strategies[strategy] = {
            "output_file": chunk_path.name,
            "count": len(strategy_rows),
            "output_sha256": sha256_file(chunk_path),
        }
    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_chunks_built",
        "generated_at": utc_now(),
        "dataset_sha256": sha256_file(dataset_path),
        "config_sha256": sha256_file(config_path),
        "strategies": strategies,
        "privacy": {"raw_text_included": False},
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _build_strategy_index(
    *,
    chunk_path: Path,
    output_dir: Path,
    chunk_sha256: str,
) -> dict:
    chunks = [
        PublicTcmQgChunk.model_validate(row).model_dump(mode="json")
        for row in read_jsonl(chunk_path, label=f"{chunk_path.name} chunks")
    ]
    if not chunks:
        raise ValueError("cannot index empty chunk file")
    rows_path = output_dir / "rows.jsonl"
    tokens_path = output_dir / "bm25_tokens.jsonl"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows_path, chunks)
    token_rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "tokens": tokenize_text(chunk["text"]),
        }
        for chunk in chunks
    ]
    if any(not row["tokens"] for row in token_rows):
        raise ValueError("BM25 tokenization produced an empty row")
    write_jsonl(tokens_path, token_rows)
    manifest = {
        "version": VERSION,
        "status": "ready",
        "strategy": chunks[0]["strategy"],
        "backend": "jieba_bm25_plus_overlap_rerank",
        "row_count": len(chunks),
        "chunk_sha256": chunk_sha256,
        "files": {
            "rows": {
                "path": rows_path.name,
                "sha256": sha256_file(rows_path),
                "bytes": rows_path.stat().st_size,
            },
            "bm25_tokens": {
                "path": tokens_path.name,
                "sha256": sha256_file(tokens_path),
                "bytes": tokens_path.stat().st_size,
            },
        },
        "built_at": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def build_public_tcm_qg_indexes(
    *,
    chunks_dir: Path,
    chunk_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict:
    chunk_manifest = read_json(chunk_manifest_path, label="chunk manifest")
    if chunk_manifest.get("status") != "ready":
        raise ValueError("chunk manifest must be ready")
    strategies = {}
    for strategy in ("b4", "child"):
        record = chunk_manifest["strategies"][strategy]
        chunk_path = chunks_dir / record["output_file"]
        actual_sha256 = sha256_file(chunk_path)
        if actual_sha256 != record["output_sha256"]:
            raise ValueError(f"{strategy} chunk sha256 mismatch")
        strategy_manifest = _build_strategy_index(
            chunk_path=chunk_path,
            output_dir=output_dir / strategy,
            chunk_sha256=actual_sha256,
        )
        strategy_manifest_path = output_dir / strategy / "manifest.json"
        strategies[strategy] = {
            "backend": strategy_manifest["backend"],
            "row_count": strategy_manifest["row_count"],
            "manifest_sha256": sha256_file(strategy_manifest_path),
            "files": strategy_manifest["files"],
        }
    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_indexes_built",
        "generated_at": utc_now(),
        "chunk_manifest_sha256": sha256_file(chunk_manifest_path),
        "backend": "jieba_bm25_plus_overlap_rerank",
        "strategies": strategies,
        "privacy": {"raw_text_included": False},
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
