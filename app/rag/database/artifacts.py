import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RagArtifactBundle:
    corpus_id: str
    corpus_manifest: dict
    index_manifest: dict
    parents: list[dict]
    chunks: list[dict]
    rows: list[dict]
    tokens_by_chunk_id: dict[str, list[str]]
    dense: np.ndarray


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_artifact_bundle(corpus_dir: Path, index_dir: Path) -> RagArtifactBundle:
    corpus_manifest = _read_json(corpus_dir / "manifest.json")
    index_manifest = _read_json(index_dir / "manifest.json")
    if corpus_manifest.get("status") != "ready":
        raise ValueError("corpus manifest status must be ready")
    if index_manifest.get("status") != "ready":
        raise ValueError("index manifest status must be ready")
    if int(index_manifest.get("vector_dimension", 0)) != 1024:
        raise ValueError("vector_dimension must be 1024 for BGE-M3 pgvector storage")

    parents = _read_jsonl(corpus_dir / "parents.jsonl")
    chunks = _read_jsonl(corpus_dir / "chunks.jsonl")
    rows = _read_jsonl(index_dir / "rows.jsonl")
    token_rows = _read_jsonl(index_dir / "bm25_tokens.jsonl")
    dense = np.load(index_dir / "dense.npy", allow_pickle=False)

    row_ids = [row["chunk_id"] for row in rows]
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    tokens_by_chunk_id = {row["chunk_id"]: row["tokens"] for row in token_rows}
    if (
        len(chunks_by_id) != len(chunks)
        or len(tokens_by_chunk_id) != len(token_rows)
        or set(row_ids) != set(chunks_by_id)
        or set(row_ids) != set(tokens_by_chunk_id)
    ):
        raise ValueError("rows, chunks, and bm25 token ids must match")
    if dense.shape != (len(rows), 1024):
        raise ValueError("dense.npy shape does not match row_count and vector_dimension")
    chunks = [chunks_by_id[chunk_id] for chunk_id in row_ids]

    parent_ids = [parent["parent_id"] for parent in parents]
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("duplicate parent_id in artifact bundle")
    parent_id_set = set(parent_ids)
    orphan_ids = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["parent_id"] not in parent_id_set
    ]
    if orphan_ids:
        raise ValueError(f"orphan chunks found: {orphan_ids[:3]}")

    version = corpus_manifest.get("version") or index_manifest.get("version") or "v1.0.0"
    return RagArtifactBundle(
        corpus_id=f"ancient-books-{version}",
        corpus_manifest=corpus_manifest,
        index_manifest=index_manifest,
        parents=parents,
        chunks=chunks,
        rows=rows,
        tokens_by_chunk_id=tokens_by_chunk_id,
        dense=dense.astype(np.float32, copy=False),
    )
