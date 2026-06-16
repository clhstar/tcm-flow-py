from pathlib import Path
from typing import Protocol

import numpy as np

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
from experiments.rag_v1_6.schema import PublicTcmQgChunk


class Embedder(Protocol):
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ):
        ...


def load_bge_m3_embedder(
    *,
    model_name: str,
    revision: str,
    device: str,
    max_length: int,
):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision, device=device)
    model.max_seq_length = max_length
    return model


def _write_dense_vectors(
    *,
    dense_path: Path,
    chunks: list[dict],
    embedder: Embedder,
    batch_size: int,
) -> None:
    vectors = embedder.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(chunks):
        raise ValueError("dense embeddings must be a two-dimensional row-aligned array")
    np.save(dense_path, array)


def _build_strategy_index(
    *,
    strategy: str,
    chunk_path: Path,
    output_dir: Path,
    chunk_sha256: str,
    embedder: Embedder,
    embedding_model: str,
    embedding_revision: str,
    batch_size: int,
) -> dict:
    chunks = [
        PublicTcmQgChunk.model_validate(row).model_dump(mode="json")
        for row in read_jsonl(chunk_path, label=f"{strategy} formal chunks")
    ]
    if not chunks:
        raise ValueError(f"cannot build an empty formal index: {strategy}")
    if any(chunk["strategy"] != strategy for chunk in chunks):
        raise ValueError(f"{strategy} chunk file contains another strategy")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    tokens_path = output_dir / "bm25_tokens.jsonl"
    dense_path = output_dir / "dense.npy"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(rows_path, chunks)
    write_jsonl(
        tokens_path,
        [
            {
                "chunk_id": chunk["chunk_id"],
                "tokens": tokenize_text(chunk["text"]),
            }
            for chunk in chunks
        ],
    )
    _write_dense_vectors(
        dense_path=dense_path,
        chunks=chunks,
        embedder=embedder,
        batch_size=batch_size,
    )
    manifest = {
        "version": VERSION,
        "status": "ready",
        "strategy": strategy,
        "backend": "bm25_dense_rrf_rerank_ready",
        "row_count": len(chunks),
        "chunk_sha256": chunk_sha256,
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
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
            "dense": {
                "path": dense_path.name,
                "sha256": sha256_file(dense_path),
                "bytes": dense_path.stat().st_size,
            },
        },
        "built_at": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def build_public_tcm_qg_formal_indexes(
    *,
    chunks_dir: Path,
    chunk_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
    embedder: Embedder | None = None,
    embedding_model: str = "BAAI/bge-m3",
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181",
    device: str = "cuda",
    batch_size: int = 4,
    max_length: int = 1024,
    prereg_manifest_path: Path | None = None,
) -> dict:
    chunk_manifest = read_json(chunk_manifest_path, label="formal chunk manifest")
    if chunk_manifest.get("status") != "ready":
        raise ValueError("formal chunk manifest must be ready")
    if embedder is None:
        embedder = load_bge_m3_embedder(
            model_name=embedding_model,
            revision=embedding_revision,
            device=device,
            max_length=max_length,
        )
    strategies = {}
    for strategy in ("b4", "child"):
        record = chunk_manifest["strategies"][strategy]
        chunk_path = chunks_dir / record["output_file"]
        actual_sha256 = sha256_file(chunk_path)
        if actual_sha256 != record["output_sha256"]:
            raise ValueError(f"{strategy} formal chunk sha256 mismatch")
        strategy_manifest = _build_strategy_index(
            strategy=strategy,
            chunk_path=chunk_path,
            output_dir=output_dir / strategy,
            chunk_sha256=actual_sha256,
            embedder=embedder,
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
            batch_size=batch_size,
        )
        strategy_manifest_path = output_dir / strategy / "manifest.json"
        strategies[strategy] = {
            "backend": strategy_manifest["backend"],
            "row_count": strategy_manifest["row_count"],
            "embedding_model": strategy_manifest["embedding_model"],
            "embedding_revision": strategy_manifest["embedding_revision"],
            "manifest_sha256": sha256_file(strategy_manifest_path),
            "files": strategy_manifest["files"],
        }
    inputs = {
        "chunk_manifest_sha256": sha256_file(chunk_manifest_path),
    }
    if prereg_manifest_path is not None and prereg_manifest_path.is_file():
        inputs["prereg_manifest_sha256"] = sha256_file(prereg_manifest_path)
    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_formal_indexes_built",
        "generated_at": utc_now(),
        "backend": "bm25_dense_rrf_rerank_ready",
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "strategies": strategies,
        "inputs": inputs,
        "privacy": {"raw_text_included": False, "committed_full_index": False},
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
