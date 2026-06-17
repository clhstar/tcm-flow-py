import json
from pathlib import Path

import jieba
import numpy as np

from .pipeline import sha256_file, write_json
from .schema import RetrievalChunk


def normalize_vectors(vectors, expected_count: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != expected_count:
        raise ValueError("Dense 向量形状不符合索引输入")
    if not np.isfinite(array).all():
        raise ValueError("Dense 向量包含 NaN 或 Inf")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Dense 向量不能为零向量")
    return (array / norms).astype(np.float32, copy=False)


def _read_chunks(path: Path) -> list[RetrievalChunk]:
    chunks = [
        RetrievalChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not chunks:
        raise ValueError("生产索引不能使用空 chunks.jsonl")
    chunks.sort(key=lambda item: item.chunk_id)
    if len({item.chunk_id for item in chunks}) != len(chunks):
        raise ValueError("生产索引存在重复 chunk_id")
    return chunks


def _write_json_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def build_index(
    *,
    chunks_path: Path,
    corpus_manifest_sha256: str,
    output_dir: Path,
    encoder,
    model_record: dict,
) -> dict:
    chunks = _read_chunks(chunks_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_path = output_dir / "rows.jsonl"
    tokens_path = output_dir / "bm25_tokens.jsonl"
    dense_path = output_dir / "dense.npy"
    _write_json_rows(
        rows_path,
        [item.model_dump(mode="json") for item in chunks],
    )
    _write_json_rows(
        tokens_path,
        [
            {
                "chunk_id": item.chunk_id,
                "tokens": jieba.lcut(item.text, HMM=False),
            }
            for item in chunks
        ],
    )
    vectors = normalize_vectors(
        encoder.encode([item.text for item in chunks]),
        len(chunks),
    )
    np.save(dense_path, vectors, allow_pickle=False)

    paths = {
        "rows": rows_path,
        "bm25_tokens": tokens_path,
        "dense": dense_path,
    }
    manifest = {
        "version": "v1.0.0",
        "status": "ready",
        "row_count": len(chunks),
        "vector_dimension": int(vectors.shape[1]),
        "corpus_manifest_sha256": corpus_manifest_sha256.upper(),
        "embedding_model": {
            "model": model_record["model"],
            "revision": model_record["revision"],
        },
        "files": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
