import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from experiments.rag_v1_5.indexing import (
    BgeM3DenseEncoder,
    DenseEncoder,
)
from experiments.rag_v1_5.reranker import (
    FlagRerankerScorer,
    RerankerScorer,
    rerank_hits,
    resolve_model_snapshot,
)
from experiments.rag_v1_5.schema import (
    ChunkStrategy,
    ChunkUnit,
    RetrievalHit,
)
from experiments.rag_v1_5.tokenization import tokenize_text


RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
DEFAULT_MODEL_MANIFEST_PATH = Path(
    "experiments/rag_v1_5/manifests/models-v1.5.0.json"
)
CLAUSE_ID_PATTERN = re.compile(
    r"^((?:shl|jgy)-chapter-\d{2}-\d{3})(?:-|$)"
)


@dataclass
class LoadedIndex:
    rows: list[ChunkUnit]
    tokens: list[list[str]]
    dense: np.ndarray
    manifest: dict
    bm25: BM25Okapi


@dataclass
class RetrievalResult:
    hits: list[RetrievalHit]
    latency: dict[str, float | int]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _verify_file(index_dir: Path, file_record: dict) -> Path:
    path = index_dir / file_record["path"]
    if not path.is_file():
        raise FileNotFoundError(f"索引文件不存在: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != file_record["sha256"]:
        raise ValueError(
            f"索引文件 SHA256 不匹配: "
            f"{path.name}, expected={file_record['sha256']}, "
            f"actual={actual_sha256}"
        )
    return path


def load_index(index_dir: Path) -> LoadedIndex:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少索引 Manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    rows_path = _verify_file(index_dir, files["rows"])
    tokens_path = _verify_file(index_dir, files["bm25_tokens"])
    dense_path = _verify_file(index_dir, files["dense"])

    rows = [
        ChunkUnit.model_validate_json(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    token_rows = [
        json.loads(line)
        for line in tokens_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dense = np.load(dense_path, allow_pickle=False)
    row_ids = [row.chunk_id for row in rows]
    token_ids = [row["chunk_id"] for row in token_rows]
    if row_ids != token_ids:
        raise ValueError("rows 与 BM25 tokens 行号不一致")
    if len(rows) != len(dense) or len(rows) != manifest["row_count"]:
        raise ValueError("rows、Dense 与 Manifest 行数不一致")
    if dense.dtype != np.float32 or not np.isfinite(dense).all():
        raise ValueError("Dense 索引 dtype 或数值非法")
    tokens = [row["tokens"] for row in token_rows]
    if any(not row_tokens for row_tokens in tokens):
        raise ValueError("BM25 索引存在空 token 行")
    return LoadedIndex(
        rows=rows,
        tokens=tokens,
        dense=dense,
        manifest=manifest,
        bm25=BM25Okapi(tokens),
    )


def _clause_ids(chunk: ChunkUnit) -> list[str]:
    if chunk.clause_id:
        return [chunk.clause_id]
    clause_ids = []
    for evidence_id in chunk.source_evidence_ids:
        match = CLAUSE_ID_PATTERN.match(evidence_id)
        if match and match.group(1) not in clause_ids:
            clause_ids.append(match.group(1))
    if not clause_ids:
        raise ValueError(f"{chunk.chunk_id} 无法映射 clause ID")
    return clause_ids


def _hit_from_chunk(
    chunk: ChunkUnit,
    *,
    rank: int,
    bm25_score: float | None = None,
    dense_score: float | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk.chunk_id,
        strategy=chunk.strategy,
        rank=rank,
        text=chunk.text,
        context_text=chunk.context_text,
        source_evidence_ids=chunk.source_evidence_ids,
        clause_ids=_clause_ids(chunk),
        retrieval_parent_id=chunk.retrieval_parent_id,
        bm25_rank=rank if bm25_score is not None else None,
        bm25_score=bm25_score,
        dense_rank=rank if dense_score is not None else None,
        dense_score=dense_score,
    )


def search_bm25(
    index: LoadedIndex,
    query: str,
    *,
    top_k: int,
) -> list[RetrievalHit]:
    query_tokens = tokenize_text(query)
    if not query_tokens:
        raise ValueError("查询不能为空")
    scores = index.bm25.get_scores(query_tokens)
    order = sorted(
        range(len(index.rows)),
        key=lambda row_index: (
            -float(scores[row_index]),
            index.rows[row_index].chunk_id,
        ),
    )
    return [
        _hit_from_chunk(
            index.rows[row_index],
            rank=rank,
            bm25_score=float(scores[row_index]),
        )
        for rank, row_index in enumerate(order[:top_k], start=1)
    ]


def search_dense(
    index: LoadedIndex,
    query_vector: np.ndarray,
    *,
    top_k: int,
) -> list[RetrievalHit]:
    vector = np.asarray(query_vector, dtype=np.float32)
    if vector.ndim == 2 and vector.shape[0] == 1:
        vector = vector[0]
    if vector.ndim != 1 or vector.shape[0] != index.dense.shape[1]:
        raise ValueError("Dense 查询向量维度不匹配")
    if not np.isfinite(vector).all():
        raise ValueError("Dense 查询向量包含 NaN 或 Inf")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Dense 查询向量不能为零向量")
    vector = vector / norm
    scores = index.dense @ vector
    order = sorted(
        range(len(index.rows)),
        key=lambda row_index: (
            -float(scores[row_index]),
            index.rows[row_index].chunk_id,
        ),
    )
    return [
        _hit_from_chunk(
            index.rows[row_index],
            rank=rank,
            dense_score=float(scores[row_index]),
        )
        for rank, row_index in enumerate(order[:top_k], start=1)
    ]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievalHit]],
    *,
    k: int = 60,
) -> list[RetrievalHit]:
    if k <= 0:
        raise ValueError("RRF k 必须大于 0")
    fused: dict[str, RetrievalHit] = {}
    scores: dict[str, float] = {}
    for source, hits in ranked_lists.items():
        if source not in {"bm25", "dense"}:
            raise ValueError(f"未知 RRF 来源: {source}")
        for hit in hits:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + (
                1.0 / (k + hit.rank)
            )
            existing = fused.get(hit.chunk_id)
            if existing is None:
                fused[hit.chunk_id] = hit
                existing = hit
            updates = {}
            if source == "bm25":
                updates.update(
                    bm25_rank=hit.bm25_rank or hit.rank,
                    bm25_score=hit.bm25_score,
                )
            else:
                updates.update(
                    dense_rank=hit.dense_rank or hit.rank,
                    dense_score=hit.dense_score,
                )
            fused[hit.chunk_id] = existing.model_copy(update=updates)

    ordered = sorted(
        fused.values(),
        key=lambda hit: (-scores[hit.chunk_id], hit.chunk_id),
    )
    return [
        hit.model_copy(
            update={
                "rank": rank,
                "rrf_score": scores[hit.chunk_id],
            }
        )
        for rank, hit in enumerate(ordered, start=1)
    ]


def _encode_query(
    query: str,
    encoder: DenseEncoder,
) -> np.ndarray:
    vectors = np.asarray(encoder.encode([query]))
    if vectors.ndim != 2 or vectors.shape[0] != 1:
        raise ValueError("Dense Encoder 查询结果必须恰好包含一个向量")
    return vectors[0]


def retrieve_loaded(
    query: str,
    *,
    index: LoadedIndex,
    mode: RetrievalMode,
    config: dict,
    dense_encoder: DenseEncoder | None = None,
    reranker_scorer: RerankerScorer | None = None,
    result_top_k: int | None = None,
) -> RetrievalResult:
    total_started = time.perf_counter()
    if not query.strip():
        raise ValueError("查询不能为空")
    if mode not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
        raise ValueError(f"未知检索模式: {mode}")

    output_top_k = (
        int(config["reranker"]["top_k"])
        if result_top_k is None
        else int(result_top_k)
    )
    if output_top_k <= 0:
        raise ValueError("result_top_k 必须大于 0")

    latency: dict[str, float | int] = {
        "bm25_ms": 0.0,
        "dense_ms": 0.0,
        "rrf_ms": 0.0,
        "reranker_ms": 0.0,
        "total_ms": 0.0,
        "returned_context_chars": 0,
    }
    bm25_hits = []
    dense_hits = []
    if mode in {"bm25", "hybrid", "hybrid_rerank"}:
        started = time.perf_counter()
        bm25_hits = search_bm25(
            index,
            query,
            top_k=max(int(config["bm25"]["top_k"]), output_top_k),
        )
        latency["bm25_ms"] = (time.perf_counter() - started) * 1000

    if mode in {"dense", "hybrid", "hybrid_rerank"}:
        if dense_encoder is None:
            raise ValueError(f"{mode} 模式必须提供已加载 Dense Encoder")
        started = time.perf_counter()
        dense_hits = search_dense(
            index,
            _encode_query(query, dense_encoder),
            top_k=max(int(config["dense"]["top_k"]), output_top_k),
        )
        latency["dense_ms"] = (time.perf_counter() - started) * 1000

    if mode == "bm25":
        hits = bm25_hits[:output_top_k]
    elif mode == "dense":
        hits = dense_hits[:output_top_k]
    else:
        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_hits, "dense": dense_hits},
            k=int(config["rrf"]["k"]),
        )
        if mode == "hybrid":
            hits = fused[:output_top_k]
        else:
            hits = fused[: int(config["reranker"]["candidate_k"])]
        latency["rrf_ms"] = (time.perf_counter() - started) * 1000

        if mode == "hybrid_rerank":
            if reranker_scorer is None:
                raise ValueError(
                    "hybrid_rerank 模式必须提供已加载 Reranker"
                )
            started = time.perf_counter()
            hits = rerank_hits(
                query,
                hits,
                scorer=reranker_scorer,
                top_k=output_top_k,
            )
            latency["reranker_ms"] = (
                time.perf_counter() - started
            ) * 1000

    latency["returned_context_chars"] = sum(
        len(hit.context_text) for hit in hits[:5]
    )
    latency["total_ms"] = (time.perf_counter() - total_started) * 1000
    return RetrievalResult(hits=hits, latency=latency)


def retrieve(
    query: str,
    *,
    strategy: ChunkStrategy,
    mode: RetrievalMode,
    indexes_dir: Path,
    config: dict,
    dense_encoder: DenseEncoder | None = None,
    reranker_scorer: RerankerScorer | None = None,
    model_manifest_path: Path = DEFAULT_MODEL_MANIFEST_PATH,
    repository_root: Path | None = None,
) -> list[RetrievalHit]:
    if not query.strip():
        raise ValueError("查询不能为空")
    if mode not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
        raise ValueError(f"未知检索模式: {mode}")
    index = load_index(indexes_dir / strategy)
    if mode in {"dense", "hybrid", "hybrid_rerank"}:
        if dense_encoder is None:
            local_path, _ = resolve_model_snapshot(
                config=config,
                role="embedding",
                model_manifest_path=model_manifest_path,
                repository_root=repository_root,
            )
            dense_encoder = BgeM3DenseEncoder(
                local_path,
                config["embedding"],
            )
    if mode == "hybrid_rerank" and reranker_scorer is None:
        local_path, _ = resolve_model_snapshot(
            config=config,
            role="reranker",
            model_manifest_path=model_manifest_path,
            repository_root=repository_root,
        )
        reranker_scorer = FlagRerankerScorer(
            local_path,
            config["reranker"],
        )
    return retrieve_loaded(
        query,
        index=index,
        mode=mode,
        config=config,
        dense_encoder=dense_encoder,
        reranker_scorer=reranker_scorer,
        result_top_k=None,
    ).hits
