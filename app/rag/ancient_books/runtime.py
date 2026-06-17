import json
from dataclasses import dataclass
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .pipeline import sha256_file
from .schema import EvidenceParent, RetrievalChunk


@dataclass
class LoadedProductionIndex:
    rows: list[RetrievalChunk]
    row_by_id: dict[str, RetrievalChunk]
    parents: dict[str, dict]
    bm25: object
    dense: np.ndarray
    manifest: dict


def _verified_path(root: Path, record: dict) -> Path:
    path = root / record["path"]
    if not path.is_file():
        raise FileNotFoundError(f"索引文件不存在: {path}")
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"索引文件大小不匹配: {path.name}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"索引文件 SHA256 不匹配: {path.name}")
    return path


def load_index(index_dir: Path, corpus_dir: Path) -> LoadedProductionIndex:
    index_manifest_path = index_dir / "manifest.json"
    corpus_manifest_path = corpus_dir / "manifest.json"
    if not index_manifest_path.is_file():
        raise FileNotFoundError(f"缺少索引 Manifest: {index_manifest_path}")
    if not corpus_manifest_path.is_file():
        raise FileNotFoundError(f"缺少语料 Manifest: {corpus_manifest_path}")

    manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        raise ValueError("生产索引状态不是 ready")
    if sha256_file(corpus_manifest_path) != manifest["corpus_manifest_sha256"]:
        raise ValueError("语料 Manifest 哈希与索引不一致")

    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    rows_path = _verified_path(index_dir, manifest["files"]["rows"])
    tokens_path = _verified_path(index_dir, manifest["files"]["bm25_tokens"])
    dense_path = _verified_path(index_dir, manifest["files"]["dense"])
    parents_path = _verified_path(corpus_dir, corpus_manifest["files"]["parents"])

    rows = [
        RetrievalChunk.model_validate_json(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    token_rows = [
        json.loads(line)
        for line in tokens_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parent_models = [
        EvidenceParent.model_validate_json(line)
        for line in parents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dense = np.load(dense_path, allow_pickle=False)

    row_ids = [row.chunk_id for row in rows]
    token_ids = [row["chunk_id"] for row in token_rows]
    if row_ids != token_ids:
        raise ValueError("rows 与 BM25 tokens 行号不一致")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("运行时索引存在重复 chunk_id")
    if len(rows) != manifest["row_count"] or len(rows) != len(dense):
        raise ValueError("rows、Dense 与 Manifest 行数不一致")
    if dense.ndim != 2 or dense.dtype != np.float32 or not np.isfinite(dense).all():
        raise ValueError("Dense 索引形状、dtype 或数值非法")
    if dense.shape[1] != manifest["vector_dimension"]:
        raise ValueError("Dense 索引维度与 Manifest 不一致")
    if not np.allclose(np.linalg.norm(dense, axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Dense 索引包含未归一化向量")

    tokens = [row["tokens"] for row in token_rows]
    if any(not isinstance(row_tokens, list) or not row_tokens for row_tokens in tokens):
        raise ValueError("BM25 索引存在空 token 行")
    parents = {
        parent.parent_id: parent.model_dump(mode="json") for parent in parent_models
    }
    if len(parents) != len(parent_models):
        raise ValueError("运行时语料存在重复 parent_id")
    orphan_ids = [row.chunk_id for row in rows if row.parent_id not in parents]
    if orphan_ids:
        raise ValueError(f"运行时索引存在孤儿 Child: {orphan_ids[:3]}")

    return LoadedProductionIndex(
        rows=rows,
        row_by_id={row.chunk_id: row for row in rows},
        parents=parents,
        bm25=BM25Okapi(tokens),
        dense=dense,
        manifest=manifest,
    )


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    *,
    rrf_k: int,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for source, chunk_ids in sorted(rankings.items()):
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def recover_parents(hits: list[dict], parents: dict[str, dict]) -> list[dict]:
    recovered = []
    seen = set()
    for hit in hits:
        parent_id = hit["parent_id"]
        if parent_id in seen:
            continue
        parent = parents.get(parent_id)
        if parent is None:
            raise ValueError(f"找不到 Parent: {parent_id}")
        recovered.append({**parent, **hit})
        seen.add(parent_id)
    return recovered


class ProductionRetrievalEngine:
    def __init__(self, *, index, encoder, reranker, settings):
        self.index = index
        self.encoder = encoder
        self.reranker = reranker
        self.settings = settings

    def _eligible(self, chief_symptom: str | None) -> list[int]:
        return [
            index
            for index, row in enumerate(self.index.rows)
            if not chief_symptom or chief_symptom in row.symptom_tags
        ]

    def _bm25(self, query: str, eligible: list[int]) -> list[str]:
        tokens = [
            token.strip()
            for token in jieba.lcut(query, HMM=False)
            if token.strip()
        ]
        scores = self.index.bm25.get_scores(tokens)
        return [
            self.index.rows[index].chunk_id
            for index in sorted(
                eligible,
                key=lambda item: (
                    -float(scores[item]),
                    self.index.rows[item].chunk_id,
                ),
            )[: int(self.settings["bm25_top_k"])]
        ]

    def _dense(self, query: str, eligible: list[int]) -> list[str]:
        encoded = np.asarray(self.encoder.encode([query]), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != 1:
            raise ValueError("查询 Dense Encoder 必须返回一行向量")
        vector = encoded[0]
        if vector.ndim != 1 or vector.shape[0] != self.index.dense.shape[1]:
            raise ValueError("查询 Dense 向量维度不匹配")
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(vector).all():
            raise ValueError("查询 Dense 向量无效")
        scores = self.index.dense @ (vector / norm)
        return [
            self.index.rows[index].chunk_id
            for index in sorted(
                eligible,
                key=lambda item: (
                    -float(scores[item]),
                    self.index.rows[item].chunk_id,
                ),
            )[: int(self.settings["dense_top_k"])]
        ]

    def retrieve(
        self,
        query: str,
        *,
        chief_symptom: str | None,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> dict:
        if mode not in {"hybrid", "vector", "keyword"}:
            mode = "hybrid"
        eligible = self._eligible(chief_symptom)
        if not eligible:
            return {
                "status": "insufficient_evidence",
                "retrieval_mode": mode,
                "degraded": False,
                "degraded_reason": None,
                "results": [],
            }

        bm25_ids = [] if mode == "vector" else self._bm25(query, eligible)
        dense_ids: list[str] = []
        degraded = False
        degraded_reason = None
        try:
            dense_ids = [] if mode == "keyword" else self._dense(query, eligible)
            rankings = {}
            if bm25_ids:
                rankings["bm25"] = bm25_ids
            if dense_ids:
                rankings["dense"] = dense_ids
            fused = reciprocal_rank_fusion(
                rankings,
                rrf_k=int(self.settings["rrf_k"]),
            )
            candidate_ids = [
                chunk_id
                for chunk_id, _ in fused[: int(self.settings["reranker_candidate_k"])]
            ]
            if mode == "hybrid":
                raw_scores = self.reranker.score(
                    [
                        [query, self.index.row_by_id[chunk_id].text]
                        for chunk_id in candidate_ids
                    ]
                )
                if isinstance(raw_scores, (int, float)):
                    scores = [float(raw_scores)]
                else:
                    scores = [float(score) for score in raw_scores]
                if len(scores) != len(candidate_ids):
                    raise ValueError("Reranker 分数数量与候选数量不一致")
                ranked = sorted(
                    zip(candidate_ids, scores),
                    key=lambda item: (-item[1], item[0]),
                )
            else:
                ranked = [
                    (chunk_id, score)
                    for chunk_id, score in fused
                    if chunk_id in candidate_ids
                ]
        except Exception as error:
            degraded = True
            degraded_reason = str(error)
            mode = "keyword"
            ranked = [(chunk_id, 0.0) for chunk_id in bm25_ids]

        bm25_ranks = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, 1)}
        dense_ranks = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, 1)}
        child_hits = []
        for chunk_id, score in ranked:
            row = self.index.row_by_id[chunk_id]
            sources = [
                source
                for source, ranks in (
                    ("bm25", bm25_ranks),
                    ("dense", dense_ranks),
                )
                if chunk_id in ranks
            ]
            child_hits.append(
                {
                    "chunk_id": chunk_id,
                    "parent_id": row.parent_id,
                    "matched_child": row.text,
                    "score": score,
                    "retrieval_sources": sources or ["bm25"],
                    "bm25_rank": bm25_ranks.get(chunk_id),
                    "dense_rank": dense_ranks.get(chunk_id),
                }
            )
        limit = min(int(top_k), int(self.settings["final_top_k"]), 5)
        parents = recover_parents(child_hits, self.index.parents)[:limit]
        for index, parent in enumerate(parents, start=1):
            parent["citation_id"] = f"E{index}"
            parent["content"] = parent["original_text"]
        return {
            "status": "ok" if parents else "insufficient_evidence",
            "retrieval_mode": mode,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "results": parents,
        }
