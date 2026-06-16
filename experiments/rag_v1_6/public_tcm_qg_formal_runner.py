from pathlib import Path
from dataclasses import dataclass
import time
from urllib.parse import urlparse

import numpy as np
import yaml
from dotenv import dotenv_values
from rank_bm25 import BM25Okapi

from experiments.rag_v1_6.common import (
    VERSION,
    atomic_write_json,
    compact_timestamp,
    mean,
    read_json,
    read_jsonl,
    sha256_file,
    tokenize_text,
    utc_now,
    write_jsonl,
)
from experiments.rag_v1_6.schema import PublicTcmQgChunk, PublicTcmQgQaPair


FORMAL_RETRIEVAL_CONFIGS = [
    {
        "config_id": "b1-public-bm25",
        "method_role": "B1",
        "strategy": "b4",
        "context_policy": "chunk",
        "retrieval_mode": "bm25",
        "uses_bm25": True,
        "uses_dense": False,
        "uses_reranker": False,
    },
    {
        "config_id": "b2-public-dense",
        "method_role": "B2",
        "strategy": "b4",
        "context_policy": "chunk",
        "retrieval_mode": "dense",
        "uses_bm25": False,
        "uses_dense": True,
        "uses_reranker": False,
    },
    {
        "config_id": "b3-public-hybrid",
        "method_role": "B3",
        "strategy": "b4",
        "context_policy": "chunk",
        "retrieval_mode": "hybrid_rrf",
        "uses_bm25": True,
        "uses_dense": True,
        "uses_reranker": False,
    },
    {
        "config_id": "b4-public-hybrid-rerank",
        "method_role": "B4",
        "strategy": "b4",
        "context_policy": "chunk",
        "retrieval_mode": "hybrid_rrf_rerank",
        "uses_bm25": True,
        "uses_dense": True,
        "uses_reranker": True,
    },
    {
        "config_id": "p-public-hybrid-rerank",
        "method_role": "P",
        "strategy": "child",
        "context_policy": "parent",
        "retrieval_mode": "hybrid_rrf_rerank",
        "uses_bm25": True,
        "uses_dense": True,
        "uses_reranker": True,
    },
    {
        "config_id": "p-public-no-parent",
        "method_role": "P-no-parent",
        "strategy": "child",
        "context_policy": "child",
        "retrieval_mode": "hybrid_rrf_rerank",
        "uses_bm25": True,
        "uses_dense": True,
        "uses_reranker": True,
        "ranking_must_match": "p-public-hybrid-rerank",
    },
    {
        "config_id": "p-public-no-reranker",
        "method_role": "P",
        "strategy": "child",
        "context_policy": "parent",
        "retrieval_mode": "hybrid_rrf",
        "uses_bm25": True,
        "uses_dense": True,
        "uses_reranker": False,
    },
]

FORMAL_ANSWER_METHODS = ["B0", "B4", "P", "P-no-parent"]

FORMAL_SUCCESS_GATES = {
    "strong_success": {
        "required": [
            "P_vs_B4_char_f1_delta > 0",
            "P_vs_B4_char_f1_95ci_lower > 0",
            "P_vs_B4_citation_recall_delta >= 0",
            "P_vs_B4_unsupported_answer_rate_delta <= 0",
            "P_vs_P_no_parent_char_f1_delta > 0",
            "P_vs_P_no_parent_char_f1_95ci_lower > 0",
            "P_vs_P_no_parent_citation_recall_delta > 0",
            "P_vs_P_no_parent_citation_recall_95ci_lower > 0",
        ],
        "allowed_thesis_claim": (
            "Parent-Child RAG is stably better than ordinary chunk RAG "
            "on public TCM-QG."
        ),
    },
    "parent_ablation_only": {
        "required": [
            "P_vs_P_no_parent_char_f1_delta > 0",
            "P_vs_P_no_parent_char_f1_95ci_lower > 0",
            "P_vs_P_no_parent_citation_recall_delta > 0",
            "P_vs_P_no_parent_citation_recall_95ci_lower > 0",
        ],
        "allowed_thesis_claim": (
            "Parent context contributes over child-only retrieval, but the "
            "complete method is not claimed to beat B4."
        ),
    },
    "failed": {
        "required": ["parent_ablation_only conditions are not met"],
        "allowed_thesis_claim": "Report failure analysis; do not package as success.",
    },
}

FORBIDDEN_MANIFEST_FIELDS = {
    "source_text",
    "question_text",
    "reference_answer",
    "answer_text",
    "evidence_text",
    "reviewer_comment",
}


@dataclass
class LoadedFormalIndex:
    chunks: list[PublicTcmQgChunk]
    tokens: list[list[str]]
    bm25: BM25Okapi
    dense: np.ndarray
    manifest: dict


def public_tcm_qg_formal_matrix() -> dict:
    return {
        "retrieval_configs": [dict(row) for row in FORMAL_RETRIEVAL_CONFIGS],
        "answer_methods": list(FORMAL_ANSWER_METHODS),
        "success_gates": {
            gate: {
                "required": list(spec["required"]),
                "allowed_thesis_claim": spec["allowed_thesis_claim"],
            }
            for gate, spec in FORMAL_SUCCESS_GATES.items()
        },
    }


def _origin_from_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("OPENAI_BASE_URL must include scheme and host")
    return f"{parsed.scheme}://{parsed.netloc}"


def read_formal_answer_model_from_env(env_path: Path) -> dict:
    env = dotenv_values(env_path) if env_path.is_file() else {}
    model_name = env.get("OPENAI_MODEL", "").strip()
    base_url = env.get("OPENAI_BASE_URL", "").strip()
    if not model_name:
        raise ValueError("OPENAI_MODEL is required for formal answer prereg")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is required for formal answer prereg")
    return {
        "model_name": model_name,
        "base_url_origin": _origin_from_base_url(base_url),
    }


def load_bge_reranker(
    *,
    model_name: str,
    revision: str,
    device: str,
    use_fp16: bool,
    batch_size: int,
    max_length: int,
):
    from FlagEmbedding import FlagReranker

    return FlagReranker(
        model_name,
        use_fp16=use_fp16,
        devices=device,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=False,
        revision=revision,
    )


def _path_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = config_path.parent / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def _ready_manifest_sha(path: Path, *, label: str) -> str:
    manifest = read_json(path, label=label)
    if manifest.get("status") != "ready":
        raise ValueError(f"{label} must be ready")
    return sha256_file(path)


def freeze_public_tcm_qg_formal_prereg(
    *,
    config_path: Path,
    env_path: Path = Path(".env"),
    output_path: Path | None = None,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    source_manifest = _path_from_config(config_path, inputs["source_manifest"])
    dataset_manifest = _path_from_config(config_path, inputs["dataset_manifest"])
    chunk_manifest = _path_from_config(config_path, inputs["chunk_manifest"])
    model_manifest_value = inputs.get("model_manifest")
    model_manifest = (
        _path_from_config(config_path, model_manifest_value)
        if model_manifest_value
        else None
    )
    dataset_path = _path_from_config(config_path, inputs["dataset_path"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"missing formal dataset: {dataset_path}")
    answer_model = read_formal_answer_model_from_env(env_path)
    matrix = public_tcm_qg_formal_matrix()
    manifest = {
        "version": config.get("version", VERSION),
        "status": "ready",
        "stage": "public_tcm_qg_formal_preregistered",
        "generated_at": utc_now(),
        "seed": int(config["seed"]),
        "retrieval_config_count": len(matrix["retrieval_configs"]),
        "retrieval_configs": matrix["retrieval_configs"],
        "answer_methods": matrix["answer_methods"],
        "answer_parameters": {
            "temperature": config["answer"]["temperature"],
            "repeats": config["answer"]["repeats"],
            "max_tokens": config["answer"]["max_tokens"],
            "test_policy": "single_frozen_run",
        },
        "answer_model": answer_model,
        "success_gates": matrix["success_gates"],
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "source_manifest_sha256": _ready_manifest_sha(
                source_manifest,
                label="formal source manifest",
            ),
            "dataset_manifest_sha256": _ready_manifest_sha(
                dataset_manifest,
                label="formal dataset manifest",
            ),
            "chunk_manifest_sha256": _ready_manifest_sha(
                chunk_manifest,
                label="formal chunk manifest",
            ),
            "dataset_sha256": sha256_file(dataset_path),
        },
        "privacy": {
            "raw_content_included": False,
            "qa_content_included": False,
            "generated_content_included": False,
            "api_key_included": False,
        },
    }
    if model_manifest is not None:
        manifest["inputs"]["model_manifest_sha256"] = _ready_manifest_sha(
            model_manifest,
            label="formal model manifest",
        )
    _assert_manifest_privacy(manifest)
    if output_path is not None:
        atomic_write_json(output_path, manifest)
    return manifest


def _assert_manifest_privacy(manifest: dict) -> None:
    serialized = str(manifest)
    leaked = [field for field in FORBIDDEN_MANIFEST_FIELDS if field in serialized]
    if leaked:
        raise ValueError(f"manifest includes forbidden fields: {', '.join(leaked)}")


def validate_formal_retrieval_inputs(
    *,
    dataset_path: Path,
    prereg_manifest_path: Path,
    index_manifest_path: Path,
) -> dict:
    prereg = read_json(prereg_manifest_path, label="formal prereg manifest")
    index_manifest = read_json(index_manifest_path, label="formal index manifest")
    if prereg.get("status") != "ready":
        raise ValueError("formal prereg manifest must be ready")
    if index_manifest.get("status") != "ready":
        raise ValueError("formal index manifest must be ready")
    expected_dataset = prereg.get("inputs", {}).get("dataset_sha256")
    actual_dataset = sha256_file(dataset_path)
    if expected_dataset != actual_dataset:
        raise ValueError("dataset sha256 mismatch")
    expected_prereg = index_manifest.get("inputs", {}).get("prereg_manifest_sha256")
    if expected_prereg is not None and expected_prereg != sha256_file(prereg_manifest_path):
        raise ValueError("prereg manifest sha256 mismatch")
    return {
        "dataset_sha256": actual_dataset,
        "prereg_manifest_sha256": sha256_file(prereg_manifest_path),
        "index_manifest_sha256": sha256_file(index_manifest_path),
    }


def _verify_index_file(index_dir: Path, file_record: dict) -> Path:
    path = index_dir / file_record["path"]
    if not path.is_file():
        raise FileNotFoundError(f"missing formal index file: {path}")
    actual = sha256_file(path)
    if actual != file_record["sha256"]:
        raise ValueError(f"formal index sha256 mismatch: {path}")
    return path


def load_public_tcm_qg_formal_index(index_dir: Path) -> LoadedFormalIndex:
    manifest = read_json(index_dir / "manifest.json", label="formal index manifest")
    files = manifest["files"]
    rows_path = _verify_index_file(index_dir, files["rows"])
    tokens_path = _verify_index_file(index_dir, files["bm25_tokens"])
    dense_path = _verify_index_file(index_dir, files["dense"])
    chunks = [
        PublicTcmQgChunk.model_validate(row)
        for row in read_jsonl(rows_path, label="formal index rows")
    ]
    token_rows = read_jsonl(tokens_path, label="formal index tokens")
    if [chunk.chunk_id for chunk in chunks] != [row["chunk_id"] for row in token_rows]:
        raise ValueError("formal index rows and tokens are not aligned")
    dense = np.load(dense_path)
    if dense.shape[0] != len(chunks):
        raise ValueError("formal dense vectors are not row-aligned")
    return LoadedFormalIndex(
        chunks=chunks,
        tokens=[row["tokens"] for row in token_rows],
        bm25=BM25Okapi([row["tokens"] for row in token_rows]),
        dense=np.asarray(dense, dtype=np.float32),
        manifest=manifest,
    )


def _top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if len(scores) == 0:
        return []
    k = min(top_k, len(scores))
    if k == len(scores):
        indexes = np.arange(len(scores))
    else:
        indexes = np.argpartition(scores, -k)[-k:]
    ordered = sorted(indexes, key=lambda index: (-float(scores[int(index)]), int(index)))
    return [int(index) for index in ordered]


def _dense_query_vector(embedder, question: str, *, batch_size: int) -> np.ndarray:
    vector = embedder.encode(
        [question],
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != 1:
        raise ValueError("query embedding must be a single row")
    return array[0]


def _rrf_scores(
    *,
    bm25_ranked: list[int],
    dense_ranked: list[int],
    rrf_k: int,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in (bm25_ranked, dense_ranked):
        for rank, index in enumerate(ranked, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def _rerank_candidates(
    *,
    question: str,
    index: LoadedFormalIndex,
    candidate_indexes: list[int],
    reranker,
    batch_size: int,
) -> list[tuple[int, float]]:
    if reranker is None:
        return [(index_value, 0.0) for index_value in candidate_indexes]
    pairs = [[question, index.chunks[index_value].text] for index_value in candidate_indexes]
    scores = reranker.compute_score(pairs, batch_size=batch_size)
    if isinstance(scores, (float, int)):
        scores = [float(scores)]
    return [
        (index_value, float(score))
        for index_value, score in zip(candidate_indexes, scores)
    ]


def retrieve_public_tcm_qg_formal(
    *,
    question: str,
    index: LoadedFormalIndex,
    config_row: dict,
    embedder,
    reranker,
    bm25_top_k: int,
    dense_top_k: int,
    rrf_k: int,
    reranker_candidate_k: int,
    final_top_k: int,
    embedding_batch_size: int,
    reranker_batch_size: int,
) -> list[dict]:
    query_tokens = tokenize_text(question)
    if not query_tokens:
        raise ValueError("question tokenization is empty")
    bm25_scores = np.asarray(index.bm25.get_scores(query_tokens), dtype=np.float32)
    dense_scores = None
    if config_row["uses_dense"]:
        query_vector = _dense_query_vector(
            embedder,
            question,
            batch_size=embedding_batch_size,
        )
        dense_scores = index.dense @ query_vector
    if config_row["retrieval_mode"] == "bm25":
        candidate_indexes = _top_indices(bm25_scores, bm25_top_k)
        base_scores = {index_value: float(bm25_scores[index_value]) for index_value in candidate_indexes}
    elif config_row["retrieval_mode"] == "dense":
        if dense_scores is None:
            raise ValueError("dense retrieval requires dense scores")
        candidate_indexes = _top_indices(dense_scores, dense_top_k)
        base_scores = {index_value: float(dense_scores[index_value]) for index_value in candidate_indexes}
    else:
        if dense_scores is None:
            raise ValueError("hybrid retrieval requires dense scores")
        bm25_ranked = _top_indices(bm25_scores, bm25_top_k)
        dense_ranked = _top_indices(dense_scores, dense_top_k)
        base_scores = _rrf_scores(
            bm25_ranked=bm25_ranked,
            dense_ranked=dense_ranked,
            rrf_k=rrf_k,
        )
        candidate_indexes = sorted(
            base_scores,
            key=lambda index_value: (-base_scores[index_value], index.chunks[index_value].chunk_id),
        )[:reranker_candidate_k]
    reranker_scores = (
        _rerank_candidates(
            question=question,
            index=index,
            candidate_indexes=candidate_indexes,
            reranker=reranker,
            batch_size=reranker_batch_size,
        )
        if config_row["uses_reranker"]
        else [(index_value, 0.0) for index_value in candidate_indexes]
    )
    reranker_lookup = dict(reranker_scores)
    ranked = sorted(
        candidate_indexes,
        key=lambda index_value: (
            -reranker_lookup.get(index_value, 0.0),
            -base_scores.get(index_value, 0.0),
            index.chunks[index_value].chunk_id,
        ),
    )[:final_top_k]
    hits = []
    for rank, row_index in enumerate(ranked, start=1):
        chunk = index.chunks[row_index]
        context_text = chunk.context_text
        context_start_index = chunk.context_start_index
        context_char_count = chunk.context_char_count
        if config_row["context_policy"] in {"chunk", "child"}:
            context_text = chunk.text
            context_start_index = chunk.start_index
            context_char_count = chunk.char_count
        hits.append(
            {
                "chunk_id": chunk.chunk_id,
                "strategy": chunk.strategy,
                "rank": rank,
                "source_doc_id": chunk.source_doc_id,
                "parent_id": chunk.parent_id,
                "text": chunk.text,
                "context_text": context_text,
                "start_index": chunk.start_index,
                "char_count": chunk.char_count,
                "context_start_index": context_start_index,
                "context_char_count": context_char_count,
                "bm25_score": float(bm25_scores[row_index]),
                "dense_score": (
                    float(dense_scores[row_index]) if dense_scores is not None else None
                ),
                "rrf_score": float(base_scores.get(row_index, 0.0)),
                "reranker_score": float(reranker_lookup.get(row_index, 0.0)),
                "score": float(reranker_lookup.get(row_index, base_scores.get(row_index, 0.0))),
            }
        )
    return hits


def _doc_recall_at_k(*, source_doc_id: str, hits: list[dict], k: int) -> float:
    return float(any(hit["source_doc_id"] == source_doc_id for hit in hits[:k]))


def _doc_mrr_at_10(*, source_doc_id: str, hits: list[dict]) -> float:
    for rank, hit in enumerate(hits[:10], start=1):
        if hit["source_doc_id"] == source_doc_id:
            return 1.0 / rank
    return 0.0


def _answer_span_hit(*, answer_start: int, answer_end: int, hits: list[dict]) -> bool:
    for hit in hits[:5]:
        start = hit["context_start_index"]
        end = start + hit["context_char_count"]
        if start <= answer_start and answer_end <= end:
            return True
    return False


def _answer_span_coverage(
    *,
    source_doc_id: str,
    answer_start: int,
    answer_end: int,
    hits: list[dict],
) -> float:
    if answer_end <= answer_start:
        return 0.0
    spans = []
    for hit in hits[:5]:
        if hit["source_doc_id"] != source_doc_id:
            continue
        start = max(answer_start, hit["context_start_index"])
        end = min(answer_end, hit["context_start_index"] + hit["context_char_count"])
        if end > start:
            spans.append((start, end))
    if not spans:
        return 0.0
    spans.sort()
    merged = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged) / (answer_end - answer_start)


def _top5_traceability_ok(hits: list[dict]) -> bool:
    top5 = hits[:5]
    return len(top5) == 5 and all(
        hit["chunk_id"] and hit["source_doc_id"] and hit["context_text"].strip()
        for hit in top5
    )


def _run_formal_config(
    *,
    config_row: dict,
    questions: list[dict],
    index: LoadedFormalIndex,
    matrix_config: dict,
    config_dir: Path,
    embedder,
    reranker,
) -> dict:
    config_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        config_dir / "run-config.json",
        {
            "version": VERSION,
            "status": "running",
            **config_row,
            "created_at": utc_now(),
        },
    )
    records = []
    errors = []
    for question in questions:
        started = time.perf_counter()
        try:
            hits = retrieve_public_tcm_qg_formal(
                question=question["question"],
                index=index,
                config_row=config_row,
                embedder=embedder,
                reranker=reranker,
                bm25_top_k=int(matrix_config["bm25_top_k"]),
                dense_top_k=int(matrix_config["dense_top_k"]),
                rrf_k=int(matrix_config["rrf_k"]),
                reranker_candidate_k=int(matrix_config["reranker_candidate_k"]),
                final_top_k=int(matrix_config["final_top_k"]),
                embedding_batch_size=int(matrix_config["embedding_batch_size"]),
                reranker_batch_size=int(matrix_config["reranker_batch_size"]),
            )
            latency_ms = (time.perf_counter() - started) * 1000
            same_doc_hits = [
                hit for hit in hits if hit["source_doc_id"] == question["source_doc_id"]
            ]
            records.append(
                {
                    "qa_id": question["qa_id"],
                    "source_doc_id": question["source_doc_id"],
                    "split": question["split"],
                    "config_id": config_row["config_id"],
                    "method_role": config_row["method_role"],
                    "hits": hits,
                    "doc_recall_at_5": _doc_recall_at_k(
                        source_doc_id=question["source_doc_id"],
                        hits=hits,
                        k=5,
                    ),
                    "doc_mrr_at_10": _doc_mrr_at_10(
                        source_doc_id=question["source_doc_id"],
                        hits=hits,
                    ),
                    "answer_span_hit_at_5": float(
                        _answer_span_hit(
                            answer_start=question["answer_start"],
                            answer_end=question["answer_end"],
                            hits=same_doc_hits,
                        )
                    ),
                    "answer_span_coverage_at_5": _answer_span_coverage(
                        source_doc_id=question["source_doc_id"],
                        answer_start=question["answer_start"],
                        answer_end=question["answer_end"],
                        hits=hits,
                    ),
                    "top5_traceability_ok": _top5_traceability_ok(hits),
                    "latency": latency_ms,
                    "latency_ms": latency_ms,
                }
            )
        except Exception as error:
            errors.append(
                {
                    "qa_id": question.get("qa_id"),
                    "config_id": config_row["config_id"],
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "recorded_at": utc_now(),
                }
            )
    write_jsonl(config_dir / "per-question.jsonl", records)
    write_jsonl(config_dir / "errors.jsonl", errors)
    metrics = {
        "status": "completed" if not errors else "failed",
        "question_count": len(questions),
        "completed_count": len(records),
        "error_count": len(errors),
        "doc_recall_at_5": mean(row["doc_recall_at_5"] for row in records),
        "doc_mrr_at_10": mean(row["doc_mrr_at_10"] for row in records),
        "answer_span_hit_at_5": mean(row["answer_span_hit_at_5"] for row in records),
        "answer_span_coverage_at_5": mean(
            row["answer_span_coverage_at_5"] for row in records
        ),
        "top5_traceability_rate": mean(row["top5_traceability_ok"] for row in records),
        "latency_ms_mean": mean(row["latency_ms"] for row in records),
    }
    atomic_write_json(config_dir / "metrics.json", metrics)
    atomic_write_json(
        config_dir / "latency.json",
        {"summary": {"latency_ms_mean": metrics["latency_ms_mean"]}},
    )
    run_config = read_json(config_dir / "run-config.json", label="formal run config")
    run_config.update(
        status=metrics["status"],
        completed_count=metrics["completed_count"],
        error_count=metrics["error_count"],
    )
    atomic_write_json(config_dir / "run-config.json", run_config)
    return {
        "config_id": config_row["config_id"],
        "method_role": config_row["method_role"],
        "strategy": config_row["strategy"],
        "context_policy": config_row["context_policy"],
        "retrieval_mode": config_row["retrieval_mode"],
        "status": metrics["status"],
        "metrics": metrics,
        "files": {
            "per_question": {
                "path": f"{config_row['config_id']}/per-question.jsonl",
                "sha256": sha256_file(config_dir / "per-question.jsonl"),
            },
            "metrics": {
                "path": f"{config_row['config_id']}/metrics.json",
                "sha256": sha256_file(config_dir / "metrics.json"),
            },
            "latency": {
                "path": f"{config_row['config_id']}/latency.json",
                "sha256": sha256_file(config_dir / "latency.json"),
            },
            "errors": {
                "path": f"{config_row['config_id']}/errors.jsonl",
                "sha256": sha256_file(config_dir / "errors.jsonl"),
            },
            "run_config": {
                "path": f"{config_row['config_id']}/run-config.json",
                "sha256": sha256_file(config_dir / "run-config.json"),
            },
        },
    }


def run_public_tcm_qg_formal_retrieval_matrix(
    *,
    split: str,
    dataset_path: Path,
    indexes_dir: Path,
    output_dir: Path,
    embedder,
    reranker=None,
    prereg_manifest_path: Path | None = None,
    index_manifest_path: Path | None = None,
    resume_dir: Path | None = None,
    bm25_top_k: int = 20,
    dense_top_k: int = 20,
    rrf_k: int = 60,
    reranker_candidate_k: int = 40,
    final_top_k: int = 10,
    embedding_batch_size: int = 4,
    reranker_batch_size: int = 2,
) -> dict:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if prereg_manifest_path is not None and index_manifest_path is not None:
        input_hashes = validate_formal_retrieval_inputs(
            dataset_path=dataset_path,
            prereg_manifest_path=prereg_manifest_path,
            index_manifest_path=index_manifest_path,
        )
    else:
        input_hashes = {
            "dataset_sha256": sha256_file(dataset_path),
            "b4_index_manifest_sha256": sha256_file(indexes_dir / "b4" / "manifest.json"),
            "child_index_manifest_sha256": sha256_file(
                indexes_dir / "child" / "manifest.json"
            ),
        }
    questions = [
        PublicTcmQgQaPair.model_validate(row).model_dump(mode="json")
        for row in read_jsonl(dataset_path, label="formal public TCM-QG dataset")
        if row.get("split") == split
    ]
    if not questions:
        raise ValueError(f"no questions for split={split}")
    matrix_id = (
        f"public_tcm_qg_formal_{split}-{compact_timestamp()}-"
        f"{input_hashes['dataset_sha256'][:8]}"
    )
    matrix_dir = resume_dir or (output_dir / matrix_id)
    matrix_dir.mkdir(parents=True, exist_ok=resume_dir is not None)
    matrix = public_tcm_qg_formal_matrix()["retrieval_configs"]
    matrix_config = {
        "version": VERSION,
        "status": "running",
        "split": split,
        "matrix_id": matrix_dir.name,
        "created_at": utc_now(),
        "matrix": matrix,
        "input_hashes": input_hashes,
        "bm25_top_k": bm25_top_k,
        "dense_top_k": dense_top_k,
        "rrf_k": rrf_k,
        "reranker_candidate_k": reranker_candidate_k,
        "final_top_k": final_top_k,
        "embedding_batch_size": embedding_batch_size,
        "reranker_batch_size": reranker_batch_size,
    }
    atomic_write_json(matrix_dir / "matrix-config.json", matrix_config)
    loaded_indexes = {
        "b4": load_public_tcm_qg_formal_index(indexes_dir / "b4"),
        "child": load_public_tcm_qg_formal_index(indexes_dir / "child"),
    }
    summaries = []
    for config_row in matrix:
        summaries.append(
            _run_formal_config(
                config_row=config_row,
                questions=questions,
                index=loaded_indexes[config_row["strategy"]],
                matrix_config=matrix_config,
                config_dir=matrix_dir / config_row["config_id"],
                embedder=embedder,
                reranker=reranker,
            )
        )
    status = (
        "completed"
        if all(summary["status"] == "completed" for summary in summaries)
        else "failed"
    )
    summary = {
        "version": VERSION,
        "status": status,
        "stage": "public_tcm_qg_formal_retrieval_completed",
        "split": split,
        "matrix_id": matrix_dir.name,
        "question_count": len(questions),
        "config_count": len(summaries),
        "completed_config_count": sum(
            summary["status"] == "completed" for summary in summaries
        ),
        "failed_config_count": sum(
            summary["status"] != "completed" for summary in summaries
        ),
        "input_hashes": input_hashes,
        "configs": summaries,
    }
    atomic_write_json(matrix_dir / "matrix-summary.json", summary)
    matrix_config["status"] = status
    atomic_write_json(matrix_dir / "matrix-config.json", matrix_config)
    return {**summary, "matrix_dir": matrix_dir.as_posix()}


def summarize_public_tcm_qg_formal_retrieval_test(*, run_dir: Path) -> dict:
    summary = read_json(run_dir / "matrix-summary.json", label="formal retrieval summary")
    if summary.get("status") != "completed" or summary.get("split") != "test":
        raise ValueError("only completed formal test retrieval can be summarized")
    config_lookup = {row["config_id"]: row for row in summary["configs"]}
    report = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_formal_retrieval_report",
        "generated_at": utc_now(),
        "retrieval_test_completed": True,
        "retrieval_report_written": True,
        "split": "test",
        "question_count": summary["question_count"],
        "config_count": summary["config_count"],
        "by_config": {
            config_id: {
                "method_role": row["method_role"],
                "doc_recall_at_5": row["metrics"]["doc_recall_at_5"],
                "doc_mrr_at_10": row["metrics"]["doc_mrr_at_10"],
                "answer_span_hit_at_5": row["metrics"]["answer_span_hit_at_5"],
                "answer_span_coverage_at_5": row["metrics"][
                    "answer_span_coverage_at_5"
                ],
                "top5_traceability_rate": row["metrics"]["top5_traceability_rate"],
            }
            for config_id, row in sorted(config_lookup.items())
        },
        "inputs": {
            "matrix_summary_sha256": sha256_file(run_dir / "matrix-summary.json"),
        },
    }
    atomic_write_json(run_dir / "retrieval-report.json", report)
    return {**report, "run_dir": run_dir.as_posix()}


def freeze_public_tcm_qg_formal_retrieval_runs(
    *,
    output_path: Path,
    prereg_manifest_path: Path,
    index_manifest_path: Path,
    dev_run_dir: Path | None,
    test_run_dir: Path,
) -> dict:
    report_path = test_run_dir / "retrieval-report.json"
    if not report_path.is_file():
        summarize_public_tcm_qg_formal_retrieval_test(run_dir=test_run_dir)
    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_formal_retrieval_runs_frozen",
        "generated_at": utc_now(),
        "retrieval_test_completed": True,
        "retrieval_report_written": True,
        "inputs": {
            "prereg_manifest_sha256": sha256_file(prereg_manifest_path),
            "index_manifest_sha256": sha256_file(index_manifest_path),
            "test_matrix_summary_sha256": sha256_file(test_run_dir / "matrix-summary.json"),
            "test_retrieval_report_sha256": sha256_file(report_path),
        },
        "privacy": {
            "raw_content_included": False,
            "qa_content_included": False,
            "retrieval_rows_committed": False,
        },
    }
    if dev_run_dir is not None:
        manifest["inputs"]["dev_matrix_summary_sha256"] = sha256_file(
            dev_run_dir / "matrix-summary.json"
        )
    _assert_manifest_privacy(manifest)
    atomic_write_json(output_path, manifest)
    return manifest
