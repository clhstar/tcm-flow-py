import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
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
from experiments.rag_v1_6.schema import (
    PublicTcmQgChunk,
    PublicTcmQgQaPair,
    PublicTcmQgRetrievalHit,
)


PUBLIC_TCM_QG_MATRIX = [
    {
        "config_id": "b4-public-bm25-rerank",
        "method": "B4",
        "strategy": "b4",
        "context_policy": "chunk",
        "mode": "bm25_overlap_rerank",
    },
    {
        "config_id": "p-public-bm25-rerank",
        "method": "P",
        "strategy": "child",
        "context_policy": "parent",
        "mode": "bm25_overlap_rerank",
    },
    {
        "config_id": "p-public-no-parent",
        "method": "P-no-parent",
        "strategy": "child",
        "context_policy": "child",
        "mode": "bm25_overlap_rerank",
    },
]


@dataclass
class LoadedPublicIndex:
    chunks: list[PublicTcmQgChunk]
    tokens: list[list[str]]
    bm25: BM25Okapi
    manifest: dict


def public_tcm_qg_matrix() -> list[dict]:
    return [dict(row) for row in PUBLIC_TCM_QG_MATRIX]


def _verify_file(index_dir: Path, file_record: dict) -> Path:
    path = index_dir / file_record["path"]
    if not path.is_file():
        raise FileNotFoundError(f"missing index file: {path}")
    actual = sha256_file(path)
    if actual != file_record["sha256"]:
        raise ValueError(f"index file sha256 mismatch: {path}")
    return path


def load_public_tcm_qg_index(index_dir: Path) -> LoadedPublicIndex:
    manifest_path = index_dir / "manifest.json"
    manifest = read_json(manifest_path, label="public index manifest")
    files = manifest["files"]
    rows_path = _verify_file(index_dir, files["rows"])
    tokens_path = _verify_file(index_dir, files["bm25_tokens"])
    chunks = [
        PublicTcmQgChunk.model_validate(row)
        for row in read_jsonl(rows_path, label="public index rows")
    ]
    token_rows = read_jsonl(tokens_path, label="public index tokens")
    if [chunk.chunk_id for chunk in chunks] != [
        row["chunk_id"] for row in token_rows
    ]:
        raise ValueError("index rows and tokens are not aligned")
    tokens = [row["tokens"] for row in token_rows]
    return LoadedPublicIndex(
        chunks=chunks,
        tokens=tokens,
        bm25=BM25Okapi(tokens),
        manifest=manifest,
    )


def _overlap_score(query_tokens: set[str], chunk_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(chunk_tokens)) / len(query_tokens)


def retrieve_public_tcm_qg(
    *,
    question: str,
    index: LoadedPublicIndex,
    config: dict,
    context_policy: str,
) -> list[dict]:
    if context_policy not in {"chunk", "parent", "child"}:
        raise ValueError(f"unknown context_policy: {context_policy}")
    query_tokens = tokenize_text(question)
    if not query_tokens:
        raise ValueError("question tokenization is empty")
    bm25_scores = np.asarray(index.bm25.get_scores(query_tokens), dtype=float)
    candidate_k = min(int(config["retrieval"]["bm25_top_k"]), len(index.chunks))
    final_top_k = min(int(config["retrieval"]["final_top_k"]), len(index.chunks))
    if candidate_k == len(index.chunks):
        candidate_indexes = np.arange(len(index.chunks))
    else:
        candidate_indexes = np.argpartition(bm25_scores, -candidate_k)[
            -candidate_k:
        ]
    query_token_set = set(query_tokens)
    scored = []
    for row_index in candidate_indexes:
        chunk = index.chunks[int(row_index)]
        overlap = _overlap_score(query_token_set, index.tokens[int(row_index)])
        score = float(bm25_scores[int(row_index)]) + overlap
        scored.append((score, overlap, chunk.chunk_id, int(row_index)))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))

    hits = []
    for rank, (score, overlap, _chunk_id, row_index) in enumerate(
        scored[:final_top_k],
        start=1,
    ):
        chunk = index.chunks[row_index]
        context_text = chunk.context_text
        context_start_index = chunk.context_start_index
        context_char_count = chunk.context_char_count
        if context_policy in {"chunk", "child"}:
            context_text = chunk.text
            context_start_index = chunk.start_index
            context_char_count = chunk.char_count
        hit = PublicTcmQgRetrievalHit(
            chunk_id=chunk.chunk_id,
            strategy=chunk.strategy,
            rank=rank,
            source_doc_id=chunk.source_doc_id,
            parent_id=chunk.parent_id,
            text=chunk.text,
            context_text=context_text,
            start_index=chunk.start_index,
            char_count=chunk.char_count,
            context_start_index=context_start_index,
            context_char_count=context_char_count,
            bm25_score=float(bm25_scores[row_index]),
            overlap_score=overlap,
            score=score,
        )
        hits.append(hit.model_dump(mode="json"))
    return hits


def doc_recall_at_k(*, source_doc_id: str, hits: list[dict], k: int) -> float:
    return float(any(hit["source_doc_id"] == source_doc_id for hit in hits[:k]))


def doc_mrr_at_10(*, source_doc_id: str, hits: list[dict]) -> float:
    for rank, hit in enumerate(hits[:10], start=1):
        if hit["source_doc_id"] == source_doc_id:
            return 1.0 / rank
    return 0.0


def answer_span_hit(*, answer_start: int, answer_end: int, hits: list[dict]) -> bool:
    for hit in hits[:5]:
        start = hit["context_start_index"]
        end = start + hit["context_char_count"]
        if start <= answer_start and answer_end <= end:
            return True
    return False


def answer_span_coverage(
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
    covered = sum(end - start for start, end in merged)
    return covered / (answer_end - answer_start)


def _top5_traceability_ok(hits: list[dict]) -> bool:
    top5 = hits[:5]
    return len(top5) == 5 and all(
        hit["chunk_id"]
        and hit["source_doc_id"]
        and hit["context_text"].strip()
        for hit in top5
    )


def _metrics_for_records(records: list[dict], errors: list[dict]) -> dict:
    return {
        "status": "completed" if not errors else "failed",
        "question_count": len(records),
        "completed_count": len(records),
        "error_count": len(errors),
        "doc_recall_at_5": mean(row["doc_recall_at_5"] for row in records),
        "doc_mrr_at_10": mean(row["doc_mrr_at_10"] for row in records),
        "answer_span_hit_at_5": mean(
            row["answer_span_hit_at_5"] for row in records
        ),
        "answer_span_coverage_at_5": mean(
            row["answer_span_coverage_at_5"] for row in records
        ),
        "top5_traceability_rate": mean(
            row["top5_traceability_ok"] for row in records
        ),
        "latency_ms_mean": mean(row["latency_ms"] for row in records),
    }


def _run_one_config(
    *,
    config_row: dict,
    questions: list[dict],
    index: LoadedPublicIndex,
    config: dict,
    config_dir: Path,
) -> dict:
    config_dir.mkdir(parents=True, exist_ok=True)
    per_question_path = config_dir / "per-question.jsonl"
    errors_path = config_dir / "errors.jsonl"
    run_config = {
        "version": VERSION,
        "status": "running",
        **config_row,
        "backend": config["retrieval"]["backend"],
        "created_at": utc_now(),
    }
    atomic_write_json(config_dir / "run-config.json", run_config)
    records = []
    errors = []
    for question in questions:
        started = time.perf_counter()
        try:
            hits = retrieve_public_tcm_qg(
                question=question["question"],
                index=index,
                config=config,
                context_policy=config_row["context_policy"],
            )
            latency_ms = (time.perf_counter() - started) * 1000
            record = {
                "config_id": config_row["config_id"],
                "method": config_row["method"],
                "qa_id": question["qa_id"],
                "source_doc_id": question["source_doc_id"],
                "split": question["split"],
                "answer_start": question["answer_start"],
                "answer_end": question["answer_end"],
                "hits": hits,
                "doc_recall_at_5": doc_recall_at_k(
                    source_doc_id=question["source_doc_id"],
                    hits=hits,
                    k=5,
                ),
                "doc_mrr_at_10": doc_mrr_at_10(
                    source_doc_id=question["source_doc_id"],
                    hits=hits,
                ),
                "answer_span_hit_at_5": float(
                    answer_span_hit(
                        answer_start=question["answer_start"],
                        answer_end=question["answer_end"],
                        hits=[
                            hit
                            for hit in hits
                            if hit["source_doc_id"] == question["source_doc_id"]
                        ],
                    )
                ),
                "answer_span_coverage_at_5": answer_span_coverage(
                    source_doc_id=question["source_doc_id"],
                    answer_start=question["answer_start"],
                    answer_end=question["answer_end"],
                    hits=hits,
                ),
                "top5_traceability_ok": _top5_traceability_ok(hits),
                "latency_ms": latency_ms,
            }
            records.append(record)
        except Exception as error:
            errors.append(
                {
                    "config_id": config_row["config_id"],
                    "qa_id": question.get("qa_id"),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "recorded_at": utc_now(),
                }
            )
    write_jsonl(per_question_path, records)
    write_jsonl(errors_path, errors)
    metrics = _metrics_for_records(records, errors)
    atomic_write_json(config_dir / "metrics.json", metrics)
    latency = {
        "summary": {
            "latency_ms_mean": metrics["latency_ms_mean"],
            "returned_context_chars_mean": mean(
                sum(len(hit["context_text"]) for hit in record["hits"][:5])
                for record in records
            ),
        }
    }
    atomic_write_json(config_dir / "latency.json", latency)
    run_config.update(
        status=metrics["status"],
        completed_count=metrics["completed_count"],
        error_count=metrics["error_count"],
    )
    atomic_write_json(config_dir / "run-config.json", run_config)
    return {
        "config_id": config_row["config_id"],
        "method": config_row["method"],
        "strategy": config_row["strategy"],
        "context_policy": config_row["context_policy"],
        "status": metrics["status"],
        "metrics": {
            key: metrics[key]
            for key in (
                "doc_recall_at_5",
                "doc_mrr_at_10",
                "answer_span_hit_at_5",
                "answer_span_coverage_at_5",
                "top5_traceability_rate",
            )
        },
        "latency": latency["summary"],
        "files": {
            "per_question": {
                "path": f"{config_row['config_id']}/per-question.jsonl",
                "sha256": sha256_file(per_question_path),
            },
            "metrics": {
                "path": f"{config_row['config_id']}/metrics.json",
                "sha256": sha256_file(config_dir / "metrics.json"),
            },
            "run_config": {
                "path": f"{config_row['config_id']}/run-config.json",
                "sha256": sha256_file(config_dir / "run-config.json"),
            },
        },
    }


def run_public_tcm_qg_retrieval_matrix(
    *,
    split: str,
    dataset_path: Path,
    config_path: Path,
    indexes_dir: Path,
    output_dir: Path,
    resume_dir: Path | None = None,
) -> dict:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    all_questions = [
        PublicTcmQgQaPair.model_validate(row).model_dump(mode="json")
        for row in read_jsonl(dataset_path, label="public TCM-QG dataset")
        if row.get("split") == split
    ]
    if not all_questions:
        raise ValueError(f"no questions for split={split}")
    matrix = public_tcm_qg_matrix()
    if resume_dir is None:
        run_id = (
            f"public_tcm_qg_{split}-{compact_timestamp()}-"
            f"{sha256_file(dataset_path)[:8]}"
        )
        matrix_dir = output_dir / run_id
        matrix_dir.mkdir(parents=True, exist_ok=False)
    else:
        matrix_dir = resume_dir
        matrix_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "dataset_sha256": sha256_file(dataset_path),
        "config_sha256": sha256_file(config_path),
        "b4_index_manifest_sha256": sha256_file(indexes_dir / "b4" / "manifest.json"),
        "child_index_manifest_sha256": sha256_file(
            indexes_dir / "child" / "manifest.json"
        ),
    }
    matrix_config = {
        "version": VERSION,
        "status": "running",
        "split": split,
        "matrix_id": matrix_dir.name,
        "created_at": utc_now(),
        "matrix": matrix,
        "backend": config["retrieval"]["backend"],
        "input_hashes": input_hashes,
    }
    atomic_write_json(matrix_dir / "matrix-config.json", matrix_config)
    loaded_indexes = {
        "b4": load_public_tcm_qg_index(indexes_dir / "b4"),
        "child": load_public_tcm_qg_index(indexes_dir / "child"),
    }
    summaries = []
    for config_row in matrix:
        summaries.append(
            _run_one_config(
                config_row=config_row,
                questions=all_questions,
                index=loaded_indexes[config_row["strategy"]],
                config=config,
                config_dir=matrix_dir / config_row["config_id"],
            )
        )
    status = (
        "completed"
        if all(summary["status"] == "completed" for summary in summaries)
        else "failed"
    )
    matrix_summary = {
        "version": VERSION,
        "status": status,
        "split": split,
        "matrix_id": matrix_dir.name,
        "question_count": len(all_questions),
        "config_count": len(summaries),
        "completed_config_count": sum(
            summary["status"] == "completed" for summary in summaries
        ),
        "failed_config_count": sum(
            summary["status"] != "completed" for summary in summaries
        ),
        "backend": config["retrieval"]["backend"],
        "input_hashes": input_hashes,
        "configs": summaries,
    }
    atomic_write_json(matrix_dir / "matrix-summary.json", matrix_summary)
    matrix_config["status"] = status
    atomic_write_json(matrix_dir / "matrix-config.json", matrix_config)
    return {**matrix_summary, "matrix_dir": matrix_dir.as_posix()}


def freeze_public_tcm_qg_runs(
    *,
    metrics: dict,
    output_path: Path | None = None,
    source_manifest_path: Path | None = None,
    dataset_manifest_path: Path | None = None,
    retrieval_run_dir: Path | None = None,
    answer_run_dir: Path | None = None,
) -> dict:
    manifest = {
        "version": VERSION,
        "status": metrics.get("status", "ready"),
        "stage": "public_tcm_qg_runs_frozen",
        "generated_at": utc_now(),
        "paper_experiment_scope": "public_tcm_qg_only",
        "formal_400_included": False,
        "answer_mode": metrics.get("answer_mode", "extractive_oracle_proxy"),
        "success_gate": metrics.get("success_gate", {}),
        "by_method": metrics.get("by_method", {}),
        "paired_comparisons": metrics.get("paired_comparisons", {}),
        "inputs": {},
        "privacy": {
            "raw_content_included": False,
            "qa_content_included": False,
            "generated_content_included": False,
            "full_results_committed": False,
        },
    }
    for key, path in (
        ("source_manifest_sha256", source_manifest_path),
        ("dataset_manifest_sha256", dataset_manifest_path),
        ("retrieval_matrix_sha256", retrieval_run_dir / "matrix-summary.json"
        if retrieval_run_dir is not None
        else None),
        ("answer_matrix_sha256", answer_run_dir / "matrix-summary.json"
        if answer_run_dir is not None
        else None),
        ("automatic_metrics_sha256", answer_run_dir / "automatic-metrics.json"
        if answer_run_dir is not None
        else None),
        ("paired_bootstrap_sha256", answer_run_dir / "paired-bootstrap.json"
        if answer_run_dir is not None
        else None),
    ):
        if path is not None and path.is_file():
            manifest["inputs"][key] = sha256_file(path)
    if output_path is not None:
        atomic_write_json(output_path, manifest)
    return manifest
