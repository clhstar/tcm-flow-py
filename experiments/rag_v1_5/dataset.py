import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from experiments.rag_v1_5.metrics import evaluate_rankings
from experiments.rag_v1_5.schema import (
    EvidenceUnit,
    PilotQuestion,
    RetrievalHit,
)


DatasetProfile = Literal["auto", "smoke", "generic"]
SmokeRetriever = Callable[[PilotQuestion], list[RetrievalHit]]
SMOKE_REVIEW_FIELDS = (
    "question_id",
    "gold_clause_ids",
    "top5_chunk_ids",
    "top5_clause_ids",
    "hit_at_5",
    "parent_recovery_ok",
    "manual_comment",
    "reviewer",
    "reviewed_at",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_jsonl(path: Path, model_type):
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(model_type.model_validate_json(line))
        except Exception as error:
            raise ValueError(
                f"{path}:{line_number} 不是合法记录: {error}"
            ) from error
    return records


def load_dataset(dataset_path: Path) -> list[PilotQuestion]:
    questions = _read_jsonl(dataset_path, PilotQuestion)
    question_ids = [question.question_id for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("问题集存在重复 question_id")
    return questions


def _load_evidence(evidence_path: Path) -> dict[str, EvidenceUnit]:
    evidence_units = _read_jsonl(evidence_path, EvidenceUnit)
    evidence_by_id = {
        evidence.evidence_id: evidence for evidence in evidence_units
    }
    if len(evidence_by_id) != len(evidence_units):
        raise ValueError("Evidence Tree 存在重复 evidence_id")
    return evidence_by_id


def _resolve_profile(
    dataset_path: Path,
    profile: DatasetProfile,
) -> DatasetProfile:
    if profile != "auto":
        return profile
    if dataset_path.stem.startswith("smoke-10"):
        return "smoke"
    return "generic"


def validate_dataset(
    *,
    dataset_path: Path,
    evidence_path: Path,
    profile: DatasetProfile = "auto",
) -> dict:
    questions = load_dataset(dataset_path)
    evidence_by_id = _load_evidence(evidence_path)
    resolved_profile = _resolve_profile(dataset_path, profile)

    answerable_count = sum(question.answerable for question in questions)
    unanswerable_count = len(questions) - answerable_count
    if resolved_profile == "smoke":
        if len(questions) != 10:
            raise ValueError("Smoke 数据集必须恰好包含 10 条问题")
        if answerable_count != 9 or unanswerable_count != 1:
            raise ValueError(
                "Smoke 数据集必须包含 9 条可回答问题和 1 条无答案问题"
            )

    clause_ids = {
        evidence.clause_id for evidence in evidence_by_id.values()
    }
    for question in questions:
        if not question.answerable:
            continue
        missing_evidence = [
            evidence_id
            for evidence_id in question.gold_evidence_ids
            if evidence_id not in evidence_by_id
        ]
        if missing_evidence:
            raise ValueError(
                f"{question.question_id} 缺少 gold Evidence: "
                f"{missing_evidence}"
            )
        missing_clauses = [
            clause_id
            for clause_id in question.gold_clause_ids
            if clause_id not in clause_ids
        ]
        if missing_clauses:
            raise ValueError(
                f"{question.question_id} 缺少 gold clause: "
                f"{missing_clauses}"
            )
        gold_texts = [
            evidence_by_id[evidence_id].normalized_text
            for evidence_id in question.gold_evidence_ids
        ]
        for support_span in question.support_spans:
            if not any(support_span in text for text in gold_texts):
                raise ValueError(
                    f"{question.question_id} support span "
                    f"不在 gold Evidence 中: {support_span}"
                )
        leaked_ids = [
            gold_id
            for gold_id in (
                question.gold_evidence_ids + question.gold_clause_ids
            )
            if gold_id in question.question
        ]
        if leaked_ids:
            raise ValueError(
                f"{question.question_id} 问题文本泄漏 gold ID: "
                f"{leaked_ids}"
            )

    return {
        "profile": resolved_profile,
        "question_count": len(questions),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "approved_count": sum(
            question.review_status == "approved"
            for question in questions
        ),
        "dataset_sha256": _sha256_file(dataset_path),
        "evidence_sha256": _sha256_file(evidence_path),
    }


def _flatten_clause_ids(hits: list[RetrievalHit]) -> list[str]:
    clause_ids = []
    for hit in hits:
        for clause_id in hit.clause_ids:
            if clause_id not in clause_ids:
                clause_ids.append(clause_id)
    return clause_ids


def _parent_recovery_ok(
    question: PilotQuestion,
    hits: list[RetrievalHit],
) -> bool:
    gold_clause_ids = set(question.gold_clause_ids)
    matching_hits = [
        hit
        for hit in hits
        if gold_clause_ids & set(hit.clause_ids)
    ]
    return bool(matching_hits) and all(
        hit.retrieval_parent_id in gold_clause_ids
        and hit.retrieval_parent_id in set(hit.clause_ids)
        and bool(hit.context_text.strip())
        for hit in matching_hits
    )


def _latency_summary(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _load_existing_manual_review(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != SMOKE_REVIEW_FIELDS:
            raise ValueError("已有 smoke-review.csv 字段不符合契约")
        return {row["question_id"]: row for row in reader}


def _write_review_csv(path: Path, rows: list[dict]) -> None:
    existing = _load_existing_manual_review(path)
    for row in rows:
        old_row = existing.get(row["question_id"], {})
        for field in ("manual_comment", "reviewer", "reviewed_at"):
            row[field] = old_row.get(field, "")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=SMOKE_REVIEW_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def run_smoke_dataset(
    *,
    questions: list[PilotQuestion],
    strategy: str,
    mode: str,
    output_dir: Path,
    review_csv_path: Path,
    retriever: SmokeRetriever,
    provenance: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings: dict[str, list[RetrievalHit]] = {}
    per_question = []
    review_rows = []
    errors = []
    total_latencies = []
    answerable_hits = []
    parent_recoveries = []

    for question in questions:
        started = time.perf_counter()
        try:
            hits = [
                RetrievalHit.model_validate(hit)
                for hit in retriever(question)
            ]
        except Exception as error:
            hits = []
            errors.append(
                {
                    "question_id": question.question_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_latencies.append(elapsed_ms)
        top5 = hits[:5]
        rankings[question.question_id] = hits

        if question.answerable:
            hit_at_5 = bool(
                set(question.gold_clause_ids)
                & set(_flatten_clause_ids(top5))
            )
            parent_recovery_ok = _parent_recovery_ok(question, top5)
            answerable_hits.append(float(hit_at_5))
            parent_recoveries.append(float(parent_recovery_ok))
        else:
            hit_at_5 = None
            parent_recovery_ok = None

        per_question.append(
            {
                "question": question.model_dump(mode="json"),
                "hits": [
                    hit.model_dump(mode="json") for hit in hits
                ],
                "hit_at_5": hit_at_5,
                "parent_recovery_ok": parent_recovery_ok,
                "total_ms": elapsed_ms,
                "returned_context_chars": sum(
                    len(hit.context_text) for hit in top5
                ),
            }
        )
        review_rows.append(
            {
                "question_id": question.question_id,
                "gold_clause_ids": "|".join(question.gold_clause_ids),
                "top5_chunk_ids": "|".join(
                    hit.chunk_id for hit in top5
                ),
                "top5_clause_ids": "|".join(
                    _flatten_clause_ids(top5)
                ),
                "hit_at_5": (
                    str(hit_at_5).lower()
                    if hit_at_5 is not None
                    else ""
                ),
                "parent_recovery_ok": (
                    str(parent_recovery_ok).lower()
                    if parent_recovery_ok is not None
                    else ""
                ),
                "manual_comment": "",
                "reviewer": "",
                "reviewed_at": "",
            }
        )

    metrics = evaluate_rankings(questions, rankings)
    answerable_hit_at_5 = (
        statistics.fmean(answerable_hits) if answerable_hits else 0.0
    )
    answerable_parent_recovery_rate = (
        statistics.fmean(parent_recoveries)
        if parent_recoveries
        else 0.0
    )
    metrics["smoke_answerable_hit_at_5"] = answerable_hit_at_5
    metrics["smoke_answerable_parent_recovery_rate"] = (
        answerable_parent_recovery_rate
    )
    run_config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "mode": mode,
        "question_count": len(questions),
        **provenance,
    }
    latency = {
        "total_ms": _latency_summary(total_latencies),
        "records": [
            {
                "question_id": record["question"]["question_id"],
                "total_ms": record["total_ms"],
                "returned_context_chars": record[
                    "returned_context_chars"
                ],
            }
            for record in per_question
        ],
    }

    _write_jsonl(output_dir / "per-question.jsonl", per_question)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "latency.json", latency)
    _write_jsonl(output_dir / "errors.jsonl", errors)
    _write_json(output_dir / "run-config.json", run_config)
    _write_review_csv(review_csv_path, review_rows)

    return {
        "question_count": len(questions),
        "error_count": len(errors),
        "answerable_hit_at_5": answerable_hit_at_5,
        "answerable_parent_recovery_rate": (
            answerable_parent_recovery_rate
        ),
        "output_dir": output_dir.as_posix(),
        "review_csv": review_csv_path.as_posix(),
    }
