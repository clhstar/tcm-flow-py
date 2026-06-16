import time
from pathlib import Path

import yaml

from experiments.rag_v1_6.common import (
    VERSION,
    atomic_write_json,
    compact_timestamp,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_jsonl,
)
from experiments.rag_v1_6.schema import PublicTcmQgAnswerRecord, PublicTcmQgQaPair


ABSTAIN_ANSWER = "给定公开文档证据中未找到可靠答案。"
METHOD_TO_CONFIG = {
    "B4": "b4-public-bm25-rerank",
    "P": "p-public-bm25-rerank",
    "P-no-parent": "p-public-no-parent",
}
PROXY_MODEL_NAME = "extractive_oracle_proxy_v1.6"


def build_public_tcm_qg_prompt(
    *,
    question: str,
    method: str,
    evidence: list[dict],
) -> str:
    evidence_text = "\n".join(
        f"{item['label']} (doc={item['source_doc_id']}): {item['text']}"
        for item in evidence
    )
    if method == "B0":
        evidence_text = "No evidence."
    return (
        "You are a TCM literature QA system. Answer only from the provided "
        "public document evidence. If evidence is insufficient, return the "
        f"fixed abstention: {ABSTAIN_ANSWER}\n"
        "Return JSON with fields: answer, abstain, citations.\n\n"
        f"Question: {question}\n\nEvidence:\n{evidence_text}"
    )


def _load_retrieval_records(
    *,
    matrix_dir: Path,
    config_id: str,
) -> dict[str, dict]:
    rows = read_jsonl(
        matrix_dir / config_id / "per-question.jsonl",
        label=f"{config_id} retrieval rows",
    )
    records = {}
    for row in rows:
        qa_id = row["qa_id"]
        if qa_id in records:
            raise ValueError(f"duplicate retrieval qa_id: {qa_id}")
        records[qa_id] = row
    return records


def _evidence_from_retrieval(record: dict, *, top_k: int) -> list[dict]:
    items = []
    seen = set()
    for hit in record["hits"][:top_k]:
        dedupe_key = hit["parent_id"] if record["method"] == "P" else hit["chunk_id"]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        items.append(
            {
                "label": f"E{len(items) + 1}",
                "source_doc_id": hit["source_doc_id"],
                "chunk_id": hit["chunk_id"],
                "parent_id": hit["parent_id"],
                "text": hit["context_text"],
                "context_start_index": hit["context_start_index"],
                "context_char_count": hit["context_char_count"],
            }
        )
    return items


def _proxy_answer(
    *,
    method: str,
    reference_answer: str,
    evidence: list[dict],
) -> tuple[str, bool, list[str], bool]:
    if method == "B0":
        return ABSTAIN_ANSWER, True, [], False
    supporting = [
        item["label"] for item in evidence if reference_answer in item["text"]
    ]
    if not supporting:
        return ABSTAIN_ANSWER, True, [], False
    return reference_answer, False, supporting[:1], True


def load_public_answer_inputs(
    *,
    split: str,
    dataset_path: Path,
    retrieval_matrix_dir: Path,
    top_k: int,
) -> dict:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    matrix_config = read_json(
        retrieval_matrix_dir / "matrix-config.json",
        label="retrieval matrix config",
    )
    matrix_summary = read_json(
        retrieval_matrix_dir / "matrix-summary.json",
        label="retrieval matrix summary",
    )
    if matrix_config.get("status") != "completed" or matrix_summary.get(
        "status"
    ) != "completed":
        raise ValueError("retrieval matrix must be completed")
    if matrix_config.get("split") != split:
        raise ValueError("retrieval matrix split mismatch")
    questions = {
        row["qa_id"]: PublicTcmQgQaPair.model_validate(row).model_dump(mode="json")
        for row in read_jsonl(dataset_path, label="public TCM-QG dataset")
        if row.get("split") == split
    }
    if not questions:
        raise ValueError(f"no questions for split={split}")
    retrieval_by_method = {}
    raw_by_method = {}
    for method, config_id in METHOD_TO_CONFIG.items():
        records = _load_retrieval_records(
            matrix_dir=retrieval_matrix_dir,
            config_id=config_id,
        )
        if set(records) != set(questions):
            raise ValueError(f"{config_id} question set mismatch")
        raw_by_method[method] = records
        retrieval_by_method[method] = {
            qa_id: {
                "evidence": _evidence_from_retrieval(record, top_k=top_k),
                "hits": record["hits"],
            }
            for qa_id, record in records.items()
        }
    for qa_id in questions:
        p_ids = [hit["chunk_id"] for hit in raw_by_method["P"][qa_id]["hits"]]
        child_ids = [
            hit["chunk_id"] for hit in raw_by_method["P-no-parent"][qa_id]["hits"]
        ]
        if p_ids != child_ids:
            raise ValueError("P-no-parent must reuse the same child ranking as P")
    return {
        "split": split,
        "questions": questions,
        "retrieval": retrieval_by_method,
        "input_hashes": {
            "dataset_sha256": sha256_file(dataset_path),
            "retrieval_matrix_config_sha256": sha256_file(
                retrieval_matrix_dir / "matrix-config.json"
            ),
            "retrieval_matrix_summary_sha256": sha256_file(
                retrieval_matrix_dir / "matrix-summary.json"
            ),
        },
    }


def run_public_tcm_qg_answer_matrix(
    *,
    split: str,
    dataset_path: Path,
    retrieval_matrix_dir: Path,
    config_path: Path,
    output_dir: Path,
    resume_dir: Path | None = None,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repeats = int(config["answer"]["repeats"])
    methods = list(config["answer"]["methods"])
    if repeats < 1:
        raise ValueError("answer.repeats must be >= 1")
    loaded = load_public_answer_inputs(
        split=split,
        dataset_path=dataset_path,
        retrieval_matrix_dir=retrieval_matrix_dir,
        top_k=int(config["retrieval"]["answer_context_top_k"]),
    )
    if resume_dir is None:
        run_id = (
            f"public_tcm_qg_answer_{split}-{compact_timestamp()}-"
            f"{loaded['input_hashes']['dataset_sha256'][:8]}"
        )
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "version": VERSION,
        "status": "running",
        "split": split,
        "run_id": run_dir.name,
        "answer_mode": config["answer"]["mode"],
        "methods": methods,
        "repeats": repeats,
        "input_hashes": {
            **loaded["input_hashes"],
            "config_sha256": sha256_file(config_path),
        },
        "model_name": PROXY_MODEL_NAME,
        "created_at": utc_now(),
    }
    atomic_write_json(run_dir / "run-config.json", run_config)
    records = []
    errors = []
    for qa_id in sorted(loaded["questions"]):
        question = loaded["questions"][qa_id]
        for method in methods:
            evidence = (
                []
                if method == "B0"
                else loaded["retrieval"][method][qa_id]["evidence"]
            )
            for repeat_index in range(repeats):
                started = time.perf_counter()
                try:
                    answer, abstain, citations, supported = _proxy_answer(
                        method=method,
                        reference_answer=question["answer"],
                        evidence=evidence,
                    )
                    prompt = build_public_tcm_qg_prompt(
                        question=question["question"],
                        method=method,
                        evidence=evidence,
                    )
                    record = PublicTcmQgAnswerRecord(
                        qa_id=qa_id,
                        source_doc_id=question["source_doc_id"],
                        split=split,
                        method=method,
                        repeat_index=repeat_index,
                        answer=answer,
                        abstain=abstain,
                        citations=citations,
                        retrieval_supported=supported,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        input_tokens=len(prompt),
                        output_tokens=len(answer),
                        model_name=PROXY_MODEL_NAME,
                    )
                    records.append(record.model_dump(mode="json"))
                except Exception as error:
                    errors.append(
                        {
                            "qa_id": qa_id,
                            "method": method,
                            "repeat_index": repeat_index,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "recorded_at": utc_now(),
                        }
                    )
    write_jsonl(run_dir / "per-answer.jsonl", records)
    write_jsonl(run_dir / "errors.jsonl", errors)
    expected_runs = len(loaded["questions"]) * len(methods) * repeats
    summary = {
        "version": VERSION,
        "status": "completed" if len(records) == expected_runs and not errors else "failed",
        "split": split,
        "run_id": run_dir.name,
        "question_count": len(loaded["questions"]),
        "methods": methods,
        "repeats": repeats,
        "expected_runs": expected_runs,
        "completed_count": len(records),
        "error_count": len(errors),
        "answer_mode": config["answer"]["mode"],
        "model_name": PROXY_MODEL_NAME,
        "input_hashes": run_config["input_hashes"],
    }
    atomic_write_json(run_dir / "matrix-summary.json", summary)
    run_config.update(status=summary["status"])
    atomic_write_json(run_dir / "run-config.json", run_config)
    return {**summary, "run_dir": run_dir.as_posix()}
