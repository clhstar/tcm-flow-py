import csv
import random
from collections import defaultdict
from pathlib import Path

from experiments.rag_v1_6.common import (
    VERSION,
    atomic_write_json,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)
from experiments.rag_v1_6.public_tcm_qg_formal_answer import (
    METHOD_TO_FORMAL_CONFIG,
    load_public_tcm_qg_formal_answer_inputs,
)


FORMAL_REVIEW_COLUMNS = [
    "review_id",
    "blind_method_id",
    "question",
    "answer",
    "citations",
    "evidence",
    "answer_correct",
    "evidence_supported",
    "citation_correct",
    "hallucination",
    "answer_completeness",
    "clinical_safety_issue",
    "reviewer_comment",
]
REVIEW_LABEL_COLUMNS = [
    "answer_correct",
    "evidence_supported",
    "citation_correct",
    "hallucination",
    "answer_completeness",
    "clinical_safety_issue",
]
METHODS = ["B0", "B4", "P", "P-no-parent"]
NUMERIC_REVIEW_VALUE_MAP = {
    "answer_correct": {"1": "yes", "0": "no"},
    "evidence_supported": {"1": "yes", "0": "no"},
    "citation_correct": {"1": "yes", "0": "no"},
    "hallucination": {"1": "yes", "0": "no"},
    "answer_completeness": {
        "3": "score_3",
        "2": "score_2",
        "1": "score_1",
        "0": "score_0",
    },
    "clinical_safety_issue": {"1": "yes", "0": "no"},
}


def review_csv_columns() -> list[str]:
    return list(FORMAL_REVIEW_COLUMNS)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[dict], str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing review CSV: {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                missing = [
                    column
                    for column in FORMAL_REVIEW_COLUMNS
                    if column not in (reader.fieldnames or [])
                ]
                if missing:
                    raise ValueError(
                        f"review CSV missing columns: {', '.join(missing)}"
                    )
                return list(reader), encoding
        except UnicodeDecodeError as error:
            last_error = error
            continue
    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"could not decode review CSV with supported encodings: {last_error}",
    )


def _answers_by_question(answer_run_dir: Path) -> dict[str, dict[str, dict]]:
    rows = read_jsonl(answer_run_dir / "per-answer.jsonl", label="formal answer rows")
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if int(row.get("repeat_index", 0)) != 0:
            continue
        grouped[row["qa_id"]][row["method"]] = row
    return grouped


def _evidence_text(*, method: str, qa_id: str, retrieval: dict[str, dict]) -> str:
    if method == "B0":
        return ""
    evidence = retrieval[method][qa_id]["evidence"]
    return "\n".join(f"{item['label']}: {item['text']}" for item in evidence)


def _review_rows_for_questions(
    *,
    question_ids: list[str],
    questions: dict[str, dict],
    answers: dict[str, dict[str, dict]],
    retrieval: dict[str, dict],
    blind_map: dict[str, str],
    track: str,
    methods: list[str],
) -> tuple[list[dict], list[dict]]:
    review_rows = []
    key_rows = []
    for qa_id in question_ids:
        question = questions[qa_id]
        for method in methods:
            answer = answers[qa_id][method]
            blind_id = blind_map[method]
            review_id = f"{track}-{qa_id}-{blind_id}"
            review_rows.append(
                {
                    "review_id": review_id,
                    "blind_method_id": blind_id,
                    "question": question["question"],
                    "answer": answer["answer"],
                    "citations": ",".join(answer["citations"]),
                    "evidence": _evidence_text(
                        method=method,
                        qa_id=qa_id,
                        retrieval=retrieval,
                    ),
                    "answer_correct": "",
                    "evidence_supported": "",
                    "citation_correct": "",
                    "hallucination": "",
                    "answer_completeness": "",
                    "clinical_safety_issue": "",
                    "reviewer_comment": "",
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "review_track": track,
                    "qa_id": qa_id,
                    "source_doc_id": question["source_doc_id"],
                    "blind_method_id": blind_id,
                    "method": method,
                    "repeat_index": answer["repeat_index"],
                }
            )
    return review_rows, key_rows


def prepare_public_tcm_qg_formal_answer_review(
    *,
    answer_run_dir: Path,
    dataset_path: Path,
    retrieval_matrix_dir: Path,
    output_dir: Path,
    main_review_questions: int,
    second_review_rate: float,
    parent_ablation_focus_questions: int,
    seed: int,
) -> dict:
    summary = read_json(answer_run_dir / "matrix-summary.json", label="formal answer summary")
    if summary.get("status") != "completed" or summary.get("split") != "test":
        raise ValueError("only a completed formal test answer run can be reviewed")
    loaded = load_public_tcm_qg_formal_answer_inputs(
        split="test",
        dataset_path=dataset_path,
        retrieval_matrix_dir=retrieval_matrix_dir,
    )
    answers = _answers_by_question(answer_run_dir)
    eligible = sorted(
        qa_id
        for qa_id in loaded["questions"]
        if all(method in answers.get(qa_id, {}) for method in METHODS)
    )
    if not eligible:
        raise ValueError("no complete answer rows available for review")
    rng = random.Random(seed)
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    main_question_ids = sorted(shuffled[: min(main_review_questions, len(shuffled))])
    parent_question_ids = sorted(
        shuffled[: min(parent_ablation_focus_questions, len(shuffled))]
    )
    blind_ids = ["A", "B", "C", "D"]
    shuffled_methods = list(METHODS)
    rng.shuffle(shuffled_methods)
    blind_map = dict(zip(shuffled_methods, blind_ids))
    main_rows, main_keys = _review_rows_for_questions(
        question_ids=main_question_ids,
        questions=loaded["questions"],
        answers=answers,
        retrieval=loaded["retrieval"],
        blind_map=blind_map,
        track="main",
        methods=METHODS,
    )
    second_count = min(len(main_rows), int(round(len(main_rows) * second_review_rate)))
    second_rows = sorted(rng.sample(main_rows, second_count), key=lambda row: row["review_id"])
    parent_rows, parent_keys = _review_rows_for_questions(
        question_ids=parent_question_ids,
        questions=loaded["questions"],
        answers=answers,
        retrieval=loaded["retrieval"],
        blind_map=blind_map,
        track="parent",
        methods=["P", "P-no-parent"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "formal-answer-review-main.csv"
    second_path = output_dir / "formal-answer-review-second.csv"
    parent_path = output_dir / "formal-answer-review-parent-ablation.csv"
    key_path = output_dir / "formal-answer-review-blind-key.csv"
    _write_csv(main_path, main_rows, FORMAL_REVIEW_COLUMNS)
    _write_csv(second_path, second_rows, FORMAL_REVIEW_COLUMNS)
    _write_csv(parent_path, parent_rows, FORMAL_REVIEW_COLUMNS)
    _write_csv(
        key_path,
        main_keys + parent_keys,
        [
            "review_id",
            "review_track",
            "qa_id",
            "source_doc_id",
            "blind_method_id",
            "method",
            "repeat_index",
        ],
    )
    manifest = {
        "version": VERSION,
        "status": "ready",
        "stage": "public_tcm_qg_formal_answer_review_prepared",
        "generated_at": utc_now(),
        "main_review_questions": len(main_question_ids),
        "main_review_rows": len(main_rows),
        "second_review_rows": len(second_rows),
        "parent_ablation_focus_questions": len(parent_question_ids),
        "parent_ablation_rows": len(parent_rows),
        "blind_key_written": True,
        "files": {
            "main_review_csv": main_path.as_posix(),
            "main_review_csv_sha256": sha256_file(main_path),
            "second_review_csv": second_path.as_posix(),
            "second_review_csv_sha256": sha256_file(second_path),
            "parent_ablation_review_csv": parent_path.as_posix(),
            "parent_ablation_review_csv_sha256": sha256_file(parent_path),
            "blind_key_csv": key_path.as_posix(),
            "blind_key_csv_sha256": sha256_file(key_path),
        },
        "inputs": {
            "answer_run_summary_sha256": sha256_file(answer_run_dir / "matrix-summary.json"),
            "per_answer_sha256": sha256_file(answer_run_dir / "per-answer.jsonl"),
            "dataset_sha256": sha256_file(dataset_path),
            "retrieval_matrix_summary_sha256": sha256_file(
                retrieval_matrix_dir / "matrix-summary.json"
            ),
        },
        "privacy": {
            "review_csv_contains_qa_and_generated_answers": True,
            "manifest_contains_raw_content": False,
        },
    }
    atomic_write_json(output_dir / "review-package-manifest.json", manifest)
    return manifest


def _completed(row: dict) -> bool:
    return all(row.get(column, "").strip() for column in REVIEW_LABEL_COLUMNS)


def _normalize_review_value(column: str, value: str) -> str:
    normalized = value.strip().lower()
    return NUMERIC_REVIEW_VALUE_MAP.get(column, {}).get(normalized, normalized)


def _rate(rows: list[dict], column: str, positive: str) -> float | None:
    values = [
        _normalize_review_value(column, row[column])
        for row in rows
        if row.get(column, "").strip()
    ]
    if not values:
        return None
    return sum(value == positive for value in values) / len(values)


def _completeness_distribution(rows: list[dict]) -> dict:
    buckets = {
        "score_3": 0,
        "score_2": 0,
        "score_1": 0,
        "score_0": 0,
    }
    for row in rows:
        value = _normalize_review_value(
            "answer_completeness",
            row.get("answer_completeness", ""),
        )
        if value in buckets:
            buckets[value] += 1
    return buckets


def import_public_tcm_qg_formal_answer_review(
    *,
    reviewed_csv_path: Path,
    second_reviewed_csv_path: Path,
    parent_ablation_reviewed_csv_path: Path | None = None,
    output_path: Path,
) -> dict:
    rows, reviewed_encoding = _read_csv(reviewed_csv_path)
    second_rows, second_reviewed_encoding = _read_csv(second_reviewed_csv_path)
    parent_rows = []
    parent_encoding = None
    if parent_ablation_reviewed_csv_path is not None:
        parent_rows, parent_encoding = _read_csv(parent_ablation_reviewed_csv_path)
    completed_rows = [row for row in rows if _completed(row)]
    pending_rows = [row for row in rows if not _completed(row)]
    second_completed_rows = [row for row in second_rows if _completed(row)]
    second_pending_rows = [row for row in second_rows if not _completed(row)]
    parent_completed_rows = [row for row in parent_rows if _completed(row)]
    parent_pending_rows = [row for row in parent_rows if not _completed(row)]
    rows_by_id = {row["review_id"]: row for row in completed_rows}
    disagreement_count = 0
    for second in second_completed_rows:
        primary = rows_by_id.get(second["review_id"])
        if not primary:
            continue
        if any(
            _normalize_review_value(column, primary[column])
            != _normalize_review_value(column, second[column])
            for column in REVIEW_LABEL_COLUMNS
        ):
            disagreement_count += 1
    answer_review_completed = (
        not pending_rows
        and not second_pending_rows
        and not parent_pending_rows
    )
    summary = {
        "version": VERSION,
        "status": "ready" if answer_review_completed else "partial",
        "stage": "public_tcm_qg_formal_answer_review_imported",
        "generated_at": utc_now(),
        "answer_review_completed": answer_review_completed,
        "reviewed_count": len(completed_rows),
        "pending_count": len(pending_rows),
        "second_review_count": len(second_completed_rows),
        "second_review_pending_count": len(second_pending_rows),
        "parent_ablation_reviewed_count": len(parent_completed_rows),
        "parent_ablation_pending_count": len(parent_pending_rows),
        "disagreement_count": disagreement_count,
        "metrics": {
            "answer_correct_rate": _rate(completed_rows, "answer_correct", "yes"),
            "evidence_supported_rate": _rate(completed_rows, "evidence_supported", "yes"),
            "citation_correct_rate": _rate(completed_rows, "citation_correct", "yes"),
            "hallucination_rate": _rate(completed_rows, "hallucination", "yes"),
            "answer_completeness_distribution": _completeness_distribution(
                completed_rows
            ),
            "clinical_safety_issue_rate": _rate(
                completed_rows,
                "clinical_safety_issue",
                "yes",
            ),
        },
        "inputs": {
            "review_csv_sha256": sha256_file(reviewed_csv_path),
            "second_review_csv_sha256": sha256_file(second_reviewed_csv_path),
            "review_csv_encoding": reviewed_encoding,
            "second_review_csv_encoding": second_reviewed_encoding,
            "parent_ablation_review_csv_sha256": (
                sha256_file(parent_ablation_reviewed_csv_path)
                if parent_ablation_reviewed_csv_path is not None
                else None
            ),
            "parent_ablation_review_csv_encoding": parent_encoding,
        },
        "privacy": {
            "summary_contains_raw_content": False,
        },
    }
    atomic_write_json(output_path, summary)
    return summary
