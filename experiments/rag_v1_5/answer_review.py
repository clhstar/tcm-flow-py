import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from experiments.rag_v1_5.runner import (
    _atomic_write_json,
    _read_json,
    _read_jsonl_strict,
    _sha256_file,
)


REVIEW_FIELDS = (
    "review_id",
    "review_track",
    "question_id",
    "book_scope",
    "question_type",
    "question",
    "reference_answer",
    "evidence",
    "answer",
    "blind_method",
    "answer_correct",
    "evidence_support",
    "citation_correct",
    "appropriate_refusal",
    "hallucination",
    "review_status",
    "review_comment",
    "second_review_required",
    "second_review_status",
)
IMMUTABLE_REVIEW_FIELDS = (
    "review_id",
    "review_track",
    "question_id",
    "book_scope",
    "question_type",
    "question",
    "reference_answer",
    "evidence",
    "answer",
    "blind_method",
    "second_review_required",
)
REVIEW_LABEL_FIELDS = (
    "answer_correct",
    "evidence_support",
    "citation_correct",
    "appropriate_refusal",
    "hallucination",
    "review_status",
)


ANSWER_CORRECT_VALUES = {"yes", "partial", "no"}
EVIDENCE_SUPPORT_VALUES = {
    "full",
    "partial",
    "none",
    "not_applicable",
}
CITATION_CORRECT_VALUES = {
    "yes",
    "partial",
    "no",
    "not_applicable",
}
APPROPRIATE_REFUSAL_VALUES = {
    "yes",
    "no",
    "not_applicable",
}
HALLUCINATION_VALUES = {"yes", "no"}
REVIEW_STATUS_VALUES = {"pass", "revise", "exclude"}


def _sample_answerable_questions(
    *,
    questions: list[dict],
    count: int,
    rng: random.Random,
) -> list[dict]:
    strata = defaultdict(list)
    for question in questions:
        if not question["answerable"]:
            continue
        strata[
            (
                question["book_scope"],
                question["question_type"],
            )
        ].append(question)
    if not strata:
        raise ValueError("盲审抽样缺少 answerable 问题")
    stratum_keys = sorted(strata)
    base, remainder = divmod(count, len(stratum_keys))
    selected = []
    for index, key in enumerate(stratum_keys):
        quota = base + int(index < remainder)
        candidates = sorted(
            strata[key],
            key=lambda row: row["question_id"],
        )
        if len(candidates) < quota:
            raise ValueError(f"盲审分层样本不足: {key}")
        selected.extend(rng.sample(candidates, quota))
    return sorted(selected, key=lambda row: row["question_id"])


def _build_review_rows(
    *,
    questions: list[dict],
    answers: list[dict],
    seed: int,
    answerable_count: int,
    unanswerable_count: int,
    canonical_repeat_index: int,
) -> tuple[list[dict], dict[str, str]]:
    rng = random.Random(seed)
    answerable = _sample_answerable_questions(
        questions=questions,
        count=answerable_count,
        rng=rng,
    )
    unanswerable_candidates = sorted(
        (
            question
            for question in questions
            if not question["answerable"]
        ),
        key=lambda row: row["question_id"],
    )
    if len(unanswerable_candidates) < unanswerable_count:
        raise ValueError("盲审 unanswerable 样本不足")
    unanswerable = sorted(
        rng.sample(unanswerable_candidates, unanswerable_count),
        key=lambda row: row["question_id"],
    )
    answer_map = {
        (
            answer["question_id"],
            answer["method"],
            answer["repeat_index"],
        ): answer
        for answer in answers
    }
    if len(answer_map) != len(answers):
        raise ValueError("盲审答案存在重复 question/method/repeat")

    rows = []
    blind_key = {}

    def add_track(
        *,
        track: str,
        selected_questions: list[dict],
        methods: tuple[str, ...],
    ) -> None:
        labels = tuple(
            chr(ord("A") + index) for index in range(len(methods))
        )
        for question in selected_questions:
            shuffled_methods = list(methods)
            rng.shuffle(shuffled_methods)
            for label, method in zip(labels, shuffled_methods):
                answer_key = (
                    question["question_id"],
                    method,
                    canonical_repeat_index,
                )
                if answer_key not in answer_map:
                    raise ValueError(
                        f"盲审缺少冻结答案: {answer_key}"
                    )
                answer = answer_map[answer_key]
                review_id = (
                    f"{track}-{question['question_id']}-{label}"
                )
                blind_key[review_id] = method
                rows.append(
                    {
                        "review_id": review_id,
                        "review_track": track,
                        "question_id": question["question_id"],
                        "book_scope": question["book_scope"],
                        "question_type": question["question_type"],
                        "question": question["question"],
                        "reference_answer": question[
                            "reference_answer"
                        ],
                        "evidence": answer.get("evidence", []),
                        "answer": answer["answer"],
                        "blind_method": label,
                        "answer_correct": "",
                        "evidence_support": "",
                        "citation_correct": "",
                        "appropriate_refusal": "",
                        "hallucination": "",
                        "review_status": "",
                        "review_comment": "",
                        "second_review_required": "no",
                        "second_review_status": "",
                    }
                )

    main_questions = sorted(
        answerable + unanswerable,
        key=lambda row: row["question_id"],
    )
    add_track(
        track="main",
        selected_questions=main_questions,
        methods=("B0", "B4", "P"),
    )
    add_track(
        track="parent_ablation",
        selected_questions=answerable,
        methods=("P", "P-no-parent"),
    )

    second_review_count = round(len(rows) * 0.10)
    second_review_indexes = set(
        rng.sample(range(len(rows)), second_review_count)
    )
    for index, row in enumerate(rows):
        if index in second_review_indexes:
            row["second_review_required"] = "yes"
    return rows, blind_key


def build_review_sample(
    *,
    questions: list[dict],
    answers: list[dict],
    seed: int,
    answerable_count: int,
    unanswerable_count: int,
    canonical_repeat_index: int,
) -> list[dict]:
    rows, _ = _build_review_rows(
        questions=questions,
        answers=answers,
        seed=seed,
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        canonical_repeat_index=canonical_repeat_index,
    )
    return rows


def _read_review_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少回答审核 CSV: {path}")
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != REVIEW_FIELDS:
            raise ValueError("回答审核 CSV 字段顺序不匹配")
        return [dict(row) for row in reader]


def _validate_review_values(row: dict) -> None:
    validations = {
        "answer_correct": ANSWER_CORRECT_VALUES,
        "evidence_support": EVIDENCE_SUPPORT_VALUES,
        "citation_correct": CITATION_CORRECT_VALUES,
        "appropriate_refusal": APPROPRIATE_REFUSAL_VALUES,
        "hallucination": HALLUCINATION_VALUES,
        "review_status": REVIEW_STATUS_VALUES,
    }
    for field, allowed in validations.items():
        if row.get(field) not in allowed:
            raise ValueError(
                f"{row.get('review_id')} 的 {field} 未完整填写"
            )


def _cohen_kappa(first: list[str], second: list[str]) -> float | None:
    if not first:
        return None
    observed = sum(
        left == right for left, right in zip(first, second)
    ) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    labels = set(first_counts) | set(second_counts)
    expected = sum(
        (first_counts[label] / len(first))
        * (second_counts[label] / len(second))
        for label in labels
    )
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def _review_metrics(rows: list[dict]) -> dict:
    distributions = {
        field: dict(Counter(row[field] for row in rows))
        for field in REVIEW_LABEL_FIELDS
    }

    def applicable_rate(field: str, positive: set[str]) -> float:
        values = [
            row[field]
            for row in rows
            if row[field] != "not_applicable"
        ]
        return (
            sum(value in positive for value in values) / len(values)
            if values
            else 0.0
        )

    return {
        "answer_correct_yes_rate": applicable_rate(
            "answer_correct",
            {"yes"},
        ),
        "answer_correct_yes_or_partial_rate": applicable_rate(
            "answer_correct",
            {"yes", "partial"},
        ),
        "evidence_support_full_or_partial_rate": applicable_rate(
            "evidence_support",
            {"full", "partial"},
        ),
        "citation_correct_yes_or_partial_rate": applicable_rate(
            "citation_correct",
            {"yes", "partial"},
        ),
        "appropriate_refusal_yes_rate": applicable_rate(
            "appropriate_refusal",
            {"yes"},
        ),
        "hallucination_rate": (
            sum(row["hallucination"] == "yes" for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "distributions": distributions,
    }


def import_formal_answer_review(
    *,
    source_snapshot_path: Path,
    reviewed_csv_path: Path,
    second_reviewed_csv_path: Path,
    summary_path: Path,
) -> dict:
    source_payload = json.loads(
        source_snapshot_path.read_text(encoding="utf-8")
    )
    source_rows = {
        row["review_id"]: row for row in source_payload["rows"]
    }
    reviewed_rows = _read_review_csv(reviewed_csv_path)
    reviewed = {
        row["review_id"]: row for row in reviewed_rows
    }
    if len(reviewed) != len(reviewed_rows):
        raise ValueError("首审 CSV 存在重复 review_id")
    if set(reviewed) != set(source_rows):
        raise ValueError("首审 CSV 行集合与冻结样本不一致")
    for review_id, row in reviewed.items():
        source = source_rows[review_id]
        for field in IMMUTABLE_REVIEW_FIELDS:
            if row.get(field, "") != str(source.get(field, "")):
                raise ValueError(
                    f"{review_id} 的冻结字段被修改: {field}"
                )
        _validate_review_values(row)
        if (
            row["second_review_required"] == "yes"
            and row.get("second_review_status")
            not in {"completed", "adjudicated"}
        ):
            raise ValueError(
                f"{review_id} 的 second_review_status 未完成"
            )

    second_rows = _read_review_csv(second_reviewed_csv_path)
    second = {row["review_id"]: row for row in second_rows}
    if len(second) != len(second_rows):
        raise ValueError("二审 CSV 存在重复 review_id")
    required_second_ids = {
        review_id
        for review_id, row in source_rows.items()
        if str(row.get("second_review_required")) == "yes"
    }
    if set(second) != required_second_ids:
        raise ValueError("二审 CSV 行集合与固定 10% 样本不一致")
    for review_id, row in second.items():
        source = source_rows[review_id]
        for field in IMMUTABLE_REVIEW_FIELDS:
            if row.get(field, "") != str(source.get(field, "")):
                raise ValueError(
                    f"{review_id} 的二审冻结字段被修改: {field}"
                )
        _validate_review_values(row)
        if row.get("second_review_status") != "completed":
            raise ValueError(f"{review_id} 的二审尚未完成")

    disagreement_ids = []
    agreement_by_field = {}
    kappa_by_field = {}
    for field in REVIEW_LABEL_FIELDS:
        agreements = []
        first_values = []
        second_values = []
        for review_id, second_row in second.items():
            first_values.append(reviewed[review_id][field])
            second_values.append(second_row[field])
            agrees = (
                reviewed[review_id][field] == second_row[field]
            )
            agreements.append(agrees)
            if not agrees and review_id not in disagreement_ids:
                disagreement_ids.append(review_id)
        agreement_by_field[field] = (
            sum(agreements) / len(agreements)
            if agreements
            else None
        )
        kappa_by_field[field] = _cohen_kappa(
            first_values,
            second_values,
        )
    unresolved_ids = [
        review_id
        for review_id in disagreement_ids
        if reviewed[review_id]["second_review_status"]
        != "adjudicated"
    ]
    final_rows = []
    for review_id, row in reviewed.items():
        final_row = dict(row)
        if (
            review_id in disagreement_ids
            and row["second_review_status"] == "adjudicated"
        ):
            try:
                adjudicated = json.loads(row["review_comment"])
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{review_id} 的裁决 review_comment 必须为 JSON"
                ) from error
            if set(adjudicated) != set(REVIEW_LABEL_FIELDS):
                raise ValueError(
                    f"{review_id} 的裁决 JSON 必须包含全部审核标签"
                )
            final_row.update(adjudicated)
            _validate_review_values(final_row)
        final_rows.append(final_row)
    status = "needs_adjudication" if unresolved_ids else "ready"
    summary = {
        "status": status,
        "reviewed_count": len(reviewed),
        "second_review_count": len(second),
        "disagreement_count": len(disagreement_ids),
        "disagreement_review_ids": sorted(disagreement_ids),
        "unresolved_adjudication_count": len(unresolved_ids),
        "unresolved_adjudication_review_ids": sorted(
            unresolved_ids
        ),
        "agreement_by_field": agreement_by_field,
        "kappa_by_field": kappa_by_field,
        "metrics": _review_metrics(final_rows),
        "answer_review_completed": not unresolved_ids,
    }
    _atomic_write_json(summary_path, summary)
    return summary


def _write_review_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=REVIEW_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_formal_answer_review(
    *,
    run_dir: Path,
    dataset_path: Path,
    matrix_dir: Path,
    answer_prereg_path: Path,
    config_path: Path,
    formal_manifest_path: Path,
    formal_runs_manifest_path: Path,
    output_dir: Path,
) -> dict:
    from experiments.rag_v1_5.formal_answer import (
        load_frozen_answer_inputs,
    )

    matrix_summary = _read_json(
        run_dir / "matrix-summary.json",
        label="Formal answer test summary",
    )
    if (
        matrix_summary.get("status") != "completed"
        or matrix_summary.get("split") != "formal_test"
        or matrix_summary.get("error_count") != 0
    ):
        raise ValueError(
            "盲审样本只允许来自无未解决错误的 completed formal_test"
        )
    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    review_config = config["human_review"]
    loaded = load_frozen_answer_inputs(
        dataset_path=dataset_path,
        matrix_dir=matrix_dir,
        answer_prereg_path=answer_prereg_path,
        split="formal_test",
        formal_manifest_path=formal_manifest_path,
        formal_runs_manifest_path=formal_runs_manifest_path,
    )
    answer_rows = _read_jsonl_strict(
        run_dir / "per-answer.jsonl",
        label="Formal answer test records",
    )
    enriched_answers = []
    for answer in answer_rows:
        if answer["method"] == "B0":
            evidence = []
        else:
            evidence = loaded["retrieval"][answer["method"]][
                answer["question_id"]
            ]["evidence"]
        enriched_answers.append(
            {
                **answer,
                "evidence": evidence,
            }
        )
    rows, blind_key = _build_review_rows(
        questions=list(loaded["questions"].values()),
        answers=enriched_answers,
        seed=review_config["sample_seed"],
        answerable_count=review_config[
            "answerable_questions"
        ],
        unanswerable_count=review_config[
            "unanswerable_questions"
        ],
        canonical_repeat_index=review_config[
            "canonical_repeat_index"
        ],
    )
    serialized_rows = []
    for row in rows:
        serialized_rows.append(
            {
                **row,
                "evidence": json.dumps(
                    row["evidence"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_csv_path = output_dir / "formal-answer-review.csv"
    second_csv_path = (
        output_dir / "formal-answer-second-review.csv"
    )
    source_snapshot_path = (
        output_dir / "formal-answer-review-source.json"
    )
    blind_key_path = output_dir / "formal-answer-blind-key.json"
    prepared_summary_path = (
        output_dir / "formal-answer-review-prepared.json"
    )
    _write_review_csv(reviewed_csv_path, serialized_rows)
    second_rows = [
        row
        for row in serialized_rows
        if row["second_review_required"] == "yes"
    ]
    _write_review_csv(second_csv_path, second_rows)
    _atomic_write_json(
        source_snapshot_path,
        {
            "rows": serialized_rows,
            "run_summary_sha256": _sha256_file(
                run_dir / "matrix-summary.json"
            ),
            "per_answer_sha256": _sha256_file(
                run_dir / "per-answer.jsonl"
            ),
        },
    )
    _atomic_write_json(
        blind_key_path,
        {
            "private": True,
            "mapping": blind_key,
        },
    )
    summary = {
        "status": "prepared",
        "review_row_count": len(serialized_rows),
        "main_row_count": sum(
            row["review_track"] == "main"
            for row in serialized_rows
        ),
        "parent_ablation_row_count": sum(
            row["review_track"] == "parent_ablation"
            for row in serialized_rows
        ),
        "second_review_row_count": len(second_rows),
        "files": {
            "reviewed_csv": {
                "path": reviewed_csv_path.as_posix(),
                "sha256": _sha256_file(reviewed_csv_path),
            },
            "second_reviewed_csv": {
                "path": second_csv_path.as_posix(),
                "sha256": _sha256_file(second_csv_path),
            },
            "source_snapshot": {
                "path": source_snapshot_path.as_posix(),
                "sha256": _sha256_file(source_snapshot_path),
            },
            "blind_key": {
                "path": blind_key_path.as_posix(),
                "sha256": _sha256_file(blind_key_path),
            },
        },
    }
    _atomic_write_json(prepared_summary_path, summary)
    return summary
