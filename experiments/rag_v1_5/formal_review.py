import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.rag_v1_5.formal_dataset import (
    FORMAL_BOOKS,
    FORMAL_PER_BOOK_SPLIT,
    FORMAL_QUESTION_TYPES,
    FORMAL_SPLITS,
    _read_jsonl,
    _write_jsonl,
)
from experiments.rag_v1_5.schema import (
    FormalEvidenceGroup,
    PilotQuestion,
)


FORMAL_SECOND_REVIEW_SEED = 20260614
FORMAL_REVIEW_IMMUTABLE_FIELDS = (
    "question_id",
    "content_sha256",
    "split",
    "book_scope",
    "question_type",
    "answerable",
    "question",
    "reference_answer",
    "gold_evidence_ids",
    "gold_clause_ids",
    "graded_relevance",
    "support_spans",
    "evidence_group_id",
    "question_version",
    "second_review_required",
)
FORMAL_REVIEW_FIRST_FIELDS = (
    "first_status",
    "first_decision",
    "first_comment",
    "first_reviewer",
    "first_reviewed_at",
)
FORMAL_REVIEW_SECOND_FIELDS = (
    "second_status",
    "second_decision",
    "second_comment",
    "second_reviewer",
    "second_reviewed_at",
)
FORMAL_REVIEW_FIELDS = (
    FORMAL_REVIEW_IMMUTABLE_FIELDS
    + FORMAL_REVIEW_FIRST_FIELDS
    + FORMAL_REVIEW_SECOND_FIELDS
)
FORMAL_REVIEW_DECISIONS = {
    "correct",
    "question_unnatural",
    "unsupported_answer",
    "gold_id_error",
    "relevance_error",
    "clinical_scope",
    "duplicate",
    "source_text_issue",
    "other",
}
FORMAL_REVIEW_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp936",
    "gb18030",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_review_inputs(
    *,
    draft_dataset_path: Path,
    evidence_groups_path: Path,
) -> tuple[list[PilotQuestion], dict[str, FormalEvidenceGroup]]:
    questions = _read_jsonl(draft_dataset_path, PilotQuestion)
    groups = _read_jsonl(
        evidence_groups_path,
        FormalEvidenceGroup,
    )
    if len(questions) != 400 or len(groups) != 400:
        raise ValueError("Formal 审核输入必须包含 400 题和 400 个 Evidence Group")

    question_ids = [question.question_id for question in questions]
    group_ids = [group.group_id for group in groups]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Formal 审核草稿存在重复 question_id")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Formal 审核 Evidence Group 存在重复 group_id")
    groups_by_id = {group.group_id: group for group in groups}

    quota = Counter(
        (
            question.book_scope,
            question.split,
            question.question_type,
        )
        for question in questions
    )
    expected_quota = {
        (book, split, question_type): count
        for book in FORMAL_BOOKS
        for split in FORMAL_SPLITS
        for question_type, count in FORMAL_PER_BOOK_SPLIT.items()
    }
    if quota != Counter(expected_quota):
        raise ValueError("Formal 审核草稿的书籍 × split × 类型配额错误")
    if sum(question.answerable for question in questions) != 320:
        raise ValueError("Formal 审核草稿必须包含 320 条可回答问题")

    referenced_group_ids = []
    for question in questions:
        if (
            question.split not in FORMAL_SPLITS
            or not question.evidence_group_id
            or question.review_status != "draft"
        ):
            raise ValueError(
                f"{question.question_id} 不是合法 Formal 审核草稿"
            )
        group = groups_by_id.get(question.evidence_group_id)
        if group is None:
            raise ValueError(
                f"{question.question_id} 找不到 Evidence Group"
            )
        if (
            group.book_scope != question.book_scope
            or group.split != question.split
            or group.question_type != question.question_type
        ):
            raise ValueError(
                f"{question.question_id} 与 Evidence Group 不一致"
            )
        referenced_group_ids.append(question.evidence_group_id)
    if (
        len(set(referenced_group_ids)) != 400
        or set(referenced_group_ids) != set(groups_by_id)
    ):
        raise ValueError("Formal 问题与 Evidence Group 必须一一对应")
    return questions, groups_by_id


def _second_review_question_ids(
    questions: list[PilotQuestion],
    *,
    seed: int,
) -> list[str]:
    selected_ids = []
    for book in FORMAL_BOOKS:
        for split in FORMAL_SPLITS:
            for question_type in FORMAL_QUESTION_TYPES:
                candidates = sorted(
                    (
                        question
                        for question in questions
                        if (
                            question.book_scope == book
                            and question.split == split
                            and question.question_type == question_type
                        )
                    ),
                    key=lambda question: question.question_id,
                )
                scored = sorted(
                    candidates,
                    key=lambda question: hashlib.sha256(
                        (
                            f"{seed}:{book}:{split}:{question_type}:"
                            f"{question.question_id}"
                        ).encode("utf-8")
                    ).hexdigest(),
                )
                selected = scored[:2]
                if len(selected) != 2:
                    raise ValueError(
                        "Formal 二审分层候选不足: "
                        f"{book}/{split}/{question_type}"
                    )
                selected_ids.extend(
                    question.question_id for question in selected
                )
    return selected_ids


def _content_payload(
    question: PilotQuestion,
    *,
    second_review_required: bool,
) -> dict:
    return {
        "question_id": question.question_id,
        "split": question.split,
        "book_scope": question.book_scope,
        "question_type": question.question_type,
        "answerable": question.answerable,
        "question": question.question,
        "reference_answer": question.reference_answer,
        "gold_evidence_ids": question.gold_evidence_ids,
        "gold_clause_ids": question.gold_clause_ids,
        "graded_relevance": question.graded_relevance,
        "support_spans": question.support_spans,
        "evidence_group_id": question.evidence_group_id,
        "question_version": question.question_version,
        "second_review_required": second_review_required,
    }


def _content_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest().upper()


def _json_cell(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _immutable_row(
    question: PilotQuestion,
    *,
    second_review_required: bool,
) -> dict[str, str]:
    payload = _content_payload(
        question,
        second_review_required=second_review_required,
    )
    return {
        "question_id": question.question_id,
        "content_sha256": _content_sha256(payload),
        "split": question.split or "",
        "book_scope": question.book_scope,
        "question_type": question.question_type,
        "answerable": str(question.answerable).lower(),
        "question": question.question,
        "reference_answer": question.reference_answer,
        "gold_evidence_ids": _json_cell(question.gold_evidence_ids),
        "gold_clause_ids": _json_cell(question.gold_clause_ids),
        "graded_relevance": _json_cell(question.graded_relevance),
        "support_spans": _json_cell(question.support_spans),
        "evidence_group_id": question.evidence_group_id or "",
        "question_version": str(question.question_version),
        "second_review_required": str(
            second_review_required
        ).lower(),
    }


def _decode_review_csv(path: Path) -> tuple[str, str, bytes]:
    raw_bytes = path.read_bytes()
    errors = []
    for encoding in FORMAL_REVIEW_ENCODINGS:
        try:
            return raw_bytes.decode(encoding), encoding, raw_bytes
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")
    raise ValueError(
        f"无法解码 Formal 审核 CSV: {path}; " + "; ".join(errors)
    )


def _parse_review_csv(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != FORMAL_REVIEW_FIELDS:
        raise ValueError("Formal 审核 CSV 列与固定模板不一致")
    rows = list(reader)
    question_ids = [row["question_id"] for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Formal 审核 CSV 存在重复 question_id")
    return rows


def _read_existing_review(
    path: Path,
) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    text, _, _ = _decode_review_csv(path)
    return {
        row["question_id"]: row
        for row in _parse_review_csv(text)
    }


def _write_review_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=FORMAL_REVIEW_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_formal_review(
    *,
    draft_dataset_path: Path,
    evidence_groups_path: Path,
    review_csv_path: Path,
    second_review_seed: int = FORMAL_SECOND_REVIEW_SEED,
) -> dict:
    questions, _ = _load_review_inputs(
        draft_dataset_path=draft_dataset_path,
        evidence_groups_path=evidence_groups_path,
    )
    second_review_ids = set(
        _second_review_question_ids(
            questions,
            seed=second_review_seed,
        )
    )
    existing_rows = _read_existing_review(review_csv_path)
    rows = []
    inherited_count = 0
    for question in sorted(
        questions,
        key=lambda item: item.question_id,
    ):
        immutable = _immutable_row(
            question,
            second_review_required=(
                question.question_id in second_review_ids
            ),
        )
        row = {
            **immutable,
            "first_status": "pending",
            "first_decision": "",
            "first_comment": "",
            "first_reviewer": "",
            "first_reviewed_at": "",
            "second_status": "pending",
            "second_decision": "",
            "second_comment": "",
            "second_reviewer": "",
            "second_reviewed_at": "",
        }
        existing = existing_rows.get(question.question_id)
        if (
            existing is not None
            and existing.get("content_sha256")
            == immutable["content_sha256"]
        ):
            for field in (
                FORMAL_REVIEW_FIRST_FIELDS
                + FORMAL_REVIEW_SECOND_FIELDS
            ):
                row[field] = existing[field]
            inherited_count += 1
        rows.append(row)
    _write_review_csv(review_csv_path, rows)
    return {
        "question_count": len(rows),
        "inherited_review_count": inherited_count,
        "reset_review_count": len(rows) - inherited_count,
        "second_review_required_count": len(second_review_ids),
        "second_review_question_ids": sorted(second_review_ids),
        "review_csv_sha256": _sha256_file(review_csv_path),
    }


def _validate_review_round(
    row: dict[str, str],
    *,
    prefix: str,
) -> str:
    status = row[f"{prefix}_status"].strip()
    decision = row[f"{prefix}_decision"].strip()
    reviewer = row[f"{prefix}_reviewer"].strip()
    reviewed_at = row[f"{prefix}_reviewed_at"].strip()
    if status not in {"pending", "pass", "fail"}:
        raise ValueError(
            f"{row['question_id']} {prefix}_status 非法: {status}"
        )
    if decision and decision not in FORMAL_REVIEW_DECISIONS:
        raise ValueError(
            f"{row['question_id']} {prefix}_decision 非法: {decision}"
        )
    if status == "pending":
        if decision or reviewer or reviewed_at:
            raise ValueError(
                f"{row['question_id']} pending 审核字段必须留空"
            )
        return status
    if not reviewer or not reviewed_at:
        raise ValueError(
            f"{row['question_id']} {prefix} reviewer/date 不能为空"
        )
    if status == "pass" and decision != "correct":
        raise ValueError(
            f"{row['question_id']} pass 必须对应 correct"
        )
    if status == "fail" and (
        not decision or decision == "correct"
    ):
        raise ValueError(
            f"{row['question_id']} fail 不能对应 correct"
        )
    return status


def _immutable_values_equal(
    *,
    field: str,
    reviewed_value: str,
    expected_value: str,
) -> bool:
    if field in {"answerable", "second_review_required"}:
        return reviewed_value.strip().lower() == expected_value
    return reviewed_value == expected_value


def _normalize_review_encoding(
    *,
    reviewed_csv_path: Path,
    rows: list[dict[str, str]],
    detected_encoding: str,
    original_bytes: bytes,
) -> dict:
    original_sha256 = hashlib.sha256(original_bytes).hexdigest().upper()
    if detected_encoding in {"utf-8-sig", "utf-8"}:
        return {
            "detected_encoding": detected_encoding,
            "converted": False,
            "original_sha256": original_sha256,
            "backup_path": None,
            "backup_sha256": None,
            "normalized_sha256": original_sha256,
            "unicode_equivalent": True,
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_path = reviewed_csv_path.with_name(
        f"{reviewed_csv_path.stem}.{detected_encoding}-backup-"
        f"{timestamp}{reviewed_csv_path.suffix}"
    )
    suffix = 1
    while backup_path.exists():
        backup_path = reviewed_csv_path.with_name(
            f"{reviewed_csv_path.stem}.{detected_encoding}-backup-"
            f"{timestamp}-{suffix:02d}{reviewed_csv_path.suffix}"
        )
        suffix += 1
    backup_path.write_bytes(original_bytes)
    _write_review_csv(reviewed_csv_path, rows)
    normalized_text, normalized_encoding, _ = _decode_review_csv(
        reviewed_csv_path
    )
    normalized_rows = _parse_review_csv(normalized_text)
    unicode_equivalent = normalized_rows == rows
    if normalized_encoding != "utf-8-sig" or not unicode_equivalent:
        raise ValueError("Formal 审核 CSV 编码规范化未保持 Unicode 等价")
    return {
        "detected_encoding": detected_encoding,
        "converted": True,
        "original_sha256": original_sha256,
        "backup_path": str(backup_path.resolve()),
        "backup_sha256": _sha256_file(backup_path),
        "normalized_sha256": _sha256_file(reviewed_csv_path),
        "unicode_equivalent": unicode_equivalent,
    }


def import_formal_review(
    *,
    draft_dataset_path: Path,
    evidence_groups_path: Path,
    reviewed_csv_path: Path,
    output_dataset_path: Path,
    summary_path: Path,
) -> dict:
    questions, _ = _load_review_inputs(
        draft_dataset_path=draft_dataset_path,
        evidence_groups_path=evidence_groups_path,
    )
    second_review_ids = set(
        _second_review_question_ids(
            questions,
            seed=FORMAL_SECOND_REVIEW_SEED,
        )
    )
    text, detected_encoding, original_bytes = _decode_review_csv(
        reviewed_csv_path
    )
    rows = _parse_review_csv(text)
    if len(rows) != 400:
        raise ValueError("Formal 审核 CSV 必须恰好包含 400 行")
    encoding_summary = _normalize_review_encoding(
        reviewed_csv_path=reviewed_csv_path,
        rows=rows,
        detected_encoding=detected_encoding,
        original_bytes=original_bytes,
    )

    questions_by_id = {
        question.question_id: question for question in questions
    }
    if set(row["question_id"] for row in rows) != set(questions_by_id):
        raise ValueError("Formal 审核 CSV question_id 与草稿不一致")

    first_statuses = []
    second_required_statuses = []
    for row in rows:
        question = questions_by_id[row["question_id"]]
        expected = _immutable_row(
            question,
            second_review_required=(
                question.question_id in second_review_ids
            ),
        )
        for field in FORMAL_REVIEW_IMMUTABLE_FIELDS:
            if not _immutable_values_equal(
                field=field,
                reviewed_value=row[field],
                expected_value=expected[field],
            ):
                raise ValueError(
                    f"{question.question_id} 不允许修改列: {field}"
                )
        first_statuses.append(
            _validate_review_round(row, prefix="first")
        )
        second_status = _validate_review_round(
            row,
            prefix="second",
        )
        if question.question_id in second_review_ids:
            second_required_statuses.append(second_status)
        elif second_status != "pending":
            raise ValueError(
                f"{question.question_id} 非二审样本不得填写二审结论"
            )

    summary = {
        "status": "blocked",
        "question_count": len(rows),
        "first_review_pass_count": first_statuses.count("pass"),
        "first_review_pending_count": first_statuses.count("pending"),
        "first_review_fail_count": first_statuses.count("fail"),
        "second_review_required_count": len(second_review_ids),
        "second_review_pass_count": (
            second_required_statuses.count("pass")
        ),
        "second_review_pending_count": (
            second_required_statuses.count("pending")
        ),
        "second_review_fail_count": (
            second_required_statuses.count("fail")
        ),
        "revision_count": sum(
            question.question_version > 1 for question in questions
        ),
        "rejected_count": sum(
            row["first_status"].strip() == "fail"
            or (
                row["question_id"] in second_review_ids
                and row["second_status"].strip() == "fail"
            )
            for row in rows
        ),
        "encoding": encoding_summary,
        "draft_dataset_sha256": _sha256_file(draft_dataset_path),
        "evidence_group_sha256": _sha256_file(evidence_groups_path),
        "review_csv_sha256": _sha256_file(reviewed_csv_path),
    }
    ready = (
        summary["first_review_pass_count"] == 400
        and summary["first_review_pending_count"] == 0
        and summary["first_review_fail_count"] == 0
        and summary["second_review_pass_count"] == 40
        and summary["second_review_pending_count"] == 0
        and summary["second_review_fail_count"] == 0
    )
    if ready:
        approved_records = [
            question.model_copy(
                update={"review_status": "approved"}
            ).model_dump(mode="json")
            for question in sorted(
                questions,
                key=lambda item: item.question_id,
            )
        ]
        output_dataset_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        _write_jsonl(output_dataset_path, approved_records)
        summary["status"] = "ready"
        summary["output_dataset_sha256"] = _sha256_file(
            output_dataset_path
        )
    _write_json(summary_path, summary)
    return summary
