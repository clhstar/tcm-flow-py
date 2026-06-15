import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from experiments.rag_v1_5.schema import (
    EvidenceUnit,
    FormalEvidenceGroup,
    PilotQuestion,
)


FORMAL_BOOKS = ("shang_han_lun", "jin_gui_yao_lue")
FORMAL_SPLITS = ("formal_dev", "formal_test")
FORMAL_PER_BOOK_SPLIT = {
    "single_clause_fact": 30,
    "formula_composition_or_use": 20,
    "source_location": 10,
    "multi_evidence": 20,
    "unanswerable": 20,
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_jsonl(path: Path, model_type) -> list:
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


def _normalize_question_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).replace("？", "?")


def _load_json_object(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"排除清单不是合法 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("排除清单顶层必须为 JSON object")
    return payload


def _collect_excluded_ids(payload: object, suffix: str) -> set[str]:
    collected = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.endswith(suffix) and isinstance(value, list):
                collected.update(
                    item for item in value if isinstance(item, str)
                )
            else:
                collected.update(_collect_excluded_ids(value, suffix))
    elif isinstance(payload, list):
        for value in payload:
            collected.update(_collect_excluded_ids(value, suffix))
    return collected


def _load_prior_questions(
    paths: tuple[Path, ...],
) -> list[PilotQuestion]:
    questions = []
    for path in paths:
        if path.is_file():
            questions.extend(_read_jsonl(path, PilotQuestion))
    return questions


def validate_formal_dataset(
    *,
    dataset_path: Path,
    evidence_path: Path,
    evidence_groups_path: Path,
    exclusions_path: Path,
    prior_dataset_paths: tuple[Path, ...],
) -> dict:
    questions = _read_jsonl(dataset_path, PilotQuestion)
    groups = _read_jsonl(evidence_groups_path, FormalEvidenceGroup)
    evidence = _read_jsonl(evidence_path, EvidenceUnit)
    exclusions = _load_json_object(exclusions_path)
    prior_questions = _load_prior_questions(prior_dataset_paths)

    question_ids = [question.question_id for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Formal 数据集存在重复 question_id")

    group_ids = [group.group_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Formal Evidence Group 存在重复 group_id")
    groups_by_id = {group.group_id: group for group in groups}

    evidence_by_id = {
        item.evidence_id: item for item in evidence
    }
    if len(evidence_by_id) != len(evidence):
        raise ValueError("Evidence Tree 存在重复 evidence_id")
    clause_books: dict[str, set[str]] = {}
    for item in evidence:
        clause_books.setdefault(item.clause_id, set()).add(item.book_id)

    normalized_questions = [
        _normalize_question_text(question.question)
        for question in questions
    ]
    duplicate_question_count = (
        len(normalized_questions) - len(set(normalized_questions))
    )
    if duplicate_question_count:
        raise ValueError("Formal 问题文本归一化后存在重复")

    prior_normalized_questions = {
        _normalize_question_text(question.question)
        for question in prior_questions
    }
    duplicated_prior_questions = (
        set(normalized_questions) & prior_normalized_questions
    )
    if duplicated_prior_questions:
        raise ValueError("Formal 问题文本与 Smoke/Pilot 重复")

    excluded_group_ids = _collect_excluded_ids(exclusions, "_group_ids")
    excluded_evidence_ids = _collect_excluded_ids(
        exclusions,
        "_evidence_ids",
    )
    excluded_clause_ids = _collect_excluded_ids(exclusions, "_clause_ids")
    excluded_evidence_ids.update(
        evidence_id
        for question in prior_questions
        for evidence_id in question.gold_evidence_ids
    )
    excluded_clause_ids.update(
        clause_id
        for question in prior_questions
        for clause_id in question.gold_clause_ids
    )

    quota = {
        book: {
            split: {
                question_type: 0
                for question_type in FORMAL_PER_BOOK_SPLIT
            }
            for split in FORMAL_SPLITS
        }
        for book in FORMAL_BOOKS
    }
    split_counts = Counter()
    book_counts = Counter()
    referenced_group_ids = []
    answerable_anchor_owners: dict[str, str] = {}
    answerable_anchor_splits: dict[str, str] = {}
    clauses_by_split = {split: set() for split in FORMAL_SPLITS}
    prior_overlap_ids = set()

    for question in questions:
        missing_metadata = {
            "split",
            "evidence_group_id",
            "question_version",
        } - question.model_fields_set
        if missing_metadata:
            raise ValueError(
                f"{question.question_id} 缺少 Formal 元数据: "
                f"{sorted(missing_metadata)}"
            )
        if question.split not in FORMAL_SPLITS:
            raise ValueError(
                f"{question.question_id} split 必须为 formal_dev/test"
            )
        if question.book_scope not in FORMAL_BOOKS:
            raise ValueError(
                f"{question.question_id} book_scope 禁止 both"
            )
        if not question.evidence_group_id:
            raise ValueError(
                f"{question.question_id} 缺少 Formal 元数据"
            )
        group = groups_by_id.get(question.evidence_group_id)
        if group is None:
            raise ValueError(
                f"{question.question_id} 找不到 Formal Evidence Group"
            )
        referenced_group_ids.append(group.group_id)
        if (
            group.split != question.split
            or group.book_scope != question.book_scope
            or group.question_type != question.question_type
        ):
            raise ValueError(
                f"{question.question_id} 与 Evidence Group 契约不一致"
            )

        quota[question.book_scope][question.split][
            question.question_type
        ] += 1
        split_counts[question.split] += 1
        book_counts[question.book_scope] += 1

        relevance_ids = set(question.graded_relevance)
        allowed_relevance_ids = set(
            question.gold_evidence_ids + question.gold_clause_ids
        )
        if not relevance_ids <= allowed_relevance_ids:
            raise ValueError(
                f"{question.question_id} graded_relevance 包含非 gold ID"
            )

        if group.group_id in excluded_group_ids:
            prior_overlap_ids.add(group.group_id)

        if question.answerable:
            if not set(question.gold_evidence_ids) <= set(
                group.anchor_evidence_ids
            ):
                raise ValueError(
                    f"{question.question_id} gold Evidence 越出 anchor"
                )
            if not set(question.gold_clause_ids) <= set(
                group.anchor_clause_ids
            ):
                raise ValueError(
                    f"{question.question_id} gold clause 越出 anchor"
                )
            for evidence_id in question.gold_evidence_ids:
                item = evidence_by_id.get(evidence_id)
                if item is None:
                    raise ValueError(
                        f"{question.question_id} 缺少 gold Evidence"
                    )
                if item.book_id != question.book_scope:
                    raise ValueError(
                        f"{question.question_id} gold Evidence "
                        "必须来自同一本书"
                    )
            for clause_id in question.gold_clause_ids:
                clauses_by_split[question.split].add(clause_id)
                first_split = answerable_anchor_splits.setdefault(
                    clause_id,
                    question.split,
                )
                if first_split != question.split:
                    raise ValueError("Formal dev/test 存在 clause 泄漏")
                owner = answerable_anchor_owners.setdefault(
                    clause_id,
                    question.question_id,
                )
                if owner != question.question_id:
                    raise ValueError(
                        "同一 answerable anchor clause 被多个问题复用"
                    )
                if clause_books.get(clause_id) != {question.book_scope}:
                    raise ValueError(
                        f"{question.question_id} gold clause "
                        "必须来自同一本书"
                    )
            for support_span in question.support_spans:
                if not any(
                    support_span in evidence_by_id[
                        evidence_id
                    ].normalized_text
                    for evidence_id in question.gold_evidence_ids
                ):
                    raise ValueError(
                        f"{question.question_id} support span "
                        "不属于 gold Evidence"
                    )
            if question.question_type == "multi_evidence":
                if len(set(question.gold_clause_ids)) < 2:
                    raise ValueError(
                        "multi_evidence 至少需要 2 个 gold clause"
                    )
                multi_books = {
                    evidence_by_id[evidence_id].book_id
                    for evidence_id in question.gold_evidence_ids
                }
                if multi_books != {question.book_scope}:
                    raise ValueError(
                        "multi_evidence gold clauses 必须来自同一本书"
                    )
        elif group.anchor_evidence_ids or group.anchor_clause_ids:
            raise ValueError("无答案 Evidence Group 不得包含 anchor")
        elif len(group.absence_queries) < 2:
            raise ValueError("无答案问题至少需要 2 条 absence query")

        prior_overlap_ids.update(
            set(question.gold_evidence_ids) & excluded_evidence_ids
        )
        prior_overlap_ids.update(
            set(question.gold_clause_ids) & excluded_clause_ids
        )

    if prior_overlap_ids:
        raise ValueError(
            "Formal 与 Smoke/Pilot Evidence、Clause 或 Group 重叠"
        )

    cross_split_clause_ids = (
        clauses_by_split["formal_dev"]
        & clauses_by_split["formal_test"]
    )
    if cross_split_clause_ids:
        raise ValueError("Formal dev/test 存在 clause 泄漏")

    if len(referenced_group_ids) != len(set(referenced_group_ids)):
        raise ValueError("同一 Formal group_id 被多个问题引用")
    if set(referenced_group_ids) != set(groups_by_id):
        raise ValueError("Formal 问题与 Evidence Group 必须一一对应")

    quota_mismatches = {
        f"{book}/{split}/{question_type}": actual
        for book, split_counts_by_type in quota.items()
        for split, type_counts in split_counts_by_type.items()
        for question_type, actual in type_counts.items()
        if actual != FORMAL_PER_BOOK_SPLIT[question_type]
    }
    if quota_mismatches:
        raise ValueError(f"Formal 固定配额错误: {quota_mismatches}")

    answerable_count = sum(question.answerable for question in questions)
    unanswerable_count = len(questions) - answerable_count
    if (
        len(questions) != 400
        or answerable_count != 320
        or unanswerable_count != 80
    ):
        raise ValueError("Formal 数据集必须满足 400/320/80")

    return {
        "status": "ready",
        "question_count": len(questions),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "approved_count": sum(
            question.review_status == "approved"
            for question in questions
        ),
        "split_counts": dict(split_counts),
        "book_counts": dict(book_counts),
        "quota_by_book_split_type": quota,
        "duplicate_question_count": duplicate_question_count,
        "prior_overlap_count": 0,
        "cross_split_clause_overlap_count": 0,
        "answerable_anchor_clause_count": len(answerable_anchor_owners),
        "dataset_sha256": _sha256_file(dataset_path),
        "evidence_sha256": _sha256_file(evidence_path),
        "evidence_group_sha256": _sha256_file(evidence_groups_path),
        "exclusions_sha256": _sha256_file(exclusions_path),
    }
