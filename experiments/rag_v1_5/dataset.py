import csv
import hashlib
import io
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from experiments.rag_v1_5.metrics import evaluate_rankings
from experiments.rag_v1_5.schema import (
    EvidenceUnit,
    PilotEvidenceGroup,
    PilotQuestion,
    RetrievalHit,
)


DatasetProfile = Literal["auto", "smoke", "pilot", "generic"]
SmokeRetriever = Callable[[PilotQuestion], list[RetrievalHit]]
PILOT_BOOKS = ("shang_han_lun", "jin_gui_yao_lue")
PILOT_QUESTION_TYPES = (
    "single_clause_fact",
    "formula_composition_or_use",
    "source_location",
    "multi_evidence",
    "unanswerable",
)
PILOT_PER_CELL = 4
PILOT_SECOND_REVIEW_SEED = 20260614
PILOT_REVIEW_IMMUTABLE_FIELDS = (
    "question_id",
    "content_sha256",
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
PILOT_REVIEW_FIRST_FIELDS = (
    "first_status",
    "first_decision",
    "first_comment",
    "first_reviewer",
    "first_reviewed_at",
)
PILOT_REVIEW_SECOND_FIELDS = (
    "second_status",
    "second_decision",
    "second_comment",
    "second_reviewer",
    "second_reviewed_at",
)
PILOT_REVIEW_FIELDS = (
    PILOT_REVIEW_IMMUTABLE_FIELDS
    + PILOT_REVIEW_FIRST_FIELDS
    + PILOT_REVIEW_SECOND_FIELDS
)
PILOT_REVIEW_DECISIONS = {
    "correct",
    "question_unnatural",
    "unsupported_answer",
    "gold_id_error",
    "relevance_error",
    "clinical_scope",
    "duplicate",
    "other",
}
PILOT_REVIEW_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "cp936",
    "gb18030",
)
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
    if dataset_path.stem.startswith("pilot-40"):
        return "pilot"
    return "generic"


def _normalize_question_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).replace("？", "?")


def _load_pilot_evidence_groups(
    evidence_groups_path: Path,
) -> dict[str, PilotEvidenceGroup]:
    groups = _read_jsonl(evidence_groups_path, PilotEvidenceGroup)
    groups_by_id = {group.group_id: group for group in groups}
    if len(groups_by_id) != len(groups):
        raise ValueError("Pilot Evidence Group 存在重复 group_id")
    return groups_by_id


def _validate_pilot_profile(
    *,
    questions: list[PilotQuestion],
    evidence_by_id: dict[str, EvidenceUnit],
    evidence_groups_path: Path | None,
) -> dict:
    if evidence_groups_path is None:
        raise ValueError("Pilot profile 必须显式提供 Evidence Group 文件")
    groups_by_id = _load_pilot_evidence_groups(evidence_groups_path)
    clause_books: dict[str, set[str]] = {}
    for evidence in evidence_by_id.values():
        clause_books.setdefault(evidence.clause_id, set()).add(
            evidence.book_id
        )

    normalized_questions = [
        _normalize_question_text(question.question)
        for question in questions
    ]
    duplicate_question_count = (
        len(normalized_questions) - len(set(normalized_questions))
    )
    if duplicate_question_count:
        raise ValueError("Pilot 问题文本归一化后存在重复")

    quota = {
        book: {question_type: 0 for question_type in PILOT_QUESTION_TYPES}
        for book in PILOT_BOOKS
    }
    group_references = []
    for question in questions:
        required_fields = {
            "split",
            "evidence_group_id",
            "question_version",
        }
        missing_fields = required_fields - question.model_fields_set
        if missing_fields:
            raise ValueError(
                f"{question.question_id} 缺少 Pilot 元数据: "
                f"{sorted(missing_fields)}"
            )
        if question.split != "pilot":
            raise ValueError(f"{question.question_id} split 必须为 pilot")
        if question.book_scope == "both":
            raise ValueError(
                f"{question.question_id} Pilot book_scope 禁止 both"
            )
        if question.evidence_group_id is None:
            raise ValueError(
                f"{question.question_id} 缺少 evidence_group_id"
            )
        if question.book_scope not in quota:
            raise ValueError(
                f"{question.question_id} Pilot book_scope 非法"
            )
        quota[question.book_scope][question.question_type] += 1
        group_references.append(question.evidence_group_id)

        group = groups_by_id.get(question.evidence_group_id)
        if group is None:
            raise ValueError(
                f"{question.question_id} 找不到 Evidence Group: "
                f"{question.evidence_group_id}"
            )
        if (
            group.book_scope != question.book_scope
            or group.question_type != question.question_type
        ):
            raise ValueError(
                f"{question.question_id} 与 Evidence Group "
                "书籍或问题类型不一致"
            )

        relevance_ids = set(question.graded_relevance)
        allowed_relevance_ids = set(
            question.gold_evidence_ids + question.gold_clause_ids
        )
        if not relevance_ids <= allowed_relevance_ids:
            raise ValueError(
                f"{question.question_id} graded_relevance "
                "包含非 gold ID"
            )

        if question.answerable:
            if not set(question.gold_evidence_ids) <= set(
                group.anchor_evidence_ids
            ):
                raise ValueError(
                    f"{question.question_id} gold Evidence "
                    "不属于 Evidence Group anchor"
                )
            if not set(question.gold_clause_ids) <= set(
                group.anchor_clause_ids
            ):
                raise ValueError(
                    f"{question.question_id} gold clause "
                    "不属于 Evidence Group anchor"
                )
        elif len(group.absence_queries) < 2:
            raise ValueError(
                f"{question.question_id} 无答案组至少需要 2 条 "
                "absence_queries"
            )

        if question.question_type == "multi_evidence":
            if len(set(question.gold_clause_ids)) < 2:
                raise ValueError(
                    f"{question.question_id} multi_evidence "
                    "至少需要 2 个 gold clause"
                )
            gold_clause_books = set()
            for clause_id in question.gold_clause_ids:
                gold_clause_books.update(clause_books.get(clause_id, set()))
            if gold_clause_books != {question.book_scope}:
                raise ValueError(
                    f"{question.question_id} multi_evidence "
                    "gold clause 必须来自同一本书"
                )

    duplicate_group_references = [
        group_id
        for group_id, count in Counter(group_references).items()
        if count > 1
    ]
    if duplicate_group_references:
        raise ValueError(
            "Pilot 问题重复引用 Evidence Group: "
            f"{duplicate_group_references}"
        )
    if set(group_references) != set(groups_by_id):
        raise ValueError("Pilot 问题与 Evidence Group 必须一一对应")

    invalid_quota = {
        f"{book}/{question_type}": count
        for book, type_counts in quota.items()
        for question_type, count in type_counts.items()
        if count != PILOT_PER_CELL
    }
    if invalid_quota:
        raise ValueError(f"Pilot 书籍 × 类型配额错误: {invalid_quota}")

    answerable_count = sum(question.answerable for question in questions)
    unanswerable_count = len(questions) - answerable_count
    if (
        len(questions) != 40
        or answerable_count != 32
        or unanswerable_count != 8
    ):
        raise ValueError("Pilot 数据集必须为 40/32/8 固定分布")

    return {
        "quota_by_book_and_type": quota,
        "duplicate_question_count": duplicate_question_count,
        "multi_evidence_count": sum(
            question.question_type == "multi_evidence"
            for question in questions
        ),
        "evidence_group_sha256": _sha256_file(evidence_groups_path),
    }


def validate_dataset(
    *,
    dataset_path: Path,
    evidence_path: Path,
    profile: DatasetProfile = "auto",
    evidence_groups_path: Path | None = None,
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
    pilot_summary = {}
    if resolved_profile == "pilot":
        pilot_summary = _validate_pilot_profile(
            questions=questions,
            evidence_by_id=evidence_by_id,
            evidence_groups_path=evidence_groups_path,
        )

    clause_ids = {
        evidence.clause_id for evidence in evidence_by_id.values()
    }
    for question in questions:
        if not question.answerable:
            continue
        relevance_ids = set(question.graded_relevance)
        allowed_relevance_ids = set(
            question.gold_evidence_ids + question.gold_clause_ids
        )
        if not relevance_ids <= allowed_relevance_ids:
            raise ValueError(
                f"{question.question_id} graded_relevance "
                "包含非 gold ID"
            )
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
        **pilot_summary,
    }


def _stable_sample(
    records: list,
    *,
    count: int,
    seed: int,
    stratum: str,
) -> list:
    shuffled = list(records)
    random.Random(f"{seed}:{stratum}").shuffle(shuffled)
    return shuffled[:count]


def _pilot_group_id(
    book: str,
    question_type: str,
    index: int,
) -> str:
    return f"pilot-{book}-{question_type}-{index:02d}"


def sample_pilot_evidence_groups(
    *,
    evidence_path: Path,
    smoke_dataset_path: Path,
    output_path: Path,
    exclusions_path: Path,
    candidate_report_path: Path,
    seed: int = 20260614,
) -> dict:
    evidence_units = sorted(
        _read_jsonl(evidence_path, EvidenceUnit),
        key=lambda evidence: evidence.evidence_id,
    )
    evidence_by_id = {
        evidence.evidence_id: evidence for evidence in evidence_units
    }
    if len(evidence_by_id) != len(evidence_units):
        raise ValueError("Evidence Tree 存在重复 evidence_id")

    smoke_questions = load_dataset(smoke_dataset_path)
    smoke_gold_evidence_ids = sorted(
        {
            evidence_id
            for question in smoke_questions
            for evidence_id in question.gold_evidence_ids
        }
    )
    smoke_gold_clause_ids = {
        clause_id
        for question in smoke_questions
        for clause_id in question.gold_clause_ids
    }
    for evidence_id in smoke_gold_evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is not None:
            smoke_gold_clause_ids.add(evidence.clause_id)
    smoke_gold_clause_ids = sorted(smoke_gold_clause_ids)

    evidence_by_clause: dict[str, list[EvidenceUnit]] = defaultdict(list)
    clause_units: dict[str, EvidenceUnit] = {}
    for evidence in evidence_units:
        evidence_by_clause[evidence.clause_id].append(evidence)
        if evidence.content_type == "clause":
            clause_units[evidence.clause_id] = evidence

    excluded_clauses = set(smoke_gold_clause_ids)
    available_by_book = {
        book: [
            clause
            for clause in sorted(
                clause_units.values(),
                key=lambda item: (
                    item.chapter_id,
                    item.clause_number
                    if item.clause_number is not None
                    else 10**9,
                    item.clause_id,
                ),
            )
            if (
                clause.book_id == book
                and clause.clause_id not in excluded_clauses
            )
        ]
        for book in PILOT_BOOKS
    }
    used_clause_ids: set[str] = set()
    groups: list[PilotEvidenceGroup] = []
    strata: dict[str, dict] = {}

    def add_single_clause_groups(
        *,
        book: str,
        question_type: str,
        candidates: list[EvidenceUnit],
        priority_candidates: list[EvidenceUnit] | None = None,
    ) -> None:
        stratum = f"{book}/{question_type}"
        unused_candidates = [
            candidate
            for candidate in candidates
            if candidate.clause_id not in used_clause_ids
        ]
        selected = []
        if priority_candidates:
            unused_priority = [
                candidate
                for candidate in priority_candidates
                if candidate.clause_id not in used_clause_ids
            ]
            selected.extend(
                _stable_sample(
                    unused_priority,
                    count=min(PILOT_PER_CELL, len(unused_priority)),
                    seed=seed,
                    stratum=f"{stratum}:priority",
                )
            )
        selected_ids = {
            candidate.clause_id for candidate in selected
        }
        if len(selected) < PILOT_PER_CELL:
            fallback_candidates = [
                candidate
                for candidate in unused_candidates
                if candidate.clause_id not in selected_ids
            ]
            selected.extend(
                _stable_sample(
                    fallback_candidates,
                    count=PILOT_PER_CELL - len(selected),
                    seed=seed,
                    stratum=f"{stratum}:fallback",
                )
            )
        if len(selected) != PILOT_PER_CELL:
            raise ValueError(
                f"Pilot 候选不足: {stratum} 需要 "
                f"{PILOT_PER_CELL}，实际 {len(unused_candidates)}"
            )
        for index, clause in enumerate(selected, start=1):
            clause_evidence = evidence_by_clause[clause.clause_id]
            if question_type == "formula_composition_or_use":
                anchor_evidence_ids = sorted(
                    evidence.evidence_id
                    for evidence in clause_evidence
                    if evidence.content_type
                    in {"formula", "ingredients", "preparation"}
                )
                selection_reason = (
                    "包含 formula、ingredients 或 preparation Child"
                )
            elif question_type == "source_location":
                note_ids = sorted(
                    evidence.evidence_id
                    for evidence in clause_evidence
                    if evidence.content_type == "note"
                )
                anchor_evidence_ids = note_ids or [clause.evidence_id]
                selection_reason = (
                    "包含 note Child 或具备明确篇章与条文定位"
                )
            else:
                anchor_evidence_ids = [clause.evidence_id]
                selection_reason = "条文正文满足事实型问题候选条件"
            groups.append(
                PilotEvidenceGroup(
                    group_id=_pilot_group_id(
                        book,
                        question_type,
                        index,
                    ),
                    split="pilot",
                    book_scope=book,
                    question_type=question_type,
                    anchor_evidence_ids=anchor_evidence_ids,
                    anchor_clause_ids=[clause.clause_id],
                    selection_seed=seed,
                    selection_reason=selection_reason,
                )
            )
            used_clause_ids.add(clause.clause_id)
        strata[stratum] = {
            "candidate_count": len(unused_candidates),
            "selected_count": len(selected),
            "rejection_counts": {
                "smoke_excluded_clause_count": sum(
                    clause.book_id == book
                    for clause in clause_units.values()
                    if clause.clause_id in excluded_clauses
                ),
                "already_selected_clause_count": (
                    len(candidates) - len(unused_candidates)
                ),
            },
        }

    for book in PILOT_BOOKS:
        book_candidates = available_by_book[book]
        formula_candidates = [
            clause
            for clause in book_candidates
            if any(
                evidence.content_type
                in {"formula", "ingredients", "preparation"}
                for evidence in evidence_by_clause[clause.clause_id]
            )
        ]
        add_single_clause_groups(
            book=book,
            question_type="formula_composition_or_use",
            candidates=formula_candidates,
        )

        source_candidates = [
            clause
            for clause in book_candidates
            if (
                any(
                    evidence.content_type == "note"
                    for evidence in evidence_by_clause[clause.clause_id]
                )
                or (
                    bool(clause.chapter_title.strip())
                    and clause.clause_number is not None
                )
            )
        ]
        note_candidates = [
            clause
            for clause in source_candidates
            if any(
                evidence.content_type == "note"
                for evidence in evidence_by_clause[clause.clause_id]
            )
        ]
        add_single_clause_groups(
            book=book,
            question_type="source_location",
            candidates=source_candidates,
            priority_candidates=note_candidates,
        )

        fact_candidates = [
            clause
            for clause in book_candidates
            if (
                len(clause.normalized_text.strip()) >= 8
                and re.search(
                    r"[\w\u4e00-\u9fff]",
                    clause.normalized_text,
                )
            )
        ]
        add_single_clause_groups(
            book=book,
            question_type="single_clause_fact",
            candidates=fact_candidates,
        )

        remaining_by_chapter: dict[str, list[EvidenceUnit]] = defaultdict(
            list
        )
        for clause in book_candidates:
            if clause.clause_id not in used_clause_ids:
                remaining_by_chapter[clause.chapter_id].append(clause)
        pair_candidates = []
        for chapter_id in sorted(remaining_by_chapter):
            chapter_clauses = sorted(
                remaining_by_chapter[chapter_id],
                key=lambda item: (
                    item.clause_number
                    if item.clause_number is not None
                    else 10**9,
                    item.clause_id,
                ),
            )
            pair_candidates.extend(
                zip(chapter_clauses, chapter_clauses[1:])
            )
        shuffled_pairs = list(pair_candidates)
        random.Random(
            f"{seed}:{book}:multi_evidence"
        ).shuffle(shuffled_pairs)
        selected_pairs = []
        paired_clause_ids: set[str] = set()
        for first, second in shuffled_pairs:
            pair_ids = {first.clause_id, second.clause_id}
            if pair_ids & paired_clause_ids:
                continue
            selected_pairs.append((first, second))
            paired_clause_ids.update(pair_ids)
            if len(selected_pairs) == PILOT_PER_CELL:
                break
        stratum = f"{book}/multi_evidence"
        if len(selected_pairs) != PILOT_PER_CELL:
            raise ValueError(
                f"Pilot 候选不足: {stratum} 需要 "
                f"{PILOT_PER_CELL}，实际 {len(selected_pairs)}"
            )
        for index, (first, second) in enumerate(
            selected_pairs,
            start=1,
        ):
            groups.append(
                PilotEvidenceGroup(
                    group_id=_pilot_group_id(
                        book,
                        "multi_evidence",
                        index,
                    ),
                    split="pilot",
                    book_scope=book,
                    question_type="multi_evidence",
                    anchor_evidence_ids=[
                        first.evidence_id,
                        second.evidence_id,
                    ],
                    anchor_clause_ids=[
                        first.clause_id,
                        second.clause_id,
                    ],
                    selection_seed=seed,
                    selection_reason=(
                        "同篇章相邻条文组合候选，仅用于事实型多证据问题"
                    ),
                )
            )
            used_clause_ids.update(
                {first.clause_id, second.clause_id}
            )
        strata[stratum] = {
            "candidate_count": len(pair_candidates),
            "selected_count": len(selected_pairs),
            "rejection_counts": {
                "overlapping_pair_count": (
                    len(pair_candidates) - len(selected_pairs)
                )
            },
        }

        unanswerable_stratum = f"{book}/unanswerable"
        for index in range(1, PILOT_PER_CELL + 1):
            groups.append(
                PilotEvidenceGroup(
                    group_id=_pilot_group_id(
                        book,
                        "unanswerable",
                        index,
                    ),
                    split="pilot",
                    book_scope=book,
                    question_type="unanswerable",
                    anchor_evidence_ids=[],
                    anchor_clause_ids=[],
                    selection_seed=seed,
                    selection_reason=(
                        "需人工填写两书无答案主题并补充 absence queries"
                    ),
                )
            )
        strata[unanswerable_stratum] = {
            "candidate_count": PILOT_PER_CELL,
            "selected_count": PILOT_PER_CELL,
            "rejection_counts": {},
        }

    book_order = {book: index for index, book in enumerate(PILOT_BOOKS)}
    type_order = {
        question_type: index
        for index, question_type in enumerate(PILOT_QUESTION_TYPES)
    }
    groups.sort(
        key=lambda group: (
            book_order[group.book_scope],
            type_order[group.question_type],
            group.group_id,
        )
    )
    group_records = [
        group.model_dump(mode="json") for group in groups
    ]
    pilot_anchor_evidence_ids = sorted(
        {
            evidence_id
            for group in groups
            for evidence_id in group.anchor_evidence_ids
        }
    )
    pilot_anchor_clause_ids = sorted(
        {
            clause_id
            for group in groups
            for clause_id in group.anchor_clause_ids
        }
    )
    exclusions = {
        "version": "v1.5.0",
        "selection_seed": seed,
        "smoke_gold_evidence_ids": smoke_gold_evidence_ids,
        "smoke_gold_clause_ids": smoke_gold_clause_ids,
        "pilot_group_ids": [group.group_id for group in groups],
        "pilot_anchor_evidence_ids": pilot_anchor_evidence_ids,
        "pilot_anchor_clause_ids": pilot_anchor_clause_ids,
        "future_formal_excluded_evidence_ids": sorted(
            set(smoke_gold_evidence_ids)
            | set(pilot_anchor_evidence_ids)
        ),
        "future_formal_excluded_clause_ids": sorted(
            set(smoke_gold_clause_ids)
            | set(pilot_anchor_clause_ids)
        ),
    }
    candidate_report = {
        "version": "v1.5.0",
        "selection_seed": seed,
        "selected_group_count": len(groups),
        "answerable_group_count": sum(
            group.question_type != "unanswerable" for group in groups
        ),
        "unanswerable_group_count": sum(
            group.question_type == "unanswerable" for group in groups
        ),
        "strata": strata,
    }

    for path in (output_path, exclusions_path, candidate_report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, group_records)
    _write_json(exclusions_path, exclusions)
    _write_json(candidate_report_path, candidate_report)
    return {
        "group_count": len(groups),
        "answerable_group_count": (
            candidate_report["answerable_group_count"]
        ),
        "unanswerable_group_count": (
            candidate_report["unanswerable_group_count"]
        ),
        "evidence_group_sha256": _sha256_file(output_path),
        "exclusions_sha256": _sha256_file(exclusions_path),
        "candidate_report_sha256": _sha256_file(
            candidate_report_path
        ),
    }


def _validate_pilot_review_inputs(
    *,
    questions: list[PilotQuestion],
    groups_by_id: dict[str, PilotEvidenceGroup],
) -> None:
    if len(questions) != 40:
        raise ValueError("Pilot 审核草稿必须恰好包含 40 条问题")
    if len(groups_by_id) != 40:
        raise ValueError("Pilot 审核必须恰好包含 40 个 Evidence Group")
    quota = Counter(
        (question.book_scope, question.question_type)
        for question in questions
    )
    expected_cells = {
        (book, question_type)
        for book in PILOT_BOOKS
        for question_type in PILOT_QUESTION_TYPES
    }
    if set(quota) != expected_cells or any(
        count != PILOT_PER_CELL for count in quota.values()
    ):
        raise ValueError("Pilot 审核草稿的书籍 × 类型配额错误")
    if sum(question.answerable for question in questions) != 32:
        raise ValueError("Pilot 审核草稿必须包含 32 条可回答问题")

    referenced_group_ids = []
    for question in questions:
        if (
            question.split != "pilot"
            or not question.evidence_group_id
            or question.review_status != "draft"
        ):
            raise ValueError(
                f"{question.question_id} 不是合法 Pilot 审核草稿"
            )
        group = groups_by_id.get(question.evidence_group_id)
        if group is None:
            raise ValueError(
                f"{question.question_id} 找不到 Evidence Group"
            )
        if (
            group.book_scope != question.book_scope
            or group.question_type != question.question_type
        ):
            raise ValueError(
                f"{question.question_id} 与 Evidence Group 不一致"
            )
        referenced_group_ids.append(question.evidence_group_id)
    if (
        len(set(referenced_group_ids)) != 40
        or set(referenced_group_ids) != set(groups_by_id)
    ):
        raise ValueError("Pilot 问题与 Evidence Group 必须一一对应")


def _second_review_question_ids(
    questions: list[PilotQuestion],
    *,
    seed: int,
) -> list[str]:
    selected_ids = []
    for book in PILOT_BOOKS:
        for question_type in PILOT_QUESTION_TYPES:
            candidates = sorted(
                (
                    question
                    for question in questions
                    if (
                        question.book_scope == book
                        and question.question_type == question_type
                    )
                ),
                key=lambda question: question.question_id,
            )
            selected = _stable_sample(
                candidates,
                count=1,
                seed=seed,
                stratum=f"second-review:{book}/{question_type}",
            )
            if len(selected) != 1:
                raise ValueError(
                    f"二审分层候选不足: {book}/{question_type}"
                )
            selected_ids.append(selected[0].question_id)
    return selected_ids


def _review_content_payload(
    question: PilotQuestion,
    *,
    second_review_required: bool,
) -> dict:
    return {
        "question_id": question.question_id,
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


def _review_content_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest().upper()


def _review_immutable_row(
    question: PilotQuestion,
    *,
    second_review_required: bool,
) -> dict[str, str]:
    payload = _review_content_payload(
        question,
        second_review_required=second_review_required,
    )
    return {
        "question_id": question.question_id,
        "content_sha256": _review_content_sha256(payload),
        "book_scope": question.book_scope,
        "question_type": question.question_type,
        "answerable": str(question.answerable).lower(),
        "question": question.question,
        "reference_answer": question.reference_answer,
        "gold_evidence_ids": "|".join(question.gold_evidence_ids),
        "gold_clause_ids": "|".join(question.gold_clause_ids),
        "graded_relevance": json.dumps(
            question.graded_relevance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "support_spans": "|".join(question.support_spans),
        "evidence_group_id": question.evidence_group_id or "",
        "question_version": str(question.question_version),
        "second_review_required": str(
            second_review_required
        ).lower(),
    }


def _decode_pilot_review_csv(path: Path) -> tuple[str, str, bytes]:
    raw_bytes = path.read_bytes()
    errors = []
    for encoding in PILOT_REVIEW_ENCODINGS:
        try:
            return raw_bytes.decode(encoding), encoding, raw_bytes
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")
    raise ValueError(
        f"无法解码 Pilot 审核 CSV: {path}; " + "; ".join(errors)
    )


def _parse_pilot_review_csv(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != PILOT_REVIEW_FIELDS:
        raise ValueError("Pilot 审核 CSV 列与固定模板不一致")
    rows = list(reader)
    question_ids = [row["question_id"] for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Pilot 审核 CSV 存在重复 question_id")
    return rows


def _read_existing_pilot_review(
    path: Path,
) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    text, _, _ = _decode_pilot_review_csv(path)
    return {
        row["question_id"]: row
        for row in _parse_pilot_review_csv(text)
    }


def _write_pilot_review_csv(
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
            fieldnames=PILOT_REVIEW_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_pilot_review(
    *,
    draft_dataset_path: Path,
    evidence_groups_path: Path,
    review_csv_path: Path,
    second_review_seed: int = PILOT_SECOND_REVIEW_SEED,
) -> dict:
    questions = load_dataset(draft_dataset_path)
    groups_by_id = _load_pilot_evidence_groups(evidence_groups_path)
    _validate_pilot_review_inputs(
        questions=questions,
        groups_by_id=groups_by_id,
    )
    second_review_ids = set(
        _second_review_question_ids(
            questions,
            seed=second_review_seed,
        )
    )
    existing_rows = _read_existing_pilot_review(review_csv_path)
    rows = []
    inherited_count = 0
    for question in questions:
        immutable_row = _review_immutable_row(
            question,
            second_review_required=(
                question.question_id in second_review_ids
            ),
        )
        row = {
            **immutable_row,
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
            == immutable_row["content_sha256"]
        ):
            for field in (
                PILOT_REVIEW_FIRST_FIELDS
                + PILOT_REVIEW_SECOND_FIELDS
            ):
                row[field] = existing[field]
            inherited_count += 1
        rows.append(row)
    _write_pilot_review_csv(review_csv_path, rows)
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
    if decision and decision not in PILOT_REVIEW_DECISIONS:
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


def _review_immutable_values_equal(
    *,
    field: str,
    reviewed_value: str,
    expected_value: str,
) -> bool:
    if field in {"answerable", "second_review_required"}:
        return reviewed_value.strip().lower() == expected_value
    return reviewed_value == expected_value


def _normalize_review_csv_encoding(
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
    _write_pilot_review_csv(reviewed_csv_path, rows)
    normalized_text, normalized_encoding, _ = (
        _decode_pilot_review_csv(reviewed_csv_path)
    )
    normalized_rows = _parse_pilot_review_csv(normalized_text)
    unicode_equivalent = normalized_rows == rows
    if normalized_encoding != "utf-8-sig" or not unicode_equivalent:
        raise ValueError("Pilot 审核 CSV 编码规范化未保持 Unicode 等价")
    return {
        "detected_encoding": detected_encoding,
        "converted": True,
        "original_sha256": original_sha256,
        "backup_path": str(backup_path.resolve()),
        "backup_sha256": _sha256_file(backup_path),
        "normalized_sha256": _sha256_file(reviewed_csv_path),
        "unicode_equivalent": unicode_equivalent,
    }


def import_pilot_review(
    *,
    draft_dataset_path: Path,
    evidence_groups_path: Path,
    reviewed_csv_path: Path,
    output_dataset_path: Path,
    summary_path: Path,
) -> dict:
    questions = load_dataset(draft_dataset_path)
    groups_by_id = _load_pilot_evidence_groups(evidence_groups_path)
    _validate_pilot_review_inputs(
        questions=questions,
        groups_by_id=groups_by_id,
    )
    second_review_ids = set(
        _second_review_question_ids(
            questions,
            seed=PILOT_SECOND_REVIEW_SEED,
        )
    )
    text, detected_encoding, original_bytes = (
        _decode_pilot_review_csv(reviewed_csv_path)
    )
    rows = _parse_pilot_review_csv(text)
    if len(rows) != 40:
        raise ValueError("Pilot 审核 CSV 必须恰好包含 40 行")
    encoding_summary = _normalize_review_csv_encoding(
        reviewed_csv_path=reviewed_csv_path,
        rows=rows,
        detected_encoding=detected_encoding,
        original_bytes=original_bytes,
    )

    questions_by_id = {
        question.question_id: question for question in questions
    }
    if set(row["question_id"] for row in rows) != set(questions_by_id):
        raise ValueError("Pilot 审核 CSV question_id 与草稿不一致")
    first_statuses = []
    second_required_statuses = []
    for row in rows:
        question = questions_by_id[row["question_id"]]
        expected = _review_immutable_row(
            question,
            second_review_required=(
                question.question_id in second_review_ids
            ),
        )
        for field in PILOT_REVIEW_IMMUTABLE_FIELDS:
            if not _review_immutable_values_equal(
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
        summary["first_review_pass_count"] == 40
        and summary["first_review_pending_count"] == 0
        and summary["first_review_fail_count"] == 0
        and summary["second_review_pass_count"] == 10
        and summary["second_review_pending_count"] == 0
        and summary["second_review_fail_count"] == 0
    )
    if ready:
        approved_records = [
            question.model_copy(
                update={"review_status": "approved"}
            ).model_dump(mode="json")
            for question in questions
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
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(summary_path, summary)
    return summary


def _load_json_object(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} 不是合法 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 顶层必须是 JSON object: {path}")
    return payload


def _manifest_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _hashed_input(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _manifest_path(path),
        "sha256": _sha256_file(path),
    }


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9A-Fa-f]{64}", value) is None
    ):
        raise ValueError(f"{label} 缺少合法 SHA256")
    return value.upper()


def _validate_pilot_review_summary(
    *,
    review_summary: dict,
    dataset_sha256: str,
    evidence_group_sha256: str,
) -> None:
    required_counts = {
        "question_count": 40,
        "first_review_pass_count": 40,
        "first_review_pending_count": 0,
        "first_review_fail_count": 0,
        "second_review_required_count": 10,
        "second_review_pass_count": 10,
        "second_review_pending_count": 0,
        "second_review_fail_count": 0,
        "rejected_count": 0,
    }
    if review_summary.get("status") != "ready":
        raise ValueError("Pilot review summary 必须为 ready")
    mismatches = {
        key: review_summary.get(key)
        for key, expected in required_counts.items()
        if review_summary.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Pilot review summary 审核门禁未通过: {mismatches}")
    revision_count = review_summary.get("revision_count")
    if not isinstance(revision_count, int) or revision_count < 0:
        raise ValueError("Pilot review summary 缺少 revision_count")
    if (
        review_summary.get("output_dataset_sha256")
        != dataset_sha256
    ):
        raise ValueError("Pilot review summary 的 dataset 哈希不一致")
    if (
        review_summary.get("evidence_group_sha256")
        != evidence_group_sha256
    ):
        raise ValueError("Pilot review summary 的 Evidence Group 哈希不一致")
    _require_sha256(
        review_summary.get("review_csv_sha256"),
        label="Pilot review CSV",
    )


def _validate_pilot_exclusions(
    *,
    exclusions: dict,
    groups_by_id: dict[str, PilotEvidenceGroup],
) -> None:
    expected = {
        "pilot_group_ids": sorted(groups_by_id),
        "pilot_anchor_evidence_ids": sorted(
            {
                evidence_id
                for group in groups_by_id.values()
                for evidence_id in group.anchor_evidence_ids
            }
        ),
        "pilot_anchor_clause_ids": sorted(
            {
                clause_id
                for group in groups_by_id.values()
                for clause_id in group.anchor_clause_ids
            }
        ),
    }
    mismatches = {}
    for key, expected_values in expected.items():
        actual_values = exclusions.get(key)
        valid = (
            isinstance(actual_values, list)
            and len(actual_values) == len(set(actual_values))
            and set(actual_values) == set(expected_values)
        )
        if not valid:
            mismatches[key] = {
                "expected_count": len(expected_values),
                "actual_count": (
                    len(actual_values)
                    if isinstance(actual_values, list)
                    else None
                ),
            }
    if mismatches:
        raise ValueError(f"Pilot 排除清单与实际数据不一致: {mismatches}")


def _validate_pilot_upstream_manifests(
    *,
    evidence_sha256: str,
    chunk_manifest: dict,
    chunk_manifest_sha256: str,
    quality_gate: dict,
    quality_gate_sha256: str,
    index_manifest: dict,
    model_manifest_sha256: str,
    config_sha256: str,
    smoke_manifest: dict,
    index_manifest_sha256: str,
) -> None:
    if chunk_manifest.get("evidence_sha256") != evidence_sha256:
        raise ValueError("Chunk Manifest 与当前 Evidence 哈希不一致")
    if quality_gate.get("status") != "ready":
        raise ValueError("Pilot 冻结要求 Quality Gate 为 ready")
    if (
        quality_gate.get("evidence_sha256") != evidence_sha256
        or quality_gate.get("chunk_manifest_sha256")
        != chunk_manifest_sha256
    ):
        raise ValueError("Quality Gate 与 Evidence/Chunk Manifest 不一致")
    if (
        index_manifest.get("chunk_manifest_sha256")
        != chunk_manifest_sha256
        or index_manifest.get("quality_gate_sha256")
        != quality_gate_sha256
        or index_manifest.get("model_manifest_sha256")
        != model_manifest_sha256
    ):
        raise ValueError("Index Manifest 上游哈希链不一致")
    if smoke_manifest.get("status") != "passed":
        raise ValueError("Pilot 冻结要求 Smoke Manifest 为 passed")
    smoke_inputs = smoke_manifest.get("inputs", {})
    expected_smoke_inputs = {
        "config_sha256": config_sha256,
        "evidence_sha256": evidence_sha256,
        "index_manifest_sha256": index_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "quality_gate_sha256": quality_gate_sha256,
    }
    mismatches = {
        key: smoke_inputs.get(key)
        for key, expected in expected_smoke_inputs.items()
        if smoke_inputs.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Smoke Manifest 上游哈希链不一致: {mismatches}")


def freeze_pilot_manifest(
    *,
    dataset_path: Path,
    evidence_groups_path: Path,
    review_summary_path: Path,
    exclusions_path: Path,
    manifest_path: Path,
    evidence_path: Path,
    chunk_manifest_path: Path,
    quality_gate_path: Path,
    index_manifest_path: Path,
    model_manifest_path: Path,
    config_path: Path,
    smoke_manifest_path: Path,
) -> dict:
    dataset_summary = validate_dataset(
        dataset_path=dataset_path,
        evidence_path=evidence_path,
        profile="pilot",
        evidence_groups_path=evidence_groups_path,
    )
    if dataset_summary["approved_count"] != 40:
        raise ValueError("Pilot-40 必须全部审核为 approved")

    review_summary = _load_json_object(
        review_summary_path,
        label="Pilot review summary",
    )
    exclusions = _load_json_object(
        exclusions_path,
        label="Pilot exclusions",
    )
    chunk_manifest = _load_json_object(
        chunk_manifest_path,
        label="Chunk Manifest",
    )
    quality_gate = _load_json_object(
        quality_gate_path,
        label="Quality Gate",
    )
    index_manifest = _load_json_object(
        index_manifest_path,
        label="Index Manifest",
    )
    _load_json_object(
        model_manifest_path,
        label="Model Manifest",
    )
    smoke_manifest = _load_json_object(
        smoke_manifest_path,
        label="Smoke Manifest",
    )

    dataset_sha256 = dataset_summary["dataset_sha256"]
    evidence_group_sha256 = dataset_summary["evidence_group_sha256"]
    _validate_pilot_review_summary(
        review_summary=review_summary,
        dataset_sha256=dataset_sha256,
        evidence_group_sha256=evidence_group_sha256,
    )
    groups_by_id = _load_pilot_evidence_groups(evidence_groups_path)
    _validate_pilot_exclusions(
        exclusions=exclusions,
        groups_by_id=groups_by_id,
    )

    evidence_sha256 = dataset_summary["evidence_sha256"]
    chunk_manifest_sha256 = _sha256_file(chunk_manifest_path)
    quality_gate_sha256 = _sha256_file(quality_gate_path)
    index_manifest_sha256 = _sha256_file(index_manifest_path)
    model_manifest_sha256 = _sha256_file(model_manifest_path)
    config_sha256 = _sha256_file(config_path)
    _validate_pilot_upstream_manifests(
        evidence_sha256=evidence_sha256,
        chunk_manifest=chunk_manifest,
        chunk_manifest_sha256=chunk_manifest_sha256,
        quality_gate=quality_gate,
        quality_gate_sha256=quality_gate_sha256,
        index_manifest=index_manifest,
        model_manifest_sha256=model_manifest_sha256,
        config_sha256=config_sha256,
        smoke_manifest=smoke_manifest,
        index_manifest_sha256=index_manifest_sha256,
    )

    manifest = {
        "version": "v1.5.0",
        "status": "ready",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": _manifest_path(dataset_path),
            "sha256": dataset_sha256,
            "question_count": dataset_summary["question_count"],
            "answerable_count": dataset_summary["answerable_count"],
            "unanswerable_count": dataset_summary[
                "unanswerable_count"
            ],
            "approved_count": dataset_summary["approved_count"],
        },
        "distribution": dataset_summary["quota_by_book_and_type"],
        "review": {
            "first_pass_count": review_summary[
                "first_review_pass_count"
            ],
            "second_required_count": review_summary[
                "second_review_required_count"
            ],
            "second_pass_count": review_summary[
                "second_review_pass_count"
            ],
            "revision_count": review_summary["revision_count"],
            "rejected_count": review_summary["rejected_count"],
        },
        "inputs": {
            "evidence_group": _hashed_input(evidence_groups_path),
            "review_summary": _hashed_input(review_summary_path),
            "review_csv_sha256": _require_sha256(
                review_summary["review_csv_sha256"],
                label="Pilot review CSV",
            ),
            "exclusions": _hashed_input(exclusions_path),
            "evidence": _hashed_input(evidence_path),
            "chunk_manifest": _hashed_input(chunk_manifest_path),
            "quality_gate": _hashed_input(quality_gate_path),
            "index_manifest": _hashed_input(index_manifest_path),
            "model_manifest": _hashed_input(model_manifest_path),
            "config": _hashed_input(config_path),
            "smoke_manifest": _hashed_input(smoke_manifest_path),
        },
        "privacy": {
            "full_questions_committed": False,
            "full_corpus_committed": False,
        },
    }
    if manifest_path.is_file():
        existing = _load_json_object(
            manifest_path,
            label="现有 Pilot Manifest",
        )
        existing_core = dict(existing)
        manifest_core = dict(manifest)
        existing_core.pop("frozen_at", None)
        manifest_core.pop("frozen_at", None)
        if existing.get("status") == "ready":
            if existing_core != manifest_core:
                raise ValueError(
                    "现有 ready Pilot Manifest 核心输入已变化，拒绝覆盖"
                )
            return existing

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    return manifest


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
