import csv
import hashlib
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
