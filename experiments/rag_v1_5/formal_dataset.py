import hashlib
import json
import random
import re
from collections import Counter, defaultdict
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
FORMAL_QUESTION_TYPES = tuple(FORMAL_PER_BOOK_SPLIT)
UNANSWERABLE_TOPICS = (
    "青霉素用法",
    "磁共振检查",
    "胰岛素剂量",
    "血型鉴定",
    "疫苗接种",
    "心电图判读",
    "CT检查",
    "基因检测",
    "器官移植",
    "血液透析",
    "抗生素疗程",
    "化学治疗",
    "放射治疗",
    "微创手术",
    "腹腔镜检查",
    "人工呼吸机",
    "血氧饱和度",
    "核酸检测",
    "抗体滴度",
    "胆固醇指标",
    "糖化血红蛋白",
    "肿瘤标志物",
    "幽门螺杆菌",
    "冠状动脉支架",
    "心脏起搏器",
    "现代麻醉剂量",
    "静脉输液",
    "血液培养",
    "药敏试验",
    "超声检查",
    "脑电图检查",
    "骨密度检查",
    "凝血功能指标",
    "现代肝功能指标",
    "现代肾功能指标",
    "电解质检查",
    "现代药品说明书",
    "国际疾病编码",
    "现代住院流程",
    "医保报销规则",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _canonical_model_sha256(records: list) -> str:
    payload = "".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in sorted(
            records,
            key=lambda item: item.evidence_id,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


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


def _formal_group_id(
    book: str,
    split: str,
    question_type: str,
    index: int,
) -> str:
    return (
        f"formal-{book}-{split}-{question_type}-{index:02d}"
    )


def _build_clause_indexes(
    evidence_units: list[EvidenceUnit],
) -> tuple[
    dict[str, EvidenceUnit],
    dict[str, list[EvidenceUnit]],
]:
    clause_units = {}
    evidence_by_clause: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for evidence in evidence_units:
        evidence_by_clause[evidence.clause_id].append(evidence)
        if evidence.content_type == "clause":
            clause_units[evidence.clause_id] = evidence
    return clause_units, evidence_by_clause


def _clause_category(
    evidence: list[EvidenceUnit],
) -> str:
    content_types = {item.content_type for item in evidence}
    if {"formula", "ingredients", "preparation"} <= content_types:
        return "formula"
    if "note" in content_types:
        return "note"
    return "other"


def _assign_splits(
    *,
    clause_units: dict[str, EvidenceUnit],
    evidence_by_clause: dict[str, list[EvidenceUnit]],
    excluded_clause_ids: set[str],
    seed: int,
) -> dict[tuple[str, str], list[EvidenceUnit]]:
    assigned = {
        (book, split): []
        for book in FORMAL_BOOKS
        for split in FORMAL_SPLITS
    }
    for book in FORMAL_BOOKS:
        categorized: dict[str, list[EvidenceUnit]] = defaultdict(list)
        for clause in clause_units.values():
            if (
                clause.book_id != book
                or clause.clause_id in excluded_clause_ids
            ):
                continue
            category = _clause_category(
                evidence_by_clause[clause.clause_id]
            )
            categorized[category].append(clause)
        for category in ("formula", "note", "other"):
            candidates = sorted(
                categorized[category],
                key=lambda item: item.clause_id,
            )
            shuffled = _stable_sample(
                candidates,
                count=len(candidates),
                seed=seed,
                stratum=f"split:{book}:{category}",
            )
            for index, clause in enumerate(shuffled):
                split = FORMAL_SPLITS[index % len(FORMAL_SPLITS)]
                assigned[(book, split)].append(clause)
    for candidates in assigned.values():
        candidates.sort(
            key=lambda item: (
                item.chapter_id,
                item.clause_number
                if item.clause_number is not None
                else 10**9,
                item.clause_id,
            )
        )
    return assigned


def _select_formal_groups(
    *,
    evidence_units: list[EvidenceUnit],
    excluded_clause_ids: set[str],
    seed: int,
) -> tuple[list[FormalEvidenceGroup], dict]:
    clause_units, evidence_by_clause = _build_clause_indexes(
        evidence_units
    )
    assigned = _assign_splits(
        clause_units=clause_units,
        evidence_by_clause=evidence_by_clause,
        excluded_clause_ids=excluded_clause_ids,
        seed=seed,
    )
    groups = []
    strata = {}
    blocked_strata = []
    used_clause_ids: set[str] = set()
    corpus_text_by_book = {
        book: "\n".join(
            evidence.normalized_text
            for evidence in evidence_units
            if evidence.book_id == book
        )
        for book in FORMAL_BOOKS
    }

    def select_single(
        *,
        book: str,
        split: str,
        question_type: str,
        candidates: list[EvidenceUnit],
        priority_candidates: list[EvidenceUnit] | None = None,
    ) -> None:
        target = FORMAL_PER_BOOK_SPLIT[question_type]
        stratum = f"{book}/{split}/{question_type}"
        available = [
            candidate
            for candidate in candidates
            if candidate.clause_id not in used_clause_ids
        ]
        selected = []
        if priority_candidates:
            priority = [
                candidate
                for candidate in priority_candidates
                if candidate.clause_id not in used_clause_ids
            ]
            selected.extend(
                _stable_sample(
                    priority,
                    count=min(target, len(priority)),
                    seed=seed,
                    stratum=f"{stratum}:priority",
                )
            )
        selected_ids = {
            candidate.clause_id for candidate in selected
        }
        if len(selected) < target:
            fallback = [
                candidate
                for candidate in available
                if candidate.clause_id not in selected_ids
            ]
            selected.extend(
                _stable_sample(
                    fallback,
                    count=target - len(selected),
                    seed=seed,
                    stratum=f"{stratum}:fallback",
                )
            )
        strata[stratum] = {
            "candidate_count": len(available),
            "selected_count": min(len(selected), target),
            "target_count": target,
        }
        if len(selected) < target:
            blocked_strata.append(stratum)
            return
        for index, clause in enumerate(selected, start=1):
            clause_evidence = evidence_by_clause[clause.clause_id]
            if question_type == "formula_composition_or_use":
                anchor_evidence_ids = sorted(
                    item.evidence_id
                    for item in clause_evidence
                    if item.content_type
                    in {"formula", "ingredients", "preparation"}
                )
                selection_reason = (
                    "包含完整 formula、ingredients 和 preparation"
                )
            elif question_type == "source_location":
                note_ids = sorted(
                    item.evidence_id
                    for item in clause_evidence
                    if item.content_type == "note"
                )
                anchor_evidence_ids = note_ids or [clause.evidence_id]
                selection_reason = (
                    "包含 note 或具备明确篇章与条文定位"
                )
            else:
                anchor_evidence_ids = [clause.evidence_id]
                selection_reason = "条文正文满足事实型问题候选条件"
            groups.append(
                FormalEvidenceGroup(
                    group_id=_formal_group_id(
                        book,
                        split,
                        question_type,
                        index,
                    ),
                    split=split,
                    book_scope=book,
                    question_type=question_type,
                    anchor_evidence_ids=anchor_evidence_ids,
                    anchor_clause_ids=[clause.clause_id],
                    selection_seed=seed,
                    selection_reason=selection_reason,
                )
            )
            used_clause_ids.add(clause.clause_id)

    for book in FORMAL_BOOKS:
        for split in FORMAL_SPLITS:
            split_candidates = assigned[(book, split)]
            formula_candidates = [
                clause
                for clause in split_candidates
                if _clause_category(
                    evidence_by_clause[clause.clause_id]
                )
                == "formula"
            ]
            select_single(
                book=book,
                split=split,
                question_type="formula_composition_or_use",
                candidates=formula_candidates,
            )

            source_candidates = [
                clause
                for clause in split_candidates
                if (
                    any(
                        item.content_type == "note"
                        for item in evidence_by_clause[clause.clause_id]
                    )
                    or (
                        bool(clause.chapter_title.strip())
                        and clause.clause_number is not None
                    )
                )
            ]
            source_priority = [
                clause
                for clause in source_candidates
                if any(
                    item.content_type == "note"
                    for item in evidence_by_clause[clause.clause_id]
                )
            ]
            select_single(
                book=book,
                split=split,
                question_type="source_location",
                candidates=source_candidates,
                priority_candidates=source_priority,
            )

            fact_candidates = [
                clause
                for clause in split_candidates
                if (
                    len(clause.normalized_text.strip()) >= 8
                    and re.search(
                        r"[\w\u4e00-\u9fff]",
                        clause.normalized_text,
                    )
                )
            ]
            select_single(
                book=book,
                split=split,
                question_type="single_clause_fact",
                candidates=fact_candidates,
            )

            remaining_by_chapter: dict[
                str, list[EvidenceUnit]
            ] = defaultdict(list)
            for clause in split_candidates:
                if clause.clause_id not in used_clause_ids:
                    remaining_by_chapter[clause.chapter_id].append(clause)
            pair_candidates = []
            for chapter_id in sorted(remaining_by_chapter):
                chapter_clauses = _stable_sample(
                    sorted(
                        remaining_by_chapter[chapter_id],
                        key=lambda item: item.clause_id,
                    ),
                    count=len(remaining_by_chapter[chapter_id]),
                    seed=seed,
                    stratum=(
                        f"{book}/{split}/multi_evidence/{chapter_id}"
                    ),
                )
                pair_candidates.extend(
                    zip(chapter_clauses[::2], chapter_clauses[1::2])
                )
            stratum = f"{book}/{split}/multi_evidence"
            target = FORMAL_PER_BOOK_SPLIT["multi_evidence"]
            selected_pairs = _stable_sample(
                pair_candidates,
                count=target,
                seed=seed,
                stratum=stratum,
            )
            strata[stratum] = {
                "candidate_count": len(pair_candidates),
                "selected_count": min(len(selected_pairs), target),
                "target_count": target,
            }
            if len(selected_pairs) < target:
                blocked_strata.append(stratum)
            else:
                for index, (first, second) in enumerate(
                    selected_pairs,
                    start=1,
                ):
                    groups.append(
                        FormalEvidenceGroup(
                            group_id=_formal_group_id(
                                book,
                                split,
                                "multi_evidence",
                                index,
                            ),
                            split=split,
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
                                "同书、同 split、同篇章条文对，"
                                "待人工确认可比较信息"
                            ),
                        )
                    )
                    used_clause_ids.update(
                        {first.clause_id, second.clause_id}
                    )

            topic_offset = 0 if split == "formal_dev" else 20
            unanswerable_stratum = (
                f"{book}/{split}/unanswerable"
            )
            selected_topics = UNANSWERABLE_TOPICS[
                topic_offset : topic_offset + 20
            ]
            invalid_topics = [
                topic
                for topic in selected_topics
                if topic in corpus_text_by_book[book]
            ]
            strata[unanswerable_stratum] = {
                "candidate_count": len(selected_topics),
                "selected_count": (
                    len(selected_topics) - len(invalid_topics)
                ),
                "target_count": 20,
                "present_in_corpus_topics": invalid_topics,
            }
            if invalid_topics or len(selected_topics) != 20:
                blocked_strata.append(unanswerable_stratum)
            else:
                for index, topic in enumerate(
                    selected_topics,
                    start=1,
                ):
                    groups.append(
                        FormalEvidenceGroup(
                            group_id=_formal_group_id(
                                book,
                                split,
                                "unanswerable",
                                index,
                            ),
                            split=split,
                            book_scope=book,
                            question_type="unanswerable",
                            anchor_evidence_ids=[],
                            anchor_clause_ids=[],
                            selection_seed=seed,
                            selection_reason=(
                                "域外现代主题，语料精确缺失，"
                                "仍需人工复核"
                            ),
                            absence_queries=[
                                topic,
                                f"{topic} 相关记载",
                            ],
                        )
                    )

    report = {
        "version": "v1.5.0",
        "seed": seed,
        "status": "blocked" if blocked_strata else "ready",
        "blocked_strata": sorted(set(blocked_strata)),
        "strata": strata,
        "excluded_clause_count": len(excluded_clause_ids),
        "selected_group_count": len(groups),
    }
    return groups, report


def audit_formal_candidates(
    *,
    evidence_path: Path,
    prior_exclusions_path: Path,
) -> dict:
    evidence_units = sorted(
        _read_jsonl(evidence_path, EvidenceUnit),
        key=lambda item: item.evidence_id,
    )
    exclusions = _load_json_object(prior_exclusions_path)
    excluded_clause_ids = _collect_excluded_ids(
        exclusions,
        "_clause_ids",
    )
    groups, report = _select_formal_groups(
        evidence_units=evidence_units,
        excluded_clause_ids=excluded_clause_ids,
        seed=20260614,
    )
    report["selected_group_count"] = len(groups)
    report["evidence_content_sha256"] = _canonical_model_sha256(
        evidence_units
    )
    report["prior_exclusions_sha256"] = _sha256_file(
        prior_exclusions_path
    )
    return report


def sample_formal_evidence_groups(
    *,
    evidence_path: Path,
    smoke_dataset_path: Path,
    pilot_dataset_path: Path,
    pilot_exclusions_path: Path,
    output_path: Path,
    exclusions_path: Path,
    candidate_report_path: Path,
    seed: int = 20260614,
) -> dict:
    evidence_units = sorted(
        _read_jsonl(evidence_path, EvidenceUnit),
        key=lambda item: item.evidence_id,
    )
    prior_questions = _load_prior_questions(
        (smoke_dataset_path, pilot_dataset_path)
    )
    pilot_exclusions = _load_json_object(pilot_exclusions_path)
    prior_evidence_ids = {
        evidence_id
        for question in prior_questions
        for evidence_id in question.gold_evidence_ids
    }
    prior_clause_ids = {
        clause_id
        for question in prior_questions
        for clause_id in question.gold_clause_ids
    }
    prior_group_ids = {
        question.evidence_group_id
        for question in prior_questions
        if question.evidence_group_id
    }
    prior_evidence_ids.update(
        _collect_excluded_ids(pilot_exclusions, "_evidence_ids")
    )
    prior_clause_ids.update(
        _collect_excluded_ids(pilot_exclusions, "_clause_ids")
    )
    prior_group_ids.update(
        _collect_excluded_ids(pilot_exclusions, "_group_ids")
    )

    groups, report = _select_formal_groups(
        evidence_units=evidence_units,
        excluded_clause_ids=prior_clause_ids,
        seed=seed,
    )
    report.update(
        {
            "evidence_content_sha256": _canonical_model_sha256(
                evidence_units
            ),
            "smoke_dataset_sha256": _sha256_file(smoke_dataset_path),
            "pilot_dataset_sha256": _sha256_file(pilot_dataset_path),
            "pilot_exclusions_sha256": _sha256_file(
                pilot_exclusions_path
            ),
            "prior_evidence_count": len(prior_evidence_ids),
            "prior_clause_count": len(prior_clause_ids),
            "prior_group_count": len(prior_group_ids),
        }
    )
    if report["status"] != "ready":
        _write_json(candidate_report_path, report)
        return report

    group_records = [
        group.model_dump(mode="json")
        for group in sorted(
            groups,
            key=lambda group: (
                FORMAL_BOOKS.index(group.book_scope),
                FORMAL_SPLITS.index(group.split),
                FORMAL_QUESTION_TYPES.index(group.question_type),
                group.group_id,
            ),
        )
    ]
    formal_anchor_clause_ids = {
        clause_id
        for group in groups
        for clause_id in group.anchor_clause_ids
    }
    formal_anchor_evidence_ids = {
        evidence_id
        for group in groups
        for evidence_id in group.anchor_evidence_ids
    }
    if formal_anchor_clause_ids & prior_clause_ids:
        raise ValueError("Formal 抽样与 Smoke/Pilot clause 重叠")
    if formal_anchor_evidence_ids & prior_evidence_ids:
        raise ValueError("Formal 抽样与 Smoke/Pilot Evidence 重叠")
    dev_clauses = {
        clause_id
        for group in groups
        if group.split == "formal_dev"
        for clause_id in group.anchor_clause_ids
    }
    test_clauses = {
        clause_id
        for group in groups
        if group.split == "formal_test"
        for clause_id in group.anchor_clause_ids
    }
    if dev_clauses & test_clauses:
        raise ValueError("Formal dev/test clause 重叠")

    exclusions = {
        "version": "v1.5.0",
        "selection_seed": seed,
        "prior_group_ids": sorted(prior_group_ids),
        "prior_evidence_ids": sorted(prior_evidence_ids),
        "prior_clause_ids": sorted(prior_clause_ids),
        "source_pilot_exclusions_sha256": _sha256_file(
            pilot_exclusions_path
        ),
    }
    _write_jsonl(output_path, group_records)
    _write_json(exclusions_path, exclusions)
    _write_json(candidate_report_path, report)

    answerable_groups = [
        group
        for group in groups
        if group.question_type != "unanswerable"
    ]
    return {
        "status": "ready",
        "group_count": len(groups),
        "answerable_group_count": len(answerable_groups),
        "unanswerable_group_count": (
            len(groups) - len(answerable_groups)
        ),
        "formal_dev_count": sum(
            group.split == "formal_dev" for group in groups
        ),
        "formal_test_count": sum(
            group.split == "formal_test" for group in groups
        ),
        "prior_overlap_count": 0,
        "cross_split_clause_overlap_count": 0,
        "answerable_anchor_clause_count": len(
            formal_anchor_clause_ids
        ),
        "evidence_group_sha256": _sha256_file(output_path),
        "exclusions_sha256": _sha256_file(exclusions_path),
        "candidate_report_sha256": _sha256_file(
            candidate_report_path
        ),
    }


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
