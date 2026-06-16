import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from experiments.rag_v1_5.schema import (
    AuditRecord,
    AuditSampleType,
    EvidenceUnit,
    ParseAnomaly,
)


SAMPLE_QUOTAS: dict[AuditSampleType, int] = {
    "clause": 30,
    "formula": 20,
    "note_or_boundary": 20,
}
SAMPLE_TYPE_ORDER = ("clause", "formula", "note_or_boundary")
CONTENT_TYPE_ORDER = {
    "clause": 0,
    "formula": 1,
    "ingredients": 2,
    "preparation": 3,
    "note": 4,
}
SPECIAL_PATTERNS = ("KT", "又方", "治之方", "附方")
CSV_FIELDS = (
    "audit_id",
    "book_id",
    "sample_type",
    "chapter_id",
    "clause_id",
    "evidence_ids",
    "original_text",
    "structured_summary",
    "status",
    "decision",
    "reviewer",
    "reviewed_at",
    "comment",
)
IMMUTABLE_REVIEW_FIELDS = (
    "audit_id",
    "book_id",
    "sample_type",
    "chapter_id",
    "clause_id",
    "evidence_ids",
    "original_text",
    "structured_summary",
)
REVIEW_MIGRATION_KEY_FIELDS = (
    "book_id",
    "sample_type",
    "clause_id",
)
REVIEW_MIGRATION_COMPARE_FIELDS = (
    "chapter_id",
    "evidence_ids",
    "original_text",
    "structured_summary",
)
REVIEW_MUTABLE_FIELDS = (
    "status",
    "decision",
    "reviewer",
    "reviewed_at",
    "comment",
)
ERROR_DECISIONS = (
    "boundary_error",
    "type_error",
    "parent_error",
    "text_error",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_jsonl(path: Path, model_type):
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _clause_sort_key(group: list[EvidenceUnit]) -> tuple:
    clause = next(unit for unit in group if unit.content_type == "clause")
    clause_number = (
        clause.clause_number
        if clause.clause_number is not None
        else 2**31
    )
    return (
        clause.book_id,
        clause.chapter_id,
        clause_number,
        clause.clause_id,
    )


def _unit_sort_key(unit: EvidenceUnit) -> tuple:
    return (
        CONTENT_TYPE_ORDER[unit.content_type],
        unit.evidence_id,
    )


def _group_text(group: list[EvidenceUnit], field: str) -> str:
    return "\n".join(
        f"[{unit.content_type}] {getattr(unit, field)}"
        for unit in sorted(group, key=_unit_sort_key)
    )


def _group_search_text(group: list[EvidenceUnit]) -> str:
    return "\n".join(
        f"{unit.original_text}\n{unit.normalized_text}" for unit in group
    )


def _rotation_offset(
    *,
    seed: int,
    book_id: str,
    sample_type: AuditSampleType,
    length: int,
) -> int:
    if length == 0:
        return 0
    digest = hashlib.sha256(
        f"{seed}:{book_id}:{sample_type}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % length


def _select_round_robin(
    candidates: list[list[EvidenceUnit]],
    *,
    count: int,
    seed: int,
    book_id: str,
    sample_type: AuditSampleType,
    forced_clause_ids: list[str] | None = None,
) -> list[list[EvidenceUnit]]:
    unique_candidates = {
        group[0].clause_id: group
        for group in sorted(candidates, key=_clause_sort_key)
    }
    if len(unique_candidates) < count:
        raise ValueError(
            f"{book_id} 的 {sample_type} 候选不足: "
            f"required={count}, actual={len(unique_candidates)}"
        )

    selected = []
    selected_ids = set()
    for clause_id in forced_clause_ids or []:
        group = unique_candidates.get(clause_id)
        if group is None or clause_id in selected_ids:
            continue
        selected.append(group)
        selected_ids.add(clause_id)
        if len(selected) == count:
            return selected

    buckets: dict[str, list[list[EvidenceUnit]]] = defaultdict(list)
    for clause_id, group in unique_candidates.items():
        if clause_id not in selected_ids:
            buckets[group[0].chapter_id].append(group)
    for groups in buckets.values():
        groups.sort(key=_clause_sort_key)

    chapter_ids = sorted(buckets)
    offset = _rotation_offset(
        seed=seed,
        book_id=book_id,
        sample_type=sample_type,
        length=len(chapter_ids),
    )
    chapter_ids = chapter_ids[offset:] + chapter_ids[:offset]
    positions = {chapter_id: 0 for chapter_id in chapter_ids}

    while len(selected) < count:
        added = False
        for chapter_id in chapter_ids:
            position = positions[chapter_id]
            groups = buckets[chapter_id]
            if position >= len(groups):
                continue
            group = groups[position]
            positions[chapter_id] += 1
            selected.append(group)
            selected_ids.add(group[0].clause_id)
            added = True
            if len(selected) == count:
                break
        if not added:
            raise ValueError(
                f"{book_id} 的 {sample_type} 无法补足 {count} 组样本"
            )
    return selected


def _forced_note_clause_ids(
    groups: list[list[EvidenceUnit]],
    *,
    anomaly_clause_ids: set[str],
) -> list[str]:
    forced = []
    for pattern in SPECIAL_PATTERNS:
        matching = [
            group
            for group in groups
            if pattern in _group_search_text(group)
        ]
        if matching:
            forced.append(min(matching, key=_clause_sort_key)[0].clause_id)
    multiline = [
        group
        for group in groups
        if any("\n" in unit.original_text for unit in group)
    ]
    if multiline:
        forced.append(min(multiline, key=_clause_sort_key)[0].clause_id)
    forced.extend(sorted(anomaly_clause_ids))
    return list(dict.fromkeys(forced))


def _to_audit_records(
    *,
    book_id: str,
    sample_type: AuditSampleType,
    groups: list[list[EvidenceUnit]],
) -> list[AuditRecord]:
    records = []
    for index, group in enumerate(groups, start=1):
        ordered_group = sorted(group, key=_unit_sort_key)
        clause = next(
            unit for unit in ordered_group if unit.content_type == "clause"
        )
        records.append(
            AuditRecord(
                audit_id=(
                    f"audit-{book_id}-{sample_type.replace('_', '-')}-"
                    f"{index:03d}"
                ),
                book_id=book_id,
                sample_type=sample_type,
                chapter_id=clause.chapter_id,
                clause_id=clause.clause_id,
                evidence_ids=[
                    unit.evidence_id for unit in ordered_group
                ],
                original_text=_group_text(ordered_group, "original_text"),
                structured_summary=_group_text(
                    ordered_group,
                    "normalized_text",
                ),
            )
        )
    return records


def sample_audit_records(
    evidence_units: list[EvidenceUnit],
    anomalies: list[ParseAnomaly],
    *,
    seed: int = 20260612,
) -> list[AuditRecord]:
    groups_by_book_clause: dict[
        tuple[str, str],
        list[EvidenceUnit],
    ] = defaultdict(list)
    for unit in sorted(
        evidence_units,
        key=lambda item: (
            item.book_id,
            item.chapter_id,
            item.clause_number or 2**31,
            item.clause_id,
            CONTENT_TYPE_ORDER[item.content_type],
            item.evidence_id,
        ),
    ):
        groups_by_book_clause[(unit.book_id, unit.clause_id)].append(unit)

    groups_by_book: dict[str, list[list[EvidenceUnit]]] = defaultdict(list)
    for (book_id, clause_id), group in groups_by_book_clause.items():
        clauses = [unit for unit in group if unit.content_type == "clause"]
        if len(clauses) != 1:
            raise ValueError(
                f"{book_id}/{clause_id} 必须恰好包含一个 clause Evidence"
            )
        groups_by_book[book_id].append(group)

    if len(groups_by_book) != 2:
        raise ValueError(
            f"审核抽样要求恰好两本书，actual={len(groups_by_book)}"
        )

    anomalies_by_book: dict[str, set[str]] = defaultdict(set)
    for anomaly in anomalies:
        if anomaly.clause_id:
            anomalies_by_book[anomaly.book_id].add(anomaly.clause_id)

    records = []
    for book_id in sorted(groups_by_book):
        all_groups = sorted(groups_by_book[book_id], key=_clause_sort_key)
        formula_groups = [
            group
            for group in all_groups
            if any(
                unit.content_type in {"formula", "ingredients", "preparation"}
                for unit in group
            )
        ]
        boundary_groups = [
            group
            for group in all_groups
            if (
                any(unit.content_type == "note" for unit in group)
                or group[0].clause_id in anomalies_by_book[book_id]
                or any(
                    pattern in _group_search_text(group)
                    for pattern in SPECIAL_PATTERNS
                )
                or any("\n" in unit.original_text for unit in group)
            )
        ]

        formula_forced = []
        target_clause_id = "jgy-chapter-25-040"
        if any(
            group[0].clause_id == target_clause_id
            for group in formula_groups
        ):
            formula_forced.append(target_clause_id)
        note_forced = _forced_note_clause_ids(
            boundary_groups,
            anomaly_clause_ids=anomalies_by_book[book_id],
        )
        candidates = {
            "clause": all_groups,
            "formula": formula_groups,
            "note_or_boundary": boundary_groups,
        }
        forced = {
            "clause": [],
            "formula": formula_forced,
            "note_or_boundary": note_forced,
        }

        for sample_type in SAMPLE_TYPE_ORDER:
            selected = _select_round_robin(
                candidates[sample_type],
                count=SAMPLE_QUOTAS[sample_type],
                seed=seed,
                book_id=book_id,
                sample_type=sample_type,
                forced_clause_ids=forced[sample_type],
            )
            records.extend(
                _to_audit_records(
                    book_id=book_id,
                    sample_type=sample_type,
                    groups=selected,
                )
            )

    if len(records) != 140:
        raise ValueError(f"审核样本必须为 140 组，actual={len(records)}")
    if len({record.audit_id for record in records}) != len(records):
        raise ValueError("audit_id 必须唯一")
    return records


def _write_jsonl(path: Path, records: list[AuditRecord]) -> None:
    payload = "".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for record in records
    )
    path.write_bytes(payload.encode("utf-8"))


def _write_csv(path: Path, records: list[AuditRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            row = record.model_dump(mode="json")
            row["evidence_ids"] = "|".join(record.evidence_ids)
            for field in ("decision", "reviewer", "reviewed_at"):
                if row[field] is None:
                    row[field] = ""
            writer.writerow(row)


def _sample_counts(records: list[AuditRecord]) -> dict:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {sample_type: 0 for sample_type in SAMPLE_TYPE_ORDER}
    )
    for record in records:
        counts[record.book_id][record.sample_type] += 1
    return {
        book_id: dict(sample_counts)
        for book_id, sample_counts in sorted(counts.items())
    }


def _forced_pattern_hits(records: list[AuditRecord]) -> dict:
    note_records = [
        record
        for record in records
        if record.sample_type == "note_or_boundary"
    ]
    hits = {
        pattern: sum(
            pattern in record.original_text for record in note_records
        )
        for pattern in SPECIAL_PATTERNS
    }
    hits["multiline"] = sum(
        "\n" in record.original_text for record in note_records
    )
    return hits


def build_audit_artifacts(
    *,
    evidence_path: Path,
    anomalies_path: Path,
    output_dir: Path,
    manifest_path: Path,
    seed: int = 20260612,
) -> dict:
    evidence_units = _load_jsonl(evidence_path, EvidenceUnit)
    anomalies = _load_jsonl(anomalies_path, ParseAnomaly)
    records = sample_audit_records(
        evidence_units,
        anomalies,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "audit-140.jsonl"
    csv_path = output_dir / "audit-140.csv"
    _write_jsonl(jsonl_path, records)
    _write_csv(csv_path, records)

    manifest = {
        "version": "v1.5.0",
        "seed": seed,
        "inputs": {
            "evidence_sha256": _sha256_file(evidence_path),
            "anomalies_sha256": _sha256_file(anomalies_path),
        },
        "counts": _sample_counts(records),
        "total_count": len(records),
        "forced_pattern_hits": _forced_pattern_hits(records),
        "outputs": {
            "jsonl_file": jsonl_path.name,
            "jsonl_sha256": _sha256_file(jsonl_path),
            "csv_file": csv_path.name,
            "csv_sha256": _sha256_file(csv_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _source_review_row(record: AuditRecord) -> dict[str, str]:
    row = record.model_dump(mode="json")
    row["evidence_ids"] = "|".join(record.evidence_ids)
    return {
        field: "" if row[field] is None else str(row[field])
        for field in CSV_FIELDS
    }


def _read_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("审核 CSV 列与固定模板不一致")
        return list(reader)


def _validate_review_row(
    *,
    source_record: AuditRecord,
    reviewed_row: dict[str, str],
) -> AuditRecord:
    source_row = _source_review_row(source_record)
    for field in IMMUTABLE_REVIEW_FIELDS:
        if reviewed_row[field] != source_row[field]:
            raise ValueError(
                f"{source_record.audit_id} 不允许修改列: {field}"
            )

    status = reviewed_row["status"].strip()
    decision = reviewed_row["decision"].strip() or None
    reviewer = reviewed_row["reviewer"].strip() or None
    reviewed_at = reviewed_row["reviewed_at"].strip() or None
    comment = reviewed_row["comment"].strip()

    if status == "pending":
        raise ValueError(f"{source_record.audit_id} 尚未完成审核")
    if status not in {"pass", "fail"}:
        raise ValueError(f"{source_record.audit_id} status 非法: {status}")
    if not reviewer or not reviewed_at:
        raise ValueError(
            f"{source_record.audit_id} reviewer/reviewed_at 不能为空"
        )
    if status == "pass" and decision != "correct":
        raise ValueError(
            f"{source_record.audit_id} pass 必须对应 decision=correct"
        )
    if status == "fail" and (
        decision not in ERROR_DECISIONS or not comment
    ):
        raise ValueError(
            f"{source_record.audit_id} fail 必须填写错误类型和 comment"
        )

    return source_record.model_copy(
        update={
            "status": status,
            "decision": decision,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "comment": comment,
        }
    )


def _review_migration_key(record: AuditRecord) -> tuple[str, str, str]:
    return tuple(
        str(getattr(record, field))
        for field in REVIEW_MIGRATION_KEY_FIELDS
    )


def _records_by_migration_key(
    records: list[AuditRecord],
) -> dict[tuple[str, str, str], list[AuditRecord]]:
    grouped: dict[tuple[str, str, str], list[AuditRecord]] = defaultdict(list)
    for record in records:
        grouped[_review_migration_key(record)].append(record)
    return grouped


def _load_validated_previous_reviews(
    *,
    source_jsonl: Path,
    reviewed_csv: Path,
) -> list[AuditRecord]:
    source_records = _load_jsonl(source_jsonl, AuditRecord)
    source_by_id = {record.audit_id: record for record in source_records}
    if len(source_by_id) != len(source_records):
        raise ValueError("旧审核源样本存在重复 audit_id")

    reviewed_rows = _read_review_csv(reviewed_csv)
    reviewed_ids = [row["audit_id"] for row in reviewed_rows]
    if len(set(reviewed_ids)) != len(reviewed_ids):
        raise ValueError("旧审核 CSV 存在重复 audit_id")
    if set(reviewed_ids) != set(source_by_id):
        raise ValueError("旧审核 CSV audit_id 与旧源样本不匹配")

    return [
        _validate_review_row(
            source_record=source_by_id[row["audit_id"]],
            reviewed_row=row,
        )
        for row in reviewed_rows
    ]


def _pending_review(record: AuditRecord) -> AuditRecord:
    return record.model_copy(
        update={
            "status": "pending",
            "decision": None,
            "reviewer": None,
            "reviewed_at": None,
            "comment": "",
        }
    )


def migrate_audit_review(
    *,
    previous_source_jsonl: Path,
    previous_reviewed_csv: Path,
    new_source_jsonl: Path,
    output_csv: Path,
    summary_path: Path,
) -> dict:
    previous_records = _load_validated_previous_reviews(
        source_jsonl=previous_source_jsonl,
        reviewed_csv=previous_reviewed_csv,
    )
    new_records = _load_jsonl(new_source_jsonl, AuditRecord)
    new_ids = [record.audit_id for record in new_records]
    if len(set(new_ids)) != len(new_ids):
        raise ValueError("新审核源样本存在重复 audit_id")

    previous_by_key = _records_by_migration_key(previous_records)
    new_by_key = _records_by_migration_key(new_records)
    migrated_records = []
    details = []
    inherited_count = 0
    missing_count = 0
    ambiguous_count = 0
    structure_changed_count = 0

    for new_record in new_records:
        key = _review_migration_key(new_record)
        previous_matches = previous_by_key.get(key, [])
        new_matches = new_by_key[key]
        reason = None
        changed_fields = []

        if len(previous_matches) > 1 or len(new_matches) > 1:
            reason = "ambiguous"
            ambiguous_count += 1
        elif not previous_matches:
            reason = "missing"
            missing_count += 1
        else:
            previous_record = previous_matches[0]
            changed_fields = [
                field
                for field in REVIEW_MIGRATION_COMPARE_FIELDS
                if getattr(previous_record, field) != getattr(new_record, field)
            ]
            if changed_fields:
                reason = "structure_changed"
                structure_changed_count += 1
            else:
                inherited_count += 1
                migrated_records.append(
                    new_record.model_copy(
                        update={
                            field: getattr(previous_record, field)
                            for field in REVIEW_MUTABLE_FIELDS
                        }
                    )
                )
                continue

        migrated_records.append(_pending_review(new_record))
        details.append(
            {
                "audit_id": new_record.audit_id,
                "book_id": new_record.book_id,
                "sample_type": new_record.sample_type,
                "clause_id": new_record.clause_id,
                "reason": reason,
                "changed_fields": changed_fields,
            }
        )

    _write_csv(output_csv, migrated_records)
    pending_audit_ids = [
        record.audit_id
        for record in migrated_records
        if record.status == "pending"
    ]
    summary = {
        "total_count": len(migrated_records),
        "inherited_count": inherited_count,
        "reset_count": len(pending_audit_ids),
        "missing_count": missing_count,
        "ambiguous_count": ambiguous_count,
        "structure_changed_count": structure_changed_count,
        "pending_audit_ids": pending_audit_ids,
        "details": details,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return summary


def _write_review_issues(
    path: Path,
    records: list[AuditRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for record in records
        if record.status == "fail"
    )
    path.write_text(payload, encoding="utf-8")


def _error_counts(records: list[AuditRecord]) -> dict[str, int]:
    return {
        decision: sum(
            record.status == "fail" and record.decision == decision
            for record in records
        )
        for decision in ERROR_DECISIONS
    }


def import_audit_review(
    *,
    source_jsonl: Path,
    reviewed_csv: Path,
    issues_path: Path,
    summary_path: Path,
) -> dict:
    source_records = _load_jsonl(source_jsonl, AuditRecord)
    if len(source_records) != 140:
        raise ValueError(
            f"审核源样本必须为 140 行，actual={len(source_records)}"
        )
    source_by_id = {record.audit_id: record for record in source_records}
    if len(source_by_id) != 140:
        raise ValueError("审核源样本存在重复 audit_id")

    reviewed_rows = _read_review_csv(reviewed_csv)
    if len(reviewed_rows) != 140:
        raise ValueError(
            f"审核 CSV 必须为 140 行，actual={len(reviewed_rows)}"
        )
    reviewed_ids = [row["audit_id"] for row in reviewed_rows]
    if len(set(reviewed_ids)) != 140:
        raise ValueError("审核 CSV 存在重复 audit_id")
    unknown_ids = set(reviewed_ids) - set(source_by_id)
    missing_ids = set(source_by_id) - set(reviewed_ids)
    if unknown_ids or missing_ids:
        raise ValueError(
            f"审核 CSV audit_id 不匹配: "
            f"unknown={sorted(unknown_ids)}, missing={sorted(missing_ids)}"
        )

    reviewed_records = [
        _validate_review_row(
            source_record=source_by_id[row["audit_id"]],
            reviewed_row=row,
        )
        for row in reviewed_rows
    ]
    fail_records = [
        record for record in reviewed_records if record.status == "fail"
    ]
    current_error_counts = _error_counts(reviewed_records)
    initial_error_counts = current_error_counts
    if summary_path.is_file():
        previous_summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        initial_error_counts = previous_summary.get(
            "initial_error_counts",
            current_error_counts,
        )

    summary = {
        "status": "ready" if not fail_records else "blocked",
        "reviewed_count": len(reviewed_records),
        "pending_count": 0,
        "pass_count": len(reviewed_records) - len(fail_records),
        "fail_count": len(fail_records),
        "initial_error_counts": initial_error_counts,
        "unresolved_error_counts": current_error_counts,
        "unresolved_boundary_errors": current_error_counts[
            "boundary_error"
        ],
        "unresolved_type_errors": current_error_counts["type_error"],
        "unresolved_parent_errors": current_error_counts["parent_error"],
        "unresolved_text_errors": current_error_counts["text_error"],
        "reviewers": sorted(
            {record.reviewer for record in reviewed_records if record.reviewer}
        ),
        "reviewed_dates": sorted(
            {
                record.reviewed_at
                for record in reviewed_records
                if record.reviewed_at
            }
        ),
        "counts": _sample_counts(reviewed_records),
    }
    _write_review_issues(issues_path, reviewed_records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return summary


def freeze_quality_gate(
    *,
    summary: dict,
    source_jsonl: Path,
    reviewed_csv: Path,
    evidence_path: Path,
    chunks_dir: Path,
    chunk_manifest_path: Path,
    quality_gate_path: Path,
) -> dict:
    if summary.get("status") not in {"ready", "blocked"}:
        raise ValueError("Quality Gate 只允许 ready 或 blocked")
    if summary.get("reviewed_count") != 140:
        raise ValueError("Quality Gate 要求 reviewed_count=140")

    chunk_manifest = json.loads(
        chunk_manifest_path.read_text(encoding="utf-8")
    )
    chunks = {}
    for strategy in ("c0", "c1", "c2", "c3", "c4"):
        strategy_manifest = chunk_manifest.get("strategies", {}).get(strategy)
        if strategy_manifest is None:
            raise ValueError(f"Chunk Manifest 缺少策略: {strategy}")
        chunk_path = chunks_dir / strategy_manifest["output_file"]
        if not chunk_path.is_file():
            raise FileNotFoundError(f"缺少 Chunk 文件: {chunk_path}")
        actual_sha256 = _sha256_file(chunk_path)
        expected_sha256 = strategy_manifest["output_sha256"]
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{strategy} SHA256 不匹配: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        chunks[strategy] = actual_sha256

    gate = {
        "version": "v1.5.0",
        "status": summary["status"],
        "reviewed_count": summary["reviewed_count"],
        "pending_count": summary["pending_count"],
        "audit_sample_sha256": _sha256_file(source_jsonl),
        "audit_review_sha256": _sha256_file(reviewed_csv),
        "evidence_sha256": _sha256_file(evidence_path),
        "chunk_manifest_sha256": _sha256_file(chunk_manifest_path),
        "chunks": chunks,
        "counts": summary["counts"],
        "initial_error_counts": summary["initial_error_counts"],
        "unresolved_error_counts": summary["unresolved_error_counts"],
        "reviewers": summary["reviewers"],
        "reviewed_dates": summary["reviewed_dates"],
    }
    quality_gate_path.parent.mkdir(parents=True, exist_ok=True)
    quality_gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gate
