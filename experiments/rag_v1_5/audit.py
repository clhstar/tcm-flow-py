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
