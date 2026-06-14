import csv
import importlib
import importlib.util
import io
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from experiments.rag_v1_5.schema import (
    AuditRecord,
    EvidenceUnit,
    ParseAnomaly,
)


BOOKS = ("shang_han_lun", "jin_gui_yao_lue")


def make_evidence() -> list[EvidenceUnit]:
    evidence = []
    source_hash = "A" * 64
    for book_id in BOOKS:
        prefix = "shl" if book_id == "shang_han_lun" else "jgy"
        book_title = "伤寒论" if prefix == "shl" else "金匮要略方论"
        for index in range(1, 41):
            chapter_number = ((index - 1) % 4) + 1
            chapter_id = f"{prefix}-chapter-{chapter_number:02d}"
            clause_id = f"{chapter_id}-{index:03d}"
            if book_id == "jin_gui_yao_lue" and index == 40:
                chapter_id = "jgy-chapter-25"
                clause_id = "jgy-chapter-25-040"
            markers = {
                1: "KT",
                2: "又方",
                3: "治之方",
                4: "附方",
                5: "第一行\n第二行",
            }
            marker = markers.get(index, "")
            clause_text = f"{book_title}第{index}条 {marker}".strip()
            evidence.append(
                EvidenceUnit(
                    evidence_id=clause_id,
                    book_id=book_id,
                    book_title=book_title,
                    volume="",
                    chapter_id=chapter_id,
                    chapter_title=f"第{chapter_number}篇",
                    clause_id=clause_id,
                    clause_number=index,
                    content_type="clause",
                    parent_id=clause_id,
                    original_text=clause_text,
                    normalized_text=clause_text.replace("\n", " "),
                    notes=[],
                    source_file=f"{book_id}.txt",
                    source_hash=source_hash,
                    corpus_version="v1.5.0",
                )
            )
            if index <= 25 or clause_id == "jgy-chapter-25-040":
                formula_id = f"{clause_id}-formula-01"
                evidence.extend(
                    [
                        EvidenceUnit(
                            evidence_id=formula_id,
                            book_id=book_id,
                            book_title=book_title,
                            volume="",
                            chapter_id=chapter_id,
                            chapter_title=f"第{chapter_number}篇",
                            clause_id=clause_id,
                            clause_number=index,
                            content_type="formula",
                            parent_id=clause_id,
                            original_text=f"方剂 {marker}".strip(),
                            normalized_text=f"方剂 {marker}".strip(),
                            notes=[],
                            source_file=f"{book_id}.txt",
                            source_hash=source_hash,
                            corpus_version="v1.5.0",
                        ),
                        EvidenceUnit(
                            evidence_id=f"{formula_id}-ingredients",
                            book_id=book_id,
                            book_title=book_title,
                            volume="",
                            chapter_id=chapter_id,
                            chapter_title=f"第{chapter_number}篇",
                            clause_id=clause_id,
                            clause_number=index,
                            content_type="ingredients",
                            parent_id=formula_id,
                            original_text=f"药物甲 药物乙 {index}",
                            normalized_text=f"药物甲 药物乙 {index}",
                            notes=[],
                            source_file=f"{book_id}.txt",
                            source_hash=source_hash,
                            corpus_version="v1.5.0",
                        ),
                    ]
                )
            if index <= 22:
                evidence.append(
                    EvidenceUnit(
                        evidence_id=f"{clause_id}-note-01",
                        book_id=book_id,
                        book_title=book_title,
                        volume="",
                        chapter_id=chapter_id,
                        chapter_title=f"第{chapter_number}篇",
                        clause_id=clause_id,
                        clause_number=index,
                        content_type="note",
                        parent_id=clause_id,
                        original_text=f"校注 {index}",
                        normalized_text=f"校注 {index}",
                        notes=[],
                        source_file=f"{book_id}.txt",
                        source_hash=source_hash,
                        corpus_version="v1.5.0",
                    )
                )
    return evidence


def make_anomalies() -> list[ParseAnomaly]:
    return [
        ParseAnomaly(
            anomaly_id="anomaly-001",
            book_id="shang_han_lun",
            source_file="shang_han_lun.txt",
            chapter_id="shl-chapter-01",
            chapter_title="第一篇",
            clause_id="shl-chapter-01-001",
            reason="missing_character_marker",
            original_text="KT",
        )
    ]


def make_audit_record(
    *,
    audit_id: str,
    clause_id: str,
    structured_summary: str | None = None,
    evidence_ids: list[str] | None = None,
) -> AuditRecord:
    return AuditRecord(
        audit_id=audit_id,
        book_id="jin_gui_yao_lue",
        sample_type="formula",
        chapter_id="jgy-chapter-01",
        clause_id=clause_id,
        evidence_ids=evidence_ids or [clause_id, f"{clause_id}-formula-01"],
        original_text=f"[clause] {clause_id}",
        structured_summary=(
            structured_summary
            if structured_summary is not None
            else f"[formula] {clause_id}"
        ),
    )


class AuditSamplingTests(unittest.TestCase):
    def audit_module(self):
        module_spec = importlib.util.find_spec("experiments.rag_v1_5.audit")
        self.assertIsNotNone(module_spec, "audit module is not implemented")
        return importlib.import_module("experiments.rag_v1_5.audit")

    def test_samples_strict_balanced_deterministic_quota(self) -> None:
        audit = self.audit_module()
        sample_audit_records = getattr(
            audit,
            "sample_audit_records",
            None,
        )
        self.assertTrue(
            callable(sample_audit_records),
            "sample_audit_records is not implemented",
        )
        evidence = make_evidence()
        anomalies = make_anomalies()

        first = sample_audit_records(evidence, anomalies, seed=20260612)
        second = sample_audit_records(evidence, anomalies, seed=20260612)

        self.assertEqual(
            [record.model_dump(mode="json") for record in first],
            [record.model_dump(mode="json") for record in second],
        )
        self.assertEqual(len(first), 140)
        self.assertEqual(
            Counter((record.book_id, record.sample_type) for record in first),
            Counter(
                {
                    ("shang_han_lun", "clause"): 30,
                    ("shang_han_lun", "formula"): 20,
                    ("shang_han_lun", "note_or_boundary"): 20,
                    ("jin_gui_yao_lue", "clause"): 30,
                    ("jin_gui_yao_lue", "formula"): 20,
                    ("jin_gui_yao_lue", "note_or_boundary"): 20,
                }
            ),
        )
        self.assertEqual(len({record.audit_id for record in first}), 140)
        self.assertIn(
            "jgy-chapter-25-040",
            {record.clause_id for record in first},
        )

        by_book_type = defaultdict(list)
        for record in first:
            by_book_type[(record.book_id, record.sample_type)].append(record)
        for key, records in by_book_type.items():
            with self.subTest(group=key):
                self.assertEqual(
                    len({record.clause_id for record in records}),
                    len(records),
                )
                self.assertGreaterEqual(
                    len({record.chapter_id for record in records}),
                    4,
                )

        expected_ids = defaultdict(set)
        for unit in evidence:
            expected_ids[(unit.book_id, unit.clause_id)].add(unit.evidence_id)
        for record in first:
            with self.subTest(audit_id=record.audit_id):
                self.assertEqual(
                    set(record.evidence_ids),
                    expected_ids[(record.book_id, record.clause_id)],
                )

        note_text = "\n".join(
            record.original_text
            for record in first
            if record.sample_type == "note_or_boundary"
        )
        for marker in ("KT", "又方", "治之方", "附方", "第一行"):
            with self.subTest(marker=marker):
                self.assertIn(marker, note_text)

    def test_raises_when_any_quota_cannot_be_met(self) -> None:
        audit = self.audit_module()
        with self.assertRaises(ValueError):
            audit.sample_audit_records(
                make_evidence()[:10],
                [],
                seed=20260612,
            )

    def test_writes_jsonl_csv_and_manifest_deterministically(self) -> None:
        audit = self.audit_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence_path = root / "evidence.jsonl"
            anomalies_path = root / "anomalies.jsonl"
            output_dir = root / "audit"
            manifest_path = root / "audit-sample-v1.5.0.json"
            evidence_path.write_text(
                "".join(
                    json.dumps(
                        unit.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                    for unit in make_evidence()
                ),
                encoding="utf-8",
            )
            anomalies_path.write_text(
                "".join(
                    json.dumps(
                        anomaly.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                    for anomaly in make_anomalies()
                ),
                encoding="utf-8",
            )

            first = audit.build_audit_artifacts(
                evidence_path=evidence_path,
                anomalies_path=anomalies_path,
                output_dir=output_dir,
                manifest_path=manifest_path,
                seed=20260612,
            )
            first_jsonl = (output_dir / "audit-140.jsonl").read_bytes()
            first_csv = (output_dir / "audit-140.csv").read_bytes()
            second = audit.build_audit_artifacts(
                evidence_path=evidence_path,
                anomalies_path=anomalies_path,
                output_dir=output_dir,
                manifest_path=manifest_path,
                seed=20260612,
            )

            self.assertEqual(first, second)
            self.assertEqual(
                first_jsonl,
                (output_dir / "audit-140.jsonl").read_bytes(),
            )
            self.assertEqual(
                first_csv,
                (output_dir / "audit-140.csv").read_bytes(),
            )
            self.assertTrue(first_csv.startswith(b"\xef\xbb\xbf"))
            rows = list(
                csv.DictReader(
                    io.StringIO(first_csv.decode("utf-8-sig"))
                )
            )
            self.assertEqual(len(rows), 140)
            self.assertEqual(
                list(rows[0]),
                [
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
                ],
            )
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                first,
            )


class AuditReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.evidence_path = self.root / "evidence.jsonl"
        self.anomalies_path = self.root / "anomalies.jsonl"
        self.output_dir = self.root / "audit"
        self.sample_manifest_path = self.root / "audit-sample.json"
        self.evidence_path.write_text(
            "".join(
                json.dumps(
                    unit.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for unit in make_evidence()
            ),
            encoding="utf-8",
        )
        self.anomalies_path.write_text(
            "".join(
                json.dumps(
                    anomaly.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for anomaly in make_anomalies()
            ),
            encoding="utf-8",
        )
        self.audit = importlib.import_module("experiments.rag_v1_5.audit")
        self.audit.build_audit_artifacts(
            evidence_path=self.evidence_path,
            anomalies_path=self.anomalies_path,
            output_dir=self.output_dir,
            manifest_path=self.sample_manifest_path,
        )
        self.source_jsonl = self.output_dir / "audit-140.jsonl"
        self.reviewed_csv = self.output_dir / "audit-140.csv"
        self.issues_path = self.output_dir / "audit-issues.jsonl"
        self.summary_path = self.output_dir / "audit-summary.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def read_rows(self) -> list[dict]:
        with self.reviewed_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            return list(csv.DictReader(file_handle))

    def write_rows(self, rows: list[dict]) -> None:
        with self.reviewed_csv.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def approve_all(self) -> list[dict]:
        rows = self.read_rows()
        for row in rows:
            row.update(
                {
                    "status": "pass",
                    "decision": "correct",
                    "reviewer": "reviewer-a",
                    "reviewed_at": "2026-06-13",
                }
            )
        self.write_rows(rows)
        return rows

    def import_review(self) -> dict:
        importer = getattr(self.audit, "import_audit_review", None)
        self.assertTrue(
            callable(importer),
            "import_audit_review is not implemented",
        )
        return importer(
            source_jsonl=self.source_jsonl,
            reviewed_csv=self.reviewed_csv,
            issues_path=self.issues_path,
            summary_path=self.summary_path,
        )

    def test_imports_complete_review_and_marks_ready(self) -> None:
        self.approve_all()

        summary = self.import_review()

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["reviewed_count"], 140)
        self.assertEqual(summary["pending_count"], 0)
        self.assertEqual(summary["pass_count"], 140)
        self.assertEqual(summary["fail_count"], 0)
        self.assertEqual(summary["reviewers"], ["reviewer-a"])
        self.assertEqual(self.issues_path.read_text(encoding="utf-8"), "")

    def test_valid_failure_marks_blocked_and_writes_issue(self) -> None:
        rows = self.approve_all()
        rows[0].update(
            {
                "status": "fail",
                "decision": "boundary_error",
                "comment": "Clause begins one line too early.",
            }
        )
        self.write_rows(rows)

        summary = self.import_review()

        self.assertEqual(summary["status"], "blocked")
        self.assertEqual(summary["fail_count"], 1)
        self.assertEqual(
            summary["unresolved_error_counts"]["boundary_error"],
            1,
        )
        issues = [
            json.loads(line)
            for line in self.issues_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["decision"], "boundary_error")

    def test_rejects_incomplete_or_semantically_invalid_review(self) -> None:
        invalid_mutations = {
            "row_count": lambda rows: rows[:-1],
            "duplicate_id": lambda rows: [
                dict(rows[0], audit_id=rows[1]["audit_id"]),
                *rows[1:],
            ],
            "unknown_id": lambda rows: [
                dict(rows[0], audit_id="unknown-audit-id"),
                *rows[1:],
            ],
            "pending": lambda rows: [
                dict(rows[0], status="pending"),
                *rows[1:],
            ],
            "pass_wrong_decision": lambda rows: [
                dict(rows[0], decision="type_error"),
                *rows[1:],
            ],
            "fail_without_error": lambda rows: [
                dict(rows[0], status="fail", decision="correct", comment="bad"),
                *rows[1:],
            ],
            "fail_without_comment": lambda rows: [
                dict(
                    rows[0],
                    status="fail",
                    decision="type_error",
                    comment="",
                ),
                *rows[1:],
            ],
            "missing_reviewer": lambda rows: [
                dict(rows[0], reviewer=""),
                *rows[1:],
            ],
            "missing_reviewed_at": lambda rows: [
                dict(rows[0], reviewed_at=""),
                *rows[1:],
            ],
        }
        for name, mutate in invalid_mutations.items():
            with self.subTest(case=name):
                rows = self.approve_all()
                self.write_rows(mutate(rows))
                with self.assertRaises(ValueError):
                    self.import_review()

    def test_rejects_changes_to_immutable_columns(self) -> None:
        immutable_mutations = {
            "book_id": "changed-book",
            "chapter_id": "changed-chapter",
            "clause_id": "changed-clause",
            "evidence_ids": "changed-evidence",
            "original_text": "changed original",
            "structured_summary": "changed summary",
            "sample_type": "formula",
        }
        for field, value in immutable_mutations.items():
            with self.subTest(field=field):
                rows = self.approve_all()
                rows[0][field] = value
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    self.import_review()

    def test_freezes_quality_gate_with_input_hashes(self) -> None:
        summary = None
        self.approve_all()
        summary = self.import_review()
        chunks_dir = self.root / "chunks"
        chunks_dir.mkdir()
        strategies = {}
        for strategy in ("c0", "c1", "c2", "c3", "c4"):
            chunk_path = chunks_dir / f"{strategy}.jsonl"
            chunk_path.write_text(
                json.dumps({"strategy": strategy}) + "\n",
                encoding="utf-8",
            )
            strategies[strategy] = {
                "output_file": chunk_path.name,
                "output_sha256": self.audit._sha256_file(chunk_path),
            }
        chunk_manifest_path = self.root / "chunks-manifest.json"
        chunk_manifest_path.write_text(
            json.dumps({"strategies": strategies}),
            encoding="utf-8",
        )
        quality_gate_path = self.root / "quality-gate.json"

        freezer = getattr(self.audit, "freeze_quality_gate", None)
        self.assertTrue(
            callable(freezer),
            "freeze_quality_gate is not implemented",
        )
        gate = freezer(
            summary=summary,
            source_jsonl=self.source_jsonl,
            reviewed_csv=self.reviewed_csv,
            evidence_path=self.evidence_path,
            chunks_dir=chunks_dir,
            chunk_manifest_path=chunk_manifest_path,
            quality_gate_path=quality_gate_path,
        )

        self.assertEqual(gate["status"], "ready")
        self.assertEqual(gate["reviewed_count"], 140)
        self.assertEqual(set(gate["chunks"]), {"c0", "c1", "c2", "c3", "c4"})
        self.assertEqual(
            json.loads(quality_gate_path.read_text(encoding="utf-8")),
            gate,
        )


class AuditMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.audit = importlib.import_module("experiments.rag_v1_5.audit")
        self.previous_source = self.root / "previous.jsonl"
        self.previous_reviewed_csv = self.root / "previous.csv"
        self.new_source = self.root / "new.jsonl"
        self.output_csv = self.root / "migrated.csv"
        self.summary_path = self.root / "migration-summary.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_previous(self, records: list[AuditRecord]) -> None:
        reviewed = [
            record.model_copy(
                update={
                    "status": "pass",
                    "decision": "correct",
                    "reviewer": "reviewer-a",
                    "reviewed_at": "2026-06-14",
                    "comment": f"reviewed {record.audit_id}",
                }
            )
            for record in records
        ]
        self.audit._write_jsonl(self.previous_source, records)
        self.audit._write_csv(self.previous_reviewed_csv, reviewed)

    def write_new(self, records: list[AuditRecord]) -> None:
        self.audit._write_jsonl(self.new_source, records)

    def migrate(self) -> dict:
        migrate_audit_review = getattr(
            self.audit,
            "migrate_audit_review",
            None,
        )
        self.assertTrue(
            callable(migrate_audit_review),
            "migrate_audit_review is not implemented",
        )
        return migrate_audit_review(
            previous_source_jsonl=self.previous_source,
            previous_reviewed_csv=self.previous_reviewed_csv,
            new_source_jsonl=self.new_source,
            output_csv=self.output_csv,
            summary_path=self.summary_path,
        )

    def read_output_rows(self) -> list[dict[str, str]]:
        with self.output_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            return list(csv.DictReader(file_handle))

    def test_inherits_only_exact_structure_matches(self) -> None:
        old_same = make_audit_record(
            audit_id="audit-old-001",
            clause_id="jgy-chapter-01-001",
        )
        old_changed = make_audit_record(
            audit_id="audit-old-002",
            clause_id="jgy-chapter-01-002",
        )
        self.write_previous([old_same, old_changed])
        new_same = old_same.model_copy(
            update={"audit_id": "audit-new-010"}
        )
        new_changed = old_changed.model_copy(
            update={
                "audit_id": "audit-new-011",
                "structured_summary": "[formula] corrected",
            }
        )
        new_missing = make_audit_record(
            audit_id="audit-new-012",
            clause_id="jgy-chapter-01-003",
        )
        self.write_new([new_same, new_changed, new_missing])

        summary = self.migrate()
        rows = {
            row["audit_id"]: row for row in self.read_output_rows()
        }

        self.assertEqual(summary["inherited_count"], 1)
        self.assertEqual(summary["reset_count"], 2)
        self.assertEqual(summary["missing_count"], 1)
        self.assertEqual(summary["ambiguous_count"], 0)
        self.assertEqual(rows["audit-new-010"]["status"], "pass")
        self.assertEqual(
            rows["audit-new-010"]["comment"],
            "reviewed audit-old-001",
        )
        self.assertEqual(rows["audit-new-011"]["status"], "pending")
        self.assertEqual(rows["audit-new-011"]["decision"], "")
        self.assertEqual(rows["audit-new-011"]["reviewer"], "")
        self.assertEqual(
            rows["audit-new-011"]["structured_summary"],
            "[formula] corrected",
        )
        self.assertEqual(rows["audit-new-012"]["status"], "pending")
        self.assertEqual(
            summary["pending_audit_ids"],
            ["audit-new-011", "audit-new-012"],
        )
        self.assertEqual(
            json.loads(self.summary_path.read_text(encoding="utf-8")),
            summary,
        )

    def test_duplicate_semantic_key_is_ambiguous(self) -> None:
        first = make_audit_record(
            audit_id="audit-old-001",
            clause_id="jgy-chapter-01-001",
        )
        second = first.model_copy(
            update={"audit_id": "audit-old-002"}
        )
        self.write_previous([first, second])
        self.write_new(
            [
                first.model_copy(
                    update={"audit_id": "audit-new-001"}
                )
            ]
        )

        summary = self.migrate()
        row = self.read_output_rows()[0]

        self.assertEqual(summary["inherited_count"], 0)
        self.assertEqual(summary["ambiguous_count"], 1)
        self.assertEqual(summary["reset_count"], 1)
        self.assertEqual(row["status"], "pending")

    def test_rejects_tampered_previous_immutable_columns(self) -> None:
        record = make_audit_record(
            audit_id="audit-old-001",
            clause_id="jgy-chapter-01-001",
        )
        self.write_previous([record])
        self.write_new(
            [
                record.model_copy(
                    update={"audit_id": "audit-new-001"}
                )
            ]
        )
        with self.previous_reviewed_csv.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            rows = list(csv.DictReader(file_handle))
        rows[0]["evidence_ids"] = "tampered-evidence"
        with self.previous_reviewed_csv.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(ValueError, "evidence_ids"):
            self.migrate()


if __name__ == "__main__":
    unittest.main()
