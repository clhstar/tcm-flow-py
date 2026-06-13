import csv
import importlib
import importlib.util
import io
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from experiments.rag_v1_5.schema import EvidenceUnit, ParseAnomaly


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


if __name__ == "__main__":
    unittest.main()
