import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from experiments.rag_v1_5.corpus import CorpusFileSpec, prepare_corpus
from experiments.rag_v1_5.parser import parse_corpus_file
from experiments.rag_v1_5.pipeline import (
    parse_prepared_corpus,
    validate_evidence_graph,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class AncientBookParserTests(unittest.TestCase):
    def test_parses_chapters_clauses_and_stable_ids(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "jin_gui_sample.txt",
            book_id="jin_gui_yao_lue",
            book_title="金匮要略方论",
            source_hash="A" * 64,
        )

        clauses = [
            unit for unit in result.evidence_units
            if unit.content_type == "clause"
        ]

        self.assertEqual(len(clauses), 3)
        self.assertEqual(clauses[0].chapter_id, "jgy-chapter-01")
        self.assertEqual(clauses[0].clause_id, "jgy-chapter-01-001")
        self.assertEqual(clauses[2].chapter_id, "jgy-chapter-02")
        self.assertEqual(clauses[2].clause_id, "jgy-chapter-02-001")
        self.assertEqual(clauses[0].volume, "卷上")
        self.assertEqual(clauses[2].volume, "卷中")
        self.assertIn("\n", clauses[0].original_text)
        self.assertNotIn("\n", clauses[0].normalized_text)

    def test_extracts_formula_ingredients_preparation_and_note_children(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "jin_gui_sample.txt",
            book_id="jin_gui_yao_lue",
            book_title="金匮要略方论",
            source_hash="A" * 64,
        )
        by_type = {}
        for unit in result.evidence_units:
            by_type.setdefault(unit.content_type, []).append(unit)

        self.assertEqual(len(by_type["formula"]), 1)
        self.assertEqual(len(by_type["ingredients"]), 1)
        self.assertEqual(len(by_type["preparation"]), 1)
        self.assertEqual(len(by_type["note"]), 1)

        formula = by_type["formula"][0]
        ingredients = by_type["ingredients"][0]
        preparation = by_type["preparation"][0]
        note = by_type["note"][0]

        self.assertEqual(formula.normalized_text.splitlines()[0], "栝蒌桂枝汤方")
        self.assertIn("栝蒌根（二两）", ingredients.normalized_text)
        self.assertIn("上三味", preparation.normalized_text)
        self.assertEqual(formula.parent_id, "jgy-chapter-01-002")
        self.assertEqual(ingredients.parent_id, formula.evidence_id)
        self.assertEqual(preparation.parent_id, formula.evidence_id)
        self.assertEqual(note.parent_id, "jgy-chapter-01-001")
        self.assertEqual(note.normalized_text, "一云痉病。")

    def test_records_missing_character_markers_without_repairing_them(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "jin_gui_sample.txt",
            book_id="jin_gui_yao_lue",
            book_title="金匮要略方论",
            source_hash="A" * 64,
        )

        self.assertEqual(len(result.anomalies), 1)
        self.assertEqual(result.anomalies[0].reason, "missing_character_marker")
        self.assertIn("KT KT", result.anomalies[0].original_text)

        clause = next(
            unit for unit in result.evidence_units
            if unit.clause_id == "jgy-chapter-01-002"
            and unit.content_type == "clause"
        )
        self.assertIn("KT KT", clause.normalized_text)

    def test_writes_evidence_and_anomaly_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "evidence.jsonl"
            anomalies_path = Path(temp_dir) / "anomalies.jsonl"

            result = parse_corpus_file(
                input_path=FIXTURES_DIR / "jin_gui_sample.txt",
                book_id="jin_gui_yao_lue",
                book_title="金匮要略方论",
                source_hash="A" * 64,
                output_path=output_path,
                anomalies_path=anomalies_path,
            )

            evidence_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            anomaly_rows = [
                json.loads(line)
                for line in anomalies_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(evidence_rows), len(result.evidence_units))
            self.assertEqual(len(anomaly_rows), len(result.anomalies))
            self.assertEqual(
                evidence_rows[0]["source_file"],
                "jin_gui_sample.txt",
            )

    def test_parses_shang_han_numbered_clauses(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "shang_han_sample.txt",
            book_id="shang_han_lun",
            book_title="伤寒论",
            source_hash="B" * 64,
        )

        self.assertEqual(result.statistics.chapter_count, 1)
        self.assertEqual(result.statistics.clause_count, 3)
        self.assertEqual(
            result.evidence_units[0].normalized_text,
            "太阳之为病，脉浮、 头项强痛而恶寒。",
        )

    def test_extracts_implicit_shang_han_formula_structure(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "shang_han_sample.txt",
            book_id="shang_han_lun",
            book_title="伤寒论",
            source_hash="B" * 64,
        )
        formula = next(
            unit for unit in result.evidence_units
            if unit.content_type == "formula"
        )
        ingredients = next(
            unit for unit in result.evidence_units
            if unit.content_type == "ingredients"
        )
        preparation = next(
            unit for unit in result.evidence_units
            if unit.content_type == "preparation"
        )

        self.assertEqual(formula.normalized_text.splitlines()[0], "桂枝汤")
        self.assertEqual(formula.parent_id, "shl-chapter-01-003")
        self.assertIn("桂枝（去皮，三两）", ingredients.normalized_text)
        self.assertIn("上三味", preparation.normalized_text)

    def test_parses_prepared_manifest_into_combined_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            manifest_path = root / "corpus-v1.5.0.json"
            source_dir.mkdir()

            source_text = (
                FIXTURES_DIR / "jin_gui_sample.txt"
            ).read_text(encoding="utf-8")
            source_bytes = source_text.encode("cp936")
            source_file = source_dir / "499-金匮要略方论.txt"
            source_file.write_bytes(source_bytes)
            source_hash = hashlib.sha256(source_bytes).hexdigest().upper()
            prepare_corpus(
                source_dir=source_dir,
                output_dir=raw_dir,
                manifest_path=manifest_path,
                specs=[
                    CorpusFileSpec(
                        book_id="jin_gui_yao_lue",
                        book_title="金匮要略方论",
                        source_filename=source_file.name,
                        expected_sha256=source_hash,
                    )
                ],
                generated_at=datetime(
                    2026,
                    6,
                    12,
                    8,
                    0,
                    tzinfo=timezone.utc,
                ),
            )

            statistics = parse_prepared_corpus(
                raw_dir=raw_dir,
                manifest_path=manifest_path,
                processed_dir=processed_dir,
            )

            self.assertEqual(
                statistics["books"]["jin_gui_yao_lue"]["clause_count"],
                3,
            )
            self.assertTrue((processed_dir / "evidence.jsonl").is_file())
            self.assertTrue((processed_dir / "anomalies.jsonl").is_file())
            self.assertTrue((processed_dir / "statistics.json").is_file())

            rows = [
                json.loads(line)
                for line in (
                    processed_dir / "evidence.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                all(row["source_hash"] == source_hash for row in rows)
            )

    def test_rejects_duplicate_and_orphan_evidence_ids(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "jin_gui_sample.txt",
            book_id="jin_gui_yao_lue",
            book_title="金匮要略方论",
            source_hash="A" * 64,
        )

        with self.assertRaisesRegex(ValueError, "重复 evidence_id"):
            validate_evidence_graph(
                [*result.evidence_units, result.evidence_units[0]]
            )

        orphan = result.evidence_units[1].model_copy(
            update={"parent_id": "missing-parent"}
        )
        with self.assertRaisesRegex(ValueError, "找不到 parent_id"):
            validate_evidence_graph([result.evidence_units[0], orphan])


if __name__ == "__main__":
    unittest.main()
