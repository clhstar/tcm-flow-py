import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.rag_v1_5.audit import CSV_FIELDS
from experiments.rag_v1_5.cli import main
from experiments.rag_v1_5.corpus import CorpusFileSpec
from experiments.rag_v1_5.schema import AuditRecord


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class RagExperimentCliTests(unittest.TestCase):
    def test_prepares_and_parses_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            manifest_path = root / "corpus-v1.5.0.json"
            source_dir.mkdir()

            source_text = (
                FIXTURES_DIR / "shang_han_sample.txt"
            ).read_text(encoding="utf-8")
            source_bytes = source_text.encode("cp936")
            source_path = source_dir / "457-伤寒论.txt"
            source_path.write_bytes(source_bytes)
            source_hash = hashlib.sha256(source_bytes).hexdigest().upper()
            specs = [
                CorpusFileSpec(
                    book_id="shang_han_lun",
                    book_title="伤寒论",
                    source_filename=source_path.name,
                    expected_sha256=source_hash,
                )
            ]

            prepare_exit = main(
                [
                    "prepare-corpus",
                    "--source-dir",
                    str(source_dir),
                    "--raw-dir",
                    str(raw_dir),
                    "--manifest",
                    str(manifest_path),
                ],
                corpus_specs=specs,
            )
            parse_exit = main(
                [
                    "parse-corpus",
                    "--raw-dir",
                    str(raw_dir),
                    "--manifest",
                    str(manifest_path),
                    "--processed-dir",
                    str(processed_dir),
                ]
            )

            self.assertEqual(prepare_exit, 0)
            self.assertEqual(parse_exit, 0)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue((processed_dir / "evidence.jsonl").is_file())

    def test_builds_all_chunk_strategies_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "chunks"
            chunk_manifest_path = root / "chunks-v1.5.0.json"
            corpus_manifest_path = root / "corpus-v1.5.0.json"
            corpus_manifest_path.write_text(
                '{"corpus_version":"v1.5.0"}\n',
                encoding="utf-8",
            )
            evidence_path = FIXTURES_DIR / "evidence_sample.jsonl"
            config_path = (
                Path(__file__).parents[2]
                / "experiments"
                / "rag_v1_5"
                / "configs"
                / "chunks.yaml"
            )
            args = [
                "build-chunks",
                "--evidence",
                str(evidence_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--manifest",
                str(chunk_manifest_path),
                "--corpus-manifest",
                str(corpus_manifest_path),
            ]

            first_exit = main(args)
            first_outputs = {
                strategy: (output_dir / f"{strategy}.jsonl").read_bytes()
                for strategy in ("c0", "c1", "c2", "c3", "c4")
            }
            second_exit = main(args)

            self.assertEqual(first_exit, 0)
            self.assertEqual(second_exit, 0)
            self.assertTrue((output_dir / "statistics.json").is_file())
            self.assertTrue(chunk_manifest_path.is_file())
            self.assertEqual(
                first_outputs,
                {
                    strategy: (
                        output_dir / f"{strategy}.jsonl"
                    ).read_bytes()
                    for strategy in ("c0", "c1", "c2", "c3", "c4")
                },
            )

            manifest = json.loads(
                chunk_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["version"], "v1.5.0")
            self.assertEqual(
                manifest["evidence_sha256"],
                hashlib.sha256(evidence_path.read_bytes())
                .hexdigest()
                .upper(),
            )
            self.assertEqual(
                set(manifest["strategies"]),
                {"c0", "c1", "c2", "c3", "c4"},
            )
            for strategy, entry in manifest["strategies"].items():
                self.assertGreater(entry["count"], 0)
                self.assertEqual(
                    entry["output_file"],
                    f"{strategy}.jsonl",
                )
                self.assertEqual(len(entry["output_sha256"]), 64)

    def test_migrates_audit_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            previous_source = root / "previous.jsonl"
            previous_csv = root / "previous.csv"
            new_source = root / "new.jsonl"
            output_csv = root / "output.csv"
            summary_path = root / "summary.json"
            old_record = AuditRecord(
                audit_id="audit-old-001",
                book_id="jin_gui_yao_lue",
                sample_type="formula",
                chapter_id="jgy-chapter-01",
                clause_id="jgy-chapter-01-001",
                evidence_ids=[
                    "jgy-chapter-01-001",
                    "jgy-chapter-01-001-formula-01",
                ],
                original_text="[clause] test",
                structured_summary="[formula] test",
            )
            new_record = old_record.model_copy(
                update={"audit_id": "audit-new-001"}
            )
            previous_source.write_text(
                json.dumps(
                    old_record.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            new_source.write_text(
                json.dumps(
                    new_record.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            reviewed_row = old_record.model_copy(
                update={
                    "status": "pass",
                    "decision": "correct",
                    "reviewer": "reviewer-a",
                    "reviewed_at": "2026-06-14",
                }
            ).model_dump(mode="json")
            reviewed_row["evidence_ids"] = "|".join(
                reviewed_row["evidence_ids"]
            )
            with previous_csv.open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as file_handle:
                writer = csv.DictWriter(
                    file_handle,
                    fieldnames=CSV_FIELDS,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(reviewed_row)

            exit_code = main(
                [
                    "migrate-audit-review",
                    "--previous-source",
                    str(previous_source),
                    "--previous-reviewed-csv",
                    str(previous_csv),
                    "--new-source",
                    str(new_source),
                    "--output-csv",
                    str(output_csv),
                    "--summary",
                    str(summary_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            with output_csv.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file_handle:
                rows = list(csv.DictReader(file_handle))
            self.assertEqual(rows[0]["audit_id"], "audit-new-001")
            self.assertEqual(rows[0]["status"], "pass")
            self.assertEqual(
                json.loads(
                    summary_path.read_text(encoding="utf-8")
                )["inherited_count"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
