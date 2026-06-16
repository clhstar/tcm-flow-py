import importlib
import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from experiments.rag_v1_5 import cli
from experiments.rag_v1_5 import schema


class ExperimentDependencyTests(unittest.TestCase):
    def test_transformers_is_pinned_to_flagembedding_compatible_v4(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        requirements = (
            repository_root / "requirements-experiment.txt"
        ).read_text(encoding="utf-8").splitlines()

        self.assertIn("transformers==4.57.6", requirements)


class RetrievalDoctorTests(unittest.TestCase):
    def test_reports_environment_and_gate_without_secrets(self) -> None:
        report_builder = getattr(cli, "build_retrieval_doctor_report", None)
        self.assertTrue(
            callable(report_builder),
            "retrieval-doctor report builder is not implemented",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks_dir = root / "chunks"
            chunks_dir.mkdir()
            for strategy in ("c0", "c1", "c2", "c3", "c4"):
                (chunks_dir / f"{strategy}.jsonl").write_text(
                    "{}\n",
                    encoding="utf-8",
                )

            report = report_builder(
                config_path=root / "retrieval-pilot.yaml",
                chunks_dir=chunks_dir,
                chunk_manifest_path=root / "chunks-manifest.json",
                quality_gate_path=root / "quality-gate.json",
                model_manifest_path=root / "models.json",
                indexes_dir=root / "indexes",
                system_reader=lambda: {
                    "python_version": "3.10.6",
                    "torch_version": "2.7.0+cu128",
                    "cuda_available": True,
                    "gpu_name": "NVIDIA GeForce RTX 2070",
                    "gpu_memory_mib": 8192,
                },
                package_version_reader=lambda package: {
                    "pydantic": "2.13.4",
                    "PyYAML": "6.0.3",
                    "numpy": "2.2.6",
                    "jieba": "0.42.1",
                    "rank-bm25": "0.2.2",
                    "FlagEmbedding": "1.4.0",
                }.get(package),
            )

        self.assertEqual(report["python_version"], "3.10.6")
        self.assertTrue(report["cuda_available"])
        self.assertEqual(report["gpu_name"], "NVIDIA GeForce RTX 2070")
        self.assertGreaterEqual(report["gpu_memory_mib"], 8192)
        self.assertEqual(report["chunk_strategy_count"], 5)
        self.assertEqual(report["quality_gate_status"], "missing")
        serialized = json.dumps(report).lower()
        self.assertNotIn("token", serialized)
        self.assertNotIn("api_key", serialized)

    def test_validates_formal_chunk_and_index_hash_chain(self) -> None:
        def sha256(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest().upper()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chunks_dir = root / "formal" / "chunks"
            indexes_dir = root / "formal" / "indexes"
            chunks_dir.mkdir(parents=True)
            indexes_dir.mkdir(parents=True)
            chunk_strategies = ("c0", "c1", "c2", "c3", "c4", "c5")
            chunk_records = {}
            for strategy in chunk_strategies:
                chunk_path = chunks_dir / f"{strategy}.jsonl"
                chunk_path.write_text("{}\n", encoding="utf-8")
                chunk_records[strategy] = {
                    "output_file": chunk_path.name,
                    "output_sha256": sha256(chunk_path),
                }
            chunk_manifest_path = chunks_dir / "manifest.json"
            chunk_manifest_path.write_text(
                json.dumps({"strategies": chunk_records}),
                encoding="utf-8",
            )
            formal_manifest_path = root / "formal-manifest.json"
            formal_manifest_path.write_text(
                json.dumps({"status": "ready"}),
                encoding="utf-8",
            )

            index_records = {}
            index_strategies = (*chunk_strategies, "c4-no-title")
            for strategy in index_strategies:
                strategy_dir = indexes_dir / strategy
                strategy_dir.mkdir()
                rows_path = strategy_dir / "rows.jsonl"
                rows_path.write_text("{}\n", encoding="utf-8")
                strategy_manifest_path = strategy_dir / "manifest.json"
                strategy_manifest_path.write_text(
                    json.dumps(
                        {
                            "files": {
                                "rows": {
                                    "path": rows_path.name,
                                    "sha256": sha256(rows_path),
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                index_records[strategy] = {
                    "manifest_sha256": sha256(
                        strategy_manifest_path
                    )
                }
            index_manifest_path = indexes_dir / "manifest.json"
            index_manifest_path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "formal_manifest_sha256": sha256(
                            formal_manifest_path
                        ),
                        "chunk_manifest": {
                            "sha256": sha256(chunk_manifest_path)
                        },
                        "strategies": index_records,
                    }
                ),
                encoding="utf-8",
            )

            report = cli.build_retrieval_doctor_report(
                config_path=root / "formal-400.yaml",
                chunks_dir=chunks_dir,
                chunk_manifest_path=chunk_manifest_path,
                quality_gate_path=root / "quality-gate.json",
                model_manifest_path=root / "models.json",
                indexes_dir=indexes_dir,
                index_manifest_path=index_manifest_path,
                formal_manifest_path=formal_manifest_path,
                expected_chunk_strategies=chunk_strategies,
                expected_index_strategies=index_strategies,
                system_reader=lambda: {
                    "python_version": "3.10.6",
                    "torch_version": "2.7.0+cu128",
                    "cuda_available": True,
                    "gpu_name": "NVIDIA GeForce RTX 2070",
                    "gpu_memory_mib": 8192,
                },
                package_version_reader=lambda package: None,
            )

        self.assertEqual(report["chunk_strategy_count"], 6)
        self.assertEqual(report["chunk_manifest_status"], "valid")
        self.assertEqual(report["index_strategy_count"], 7)
        self.assertEqual(report["index_manifest_status"], "valid")

    def test_cli_accepts_formal_doctor_profile(self) -> None:
        args = cli.build_parser().parse_args(
            ["retrieval-doctor", "--formal"]
        )

        self.assertTrue(args.formal)


class RetrievalSchemaTests(unittest.TestCase):
    def test_pilot_question_enforces_answerable_contract(self) -> None:
        pilot_question = getattr(schema, "PilotQuestion", None)
        self.assertIsNotNone(
            pilot_question,
            "PilotQuestion schema is not implemented",
        )

        valid = {
            "question_id": "q-001",
            "question": "Which herbs are listed?",
            "question_type": "formula_composition_or_use",
            "book_scope": "jin_gui_yao_lue",
            "answerable": True,
            "reference_answer": "Herb A and herb B.",
            "gold_evidence_ids": ["e-001"],
            "gold_clause_ids": ["c-001"],
            "graded_relevance": {"c-001": 2},
            "support_spans": ["Herb A Herb B"],
            "review_status": "approved",
        }
        pilot_question.model_validate(valid)

        for field in (
            "gold_evidence_ids",
            "gold_clause_ids",
            "support_spans",
        ):
            invalid = dict(valid)
            invalid[field] = []
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    pilot_question.model_validate(invalid)

    def test_pilot_question_enforces_unanswerable_and_relevance_contract(self) -> None:
        pilot_question = getattr(schema, "PilotQuestion", None)
        self.assertIsNotNone(
            pilot_question,
            "PilotQuestion schema is not implemented",
        )

        valid = {
            "question_id": "q-002",
            "question": "A question not covered by the corpus?",
            "question_type": "unanswerable",
            "book_scope": "both",
            "answerable": False,
            "reference_answer": "无答案",
            "gold_evidence_ids": [],
            "gold_clause_ids": [],
            "graded_relevance": {},
            "support_spans": [],
            "review_status": "approved",
        }
        pilot_question.model_validate(valid)

        invalid_gold = dict(valid)
        invalid_gold["gold_clause_ids"] = ["c-001"]
        with self.assertRaises(ValidationError):
            pilot_question.model_validate(invalid_gold)

        invalid_relevance = dict(valid)
        invalid_relevance["graded_relevance"] = {"c-001": 3}
        with self.assertRaises(ValidationError):
            pilot_question.model_validate(invalid_relevance)

        invalid_answer = dict(valid)
        invalid_answer["reference_answer"] = ""
        with self.assertRaises(ValidationError):
            pilot_question.model_validate(invalid_answer)

    def test_audit_record_and_retrieval_hit_forbid_extra_fields(self) -> None:
        audit_record = getattr(schema, "AuditRecord", None)
        retrieval_hit = getattr(schema, "RetrievalHit", None)
        self.assertIsNotNone(audit_record, "AuditRecord schema is not implemented")
        self.assertIsNotNone(
            retrieval_hit,
            "RetrievalHit schema is not implemented",
        )

        with self.assertRaises(ValidationError):
            audit_record.model_validate(
                {
                    "audit_id": "audit-001",
                    "book_id": "book",
                    "sample_type": "clause",
                    "chapter_id": "chapter",
                    "clause_id": "clause",
                    "evidence_ids": ["evidence"],
                    "original_text": "original",
                    "structured_summary": "summary",
                    "unexpected": True,
                }
            )

        hit = retrieval_hit.model_validate(
            {
                "chunk_id": "chunk",
                "strategy": "c4",
                "rank": 1,
                "text": "child",
                "context_text": "parent",
                "source_evidence_ids": ["evidence"],
                "clause_ids": ["clause"],
                "retrieval_parent_id": "clause",
            }
        )
        self.assertEqual(hit.chunk_id, "chunk")


class ModelStoreTests(unittest.TestCase):
    def test_prepares_fixed_revision_snapshots_and_writes_relative_manifest(
        self,
    ) -> None:
        module_spec = importlib.util.find_spec(
            "experiments.rag_v1_5.model_store"
        )
        self.assertIsNotNone(module_spec, "model_store module is not implemented")
        model_store = importlib.import_module(
            "experiments.rag_v1_5.model_store"
        )
        prepare_models = getattr(model_store, "prepare_models", None)
        self.assertTrue(callable(prepare_models), "prepare_models is missing")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository_root = root / "repo"
            repository_root.mkdir()
            config_path = repository_root / "retrieval-pilot.yaml"
            output_dir = repository_root / "data" / "rag_v1_5" / "models"
            manifest_path = repository_root / "models-v1.5.0.json"
            config = {
                "version": "v1.5.0",
                "embedding": {
                    "model": "BAAI/bge-m3",
                    "revision": "1" * 40,
                },
                "reranker": {
                    "model": "BAAI/bge-reranker-v2-m3",
                    "revision": "2" * 40,
                },
            }
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            calls: list[dict] = []

            def fake_snapshot_download(**kwargs):
                calls.append(kwargs)
                local_dir = Path(kwargs["local_dir"])
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "config.json").write_text(
                    json.dumps({"model": kwargs["repo_id"]}),
                    encoding="utf-8",
                )
                cache_dir = local_dir / ".cache"
                cache_dir.mkdir()
                (cache_dir / "ignored").write_text(
                    "cache",
                    encoding="utf-8",
                )
                return str(local_dir)

            manifest = prepare_models(
                config_path=config_path,
                output_dir=output_dir,
                manifest_path=manifest_path,
                repository_root=repository_root,
                snapshot_downloader=fake_snapshot_download,
                library_version_reader=lambda package: "1.4.0",
            )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["revision"], "1" * 40)
            self.assertEqual(calls[1]["revision"], "2" * 40)
            self.assertNotIn("\\", manifest["embedding"]["local_path"])
            self.assertFalse(
                Path(manifest["embedding"]["local_path"]).is_absolute()
            )
            self.assertEqual(
                manifest["embedding"]["files"][0]["path"],
                "config.json",
            )
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                manifest,
            )

            prepare_models(
                config_path=config_path,
                output_dir=output_dir,
                manifest_path=manifest_path,
                repository_root=repository_root,
                snapshot_downloader=fake_snapshot_download,
                library_version_reader=lambda package: "1.4.0",
            )
            self.assertEqual(len(calls), 2)

    def test_partial_snapshot_without_manifest_is_resumed(self) -> None:
        module_spec = importlib.util.find_spec(
            "experiments.rag_v1_5.model_store"
        )
        self.assertIsNotNone(module_spec, "model_store module is not implemented")
        model_store = importlib.import_module(
            "experiments.rag_v1_5.model_store"
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "retrieval-pilot.yaml"
            output_dir = root / "data" / "rag_v1_5" / "models"
            manifest_path = root / "models-v1.5.0.json"
            config = {
                "version": "v1.5.0",
                "embedding": {
                    "model": "BAAI/bge-m3",
                    "revision": "1" * 40,
                },
                "reranker": {
                    "model": "BAAI/bge-reranker-v2-m3",
                    "revision": "2" * 40,
                },
            }
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            partial_dir = output_dir / "bge-m3" / ("1" * 40)
            partial_dir.mkdir(parents=True)
            (partial_dir / "partial.bin").write_bytes(b"partial")
            calls = []

            def fake_snapshot_download(**kwargs):
                calls.append(kwargs)
                local_dir = Path(kwargs["local_dir"])
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "config.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return str(local_dir)

            model_store.prepare_models(
                config_path=config_path,
                output_dir=output_dir,
                manifest_path=manifest_path,
                repository_root=root,
                snapshot_downloader=fake_snapshot_download,
                library_version_reader=lambda package: "1.4.0",
            )

            self.assertEqual(len(calls), 2)

    def test_rejects_floating_model_revision(self) -> None:
        module_spec = importlib.util.find_spec(
            "experiments.rag_v1_5.model_store"
        )
        self.assertIsNotNone(module_spec, "model_store module is not implemented")
        model_store = importlib.import_module(
            "experiments.rag_v1_5.model_store"
        )
        validate_revision = getattr(model_store, "validate_revision", None)
        self.assertTrue(callable(validate_revision), "validate_revision is missing")

        for revision in ("main", "v1.0", "abc123"):
            with self.subTest(revision=revision):
                with self.assertRaises(ValueError):
                    validate_revision(revision)


if __name__ == "__main__":
    unittest.main()
