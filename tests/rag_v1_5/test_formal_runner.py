import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from experiments.rag_v1_5.retrieval import RetrievalResult
from experiments.rag_v1_5.schema import PilotQuestion, RetrievalHit
from tests.rag_v1_5.test_formal_dataset import make_formal_artifacts


FIXED_NOW = datetime(2026, 6, 15, 10, 0, 0, tzinfo=timezone.utc)


def approved_formal_questions() -> list[PilotQuestion]:
    questions, _, _ = make_formal_artifacts()
    return [PilotQuestion.model_validate(question) for question in questions]


def fake_validated_inputs() -> dict:
    return {
        "formal_manifest_status": "ready",
        "questions": approved_formal_questions(),
        "config": {
            "version": "v1.5.0",
            "bm25": {"top_k": 20},
            "dense": {"top_k": 20},
            "rrf": {"k": 60},
            "reranker": {"candidate_k": 40, "top_k": 10},
            "evaluation": {"top_ks": [1, 5, 10]},
        },
        "input_hashes": {
            "formal_manifest_sha256": "1" * 64,
            "prereg_manifest_sha256": "2" * 64,
            "dataset_sha256": "3" * 64,
            "config_sha256": "4" * 64,
            "chunk_manifest_sha256": "5" * 64,
            "index_manifest_sha256": "6" * 64,
            "model_manifest_sha256": "7" * 64,
            "strategy_manifest_sha256": {
                key: str(index % 10) * 64
                for index, key in enumerate(
                    (
                        "c0",
                        "c1",
                        "c2",
                        "c3",
                        "c4",
                        "c5",
                        "c4-no-title",
                    ),
                    start=1,
                )
            },
        },
        "paths": {"model_manifest": Path("models.json")},
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class FakeFormalRuntime:
    def __init__(self, row) -> None:
        self.row = row

    def retrieve(self, question: PilotQuestion) -> RetrievalResult:
        hits = []
        for rank in range(1, 11):
            clause_id = (
                question.gold_clause_ids[0]
                if question.answerable
                else f"none-{rank}"
            )
            text = f"child {rank}"
            context = (
                text
                if self.row.context_policy == "child"
                else f"parent context {rank}"
            )
            hits.append(
                RetrievalHit(
                    chunk_id=(
                        f"{self.row.strategy}-"
                        f"{question.question_id}-{rank}"
                    ),
                    strategy=self.row.strategy,
                    rank=rank,
                    text=text,
                    context_text=context,
                    source_evidence_ids=[f"evidence-{rank}"],
                    clause_ids=[clause_id],
                    retrieval_parent_id=f"parent-{rank}",
                    reranker_score=1.0 / rank,
                )
            )
        return RetrievalResult(
            hits=hits,
            latency={
                "bm25_ms": 1.0,
                "dense_ms": 2.0,
                "rrf_ms": 3.0,
                "reranker_ms": 4.0,
                "total_ms": 10.0,
                "returned_context_chars": sum(
                    len(hit.context_text) for hit in hits[:5]
                ),
            },
        )


class FakeFormalRuntimeFactory:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, row, config):
        from experiments.rag_v1_5.formal_runner import (
            FormalConfigRuntime,
        )

        self.calls.append(row.config_id)
        runtime = FakeFormalRuntime(row)
        return FormalConfigRuntime(
            retrieve=runtime.retrieve,
            index_load_ms=1.0,
            embedding_load_ms=2.0,
            reranker_load_ms=3.0,
            warmup_ms=4.0,
            index_size_bytes=1024,
        )


class FormalRunnerTests(unittest.TestCase):
    def test_matrix_has_14_unique_single_variable_configs(self) -> None:
        from experiments.rag_v1_5.formal_runner import (
            FORMAL_RETRIEVAL_MATRIX,
        )

        self.assertEqual(len(FORMAL_RETRIEVAL_MATRIX), 14)
        self.assertEqual(
            len({row.config_id for row in FORMAL_RETRIEVAL_MATRIX}),
            14,
        )
        for row in FORMAL_RETRIEVAL_MATRIX:
            self.assertTrue(row.paper_role)
            self.assertTrue(row.strategy)
            self.assertTrue(row.mode)
            self.assertIn(row.context_policy, {"parent", "child"})
            self.assertIn(
                row.metadata_policy,
                {"with_titles", "without_titles"},
            )

    def test_runs_only_requested_split_and_supports_exact_resume(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_runner import run_formal_matrix

        validated = fake_validated_inputs()

        def validator(**kwargs):
            return validated

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            formal_manifest_path = root / "formal-manifest.json"
            prereg_manifest_path = root / "prereg.json"
            formal_manifest_path.write_text(
                json.dumps({"status": "ready"}),
                encoding="utf-8",
            )
            prereg_manifest_path.write_text(
                json.dumps({"status": "ready"}),
                encoding="utf-8",
            )
            validated["input_hashes"]["formal_manifest_sha256"] = (
                hashlib.sha256(
                    formal_manifest_path.read_bytes()
                ).hexdigest().upper()
            )
            validated["input_hashes"]["prereg_manifest_sha256"] = (
                hashlib.sha256(
                    prereg_manifest_path.read_bytes()
                ).hexdigest().upper()
            )
            first = run_formal_matrix(
                split="formal_dev",
                dataset_path=root / "formal-400.jsonl",
                formal_manifest_path=formal_manifest_path,
                prereg_manifest_path=prereg_manifest_path,
                config_path=root / "config.yaml",
                indexes_dir=root / "indexes",
                output_dir=root / "runs",
                input_validator=validator,
                runtime_factory=FakeFormalRuntimeFactory(),
                now=FIXED_NOW,
            )
            matrix_dir = Path(first["matrix_dir"])
            resumed = run_formal_matrix(
                split="formal_dev",
                dataset_path=root / "formal-400.jsonl",
                formal_manifest_path=formal_manifest_path,
                prereg_manifest_path=prereg_manifest_path,
                config_path=root / "config.yaml",
                indexes_dir=root / "indexes",
                output_dir=root / "runs",
                resume_dir=matrix_dir,
                input_validator=validator,
                runtime_factory=FakeFormalRuntimeFactory(),
                now=FIXED_NOW,
            )
            from experiments.rag_v1_5.formal_runner import (
                freeze_formal_runs,
            )

            frozen = freeze_formal_runs(
                run_dir=matrix_dir,
                formal_manifest_path=formal_manifest_path,
                prereg_manifest_path=prereg_manifest_path,
                output_path=root / "formal-runs.json",
            )
            serialized = json.dumps(frozen, ensure_ascii=False)

        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["config_count"], 14)
        self.assertEqual(first["completed_config_count"], 14)
        self.assertEqual(first["total_question_runs"], 2800)
        self.assertEqual(resumed["matrix_dir"], first["matrix_dir"])
        for config in first["configs"]:
            self.assertEqual(config["completed_count"], 200)
            self.assertEqual(config["error_count"], 0)
            self.assertEqual(config["top5_traceability_rate"], 1.0)
        self.assertEqual(frozen["status"], "ready")
        self.assertEqual(frozen["config_count"], 14)
        self.assertNotIn("child 1", serialized)
        self.assertNotIn('"hits"', serialized)
        self.assertNotIn(
            approved_formal_questions()[0].question,
            serialized,
        )

    def test_rejects_bad_split_or_resume_hash_drift(self) -> None:
        from experiments.rag_v1_5.formal_runner import run_formal_matrix

        validated = fake_validated_inputs()

        def validator(**kwargs):
            return validated

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaises(ValueError):
                run_formal_matrix(
                    split="pilot",
                    dataset_path=root / "formal-400.jsonl",
                    formal_manifest_path=root / "formal-manifest.json",
                    prereg_manifest_path=root / "prereg.json",
                    config_path=root / "config.yaml",
                    indexes_dir=root / "indexes",
                    output_dir=root / "runs",
                    input_validator=validator,
                    runtime_factory=FakeFormalRuntimeFactory(),
                )

            first = run_formal_matrix(
                split="formal_test",
                dataset_path=root / "formal-400.jsonl",
                formal_manifest_path=root / "formal-manifest.json",
                prereg_manifest_path=root / "prereg.json",
                config_path=root / "config.yaml",
                indexes_dir=root / "indexes",
                output_dir=root / "runs",
                input_validator=validator,
                runtime_factory=FakeFormalRuntimeFactory(),
                now=FIXED_NOW,
            )
            validated["input_hashes"]["dataset_sha256"] = "F" * 64
            with self.assertRaises(ValueError):
                run_formal_matrix(
                    split="formal_test",
                    dataset_path=root / "formal-400.jsonl",
                    formal_manifest_path=root / "formal-manifest.json",
                    prereg_manifest_path=root / "prereg.json",
                    config_path=root / "config.yaml",
                    indexes_dir=root / "indexes",
                    output_dir=root / "runs",
                    resume_dir=Path(first["matrix_dir"]),
                    input_validator=validator,
                    runtime_factory=FakeFormalRuntimeFactory(),
                )

    def test_runtime_validation_rejects_index_from_other_formal_manifest(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_runner import (
            validate_formal_runtime_inputs,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "formal-400.jsonl"
            dataset_path.write_text(
                "".join(
                    json.dumps(
                        question.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                    for question in approved_formal_questions()
                ),
                encoding="utf-8",
            )
            config_path = root / "formal-400.yaml"
            config_path.write_text("version: v1.5.0\n", encoding="utf-8")
            prereg_path = root / "formal-prereg.json"
            formal_path = root / "formal-manifest.json"
            chunk_path = root / "chunks" / "manifest.json"
            chunk_path.parent.mkdir()
            chunk_path.write_text("{}\n", encoding="utf-8")
            model_path = (
                root
                / "experiments"
                / "rag_v1_5"
                / "manifests"
                / "models-v1.5.0.json"
            )
            model_path.parent.mkdir(parents=True)
            model_path.write_text("{}\n", encoding="utf-8")
            prereg_path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "inputs": {
                            "config": {"sha256": sha256(config_path)}
                        },
                        "models": {
                            "pilot_model_manifest_sha256": sha256(
                                model_path
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            formal_path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "dataset": {"sha256": sha256(dataset_path)},
                        "inputs": {
                            "prereg_manifest": {
                                "sha256": sha256(prereg_path)
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            indexes_dir = root / "indexes"
            strategy_records = {}
            for key in (
                "c0",
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
                "c4-no-title",
            ):
                strategy_path = indexes_dir / key / "manifest.json"
                strategy_path.parent.mkdir(parents=True)
                strategy_path.write_text("{}\n", encoding="utf-8")
                strategy_records[key] = {
                    "manifest_sha256": sha256(strategy_path)
                }
            (indexes_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "formal_manifest_sha256": "0" * 64,
                        "model_manifest_sha256": sha256(model_path),
                        "chunk_manifest": {
                            "path": chunk_path.relative_to(root).as_posix(),
                            "sha256": sha256(chunk_path),
                        },
                        "strategies": strategy_records,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                validate_formal_runtime_inputs(
                    dataset_path=dataset_path,
                    formal_manifest_path=formal_path,
                    prereg_manifest_path=prereg_path,
                    config_path=config_path,
                    indexes_dir=indexes_dir,
                    repository_root=root,
                    cuda_checker=lambda: True,
                )


class FormalRunnerCliTests(unittest.TestCase):
    def test_formal_runner_cli_contracts(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        dev = build_parser().parse_args(["run-formal-dev"])
        test = build_parser().parse_args(["run-formal-test"])
        chunks = build_parser().parse_args(["build-formal-chunks"])
        indexes = build_parser().parse_args(["build-formal-indexes"])
        run_dir = Path("data/rag_v1_5/formal/runs/test/example")
        summary = build_parser().parse_args(
            ["summarize-formal-test", "--run-dir", str(run_dir)]
        )
        freeze = build_parser().parse_args(
            ["freeze-formal-runs", "--run-dir", str(run_dir)]
        )

        self.assertEqual(dev.command, "run-formal-dev")
        self.assertEqual(
            chunks.output_dir,
            Path("data/rag_v1_5/formal/chunks"),
        )
        self.assertEqual(
            indexes.output_dir,
            Path("data/rag_v1_5/formal/indexes"),
        )
        self.assertEqual(dev.output_dir, Path("data/rag_v1_5/formal/runs/dev"))
        self.assertEqual(test.command, "run-formal-test")
        self.assertEqual(
            test.output_dir,
            Path("data/rag_v1_5/formal/runs/test"),
        )
        self.assertEqual(summary.run_dir, run_dir)
        self.assertEqual(freeze.run_dir, run_dir)
        self.assertEqual(
            freeze.output,
            Path(
                "experiments/rag_v1_5/manifests/"
                "formal-runs-v1.5.0.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
