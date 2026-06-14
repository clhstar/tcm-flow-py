import json
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from experiments.rag_v1_5.retrieval import RetrievalResult
from experiments.rag_v1_5.schema import PilotQuestion, RetrievalHit
from tests.rag_v1_5.test_dataset import make_pilot_artifacts


FIXED_NOW = datetime(2026, 6, 14, 15, 30, 0, tzinfo=timezone.utc)


def approved_questions() -> list[PilotQuestion]:
    questions, _, _ = make_pilot_artifacts()
    return [
        PilotQuestion.model_validate(
            {**question, "review_status": "approved"}
        )
        for question in questions
    ]


def fake_validated_inputs(
    questions: list[PilotQuestion] | None = None,
    *,
    status: str = "ready",
) -> dict:
    config = {
        "version": "v1.5.0",
        "bm25": {"top_k": 20},
        "dense": {"top_k": 20},
        "rrf": {"k": 60},
        "reranker": {"candidate_k": 40, "top_k": 5},
        "evaluation": {"top_ks": [1, 5, 10]},
    }
    return {
        "pilot_manifest_status": status,
        "questions": questions or approved_questions(),
        "config": config,
        "input_hashes": {
            "pilot_manifest_sha256": "1" * 64,
            "dataset_sha256": "2" * 64,
            "evidence_group_sha256": "3" * 64,
            "review_summary_sha256": "4" * 64,
            "quality_gate_sha256": "5" * 64,
            "smoke_manifest_sha256": "6" * 64,
            "index_manifest_sha256": "7" * 64,
            "model_manifest_sha256": "8" * 64,
            "config_sha256": "9" * 64,
            "evidence_sha256": "A" * 64,
            "chunk_manifest_sha256": "B" * 64,
            "strategy_manifest_sha256": {
                strategy: character * 64
                for strategy, character in zip(
                    ("c0", "c1", "c2", "c3", "c4"),
                    "CDEFG",
                )
            },
        },
        "paths": {
            "model_manifest": Path("models.json"),
        },
    }


def fake_hit(
    *,
    strategy: str,
    question: PilotQuestion,
    rank: int,
) -> RetrievalHit:
    clause_id = (
        question.gold_clause_ids[0]
        if question.answerable
        else f"{strategy}-unanswerable-{rank}"
    )
    return RetrievalHit(
        chunk_id=f"{strategy}-{question.question_id}-{rank:02d}",
        strategy=strategy,
        rank=rank,
        text=f"child text {rank}",
        context_text=f"complete context {rank}",
        source_evidence_ids=[
            (
                question.gold_evidence_ids[0]
                if question.answerable
                else f"{strategy}-evidence-{rank}"
            )
        ],
        clause_ids=[clause_id],
        retrieval_parent_id=(
            clause_id if strategy == "c4" else None
        ),
        reranker_score=1.0 / rank,
    )


class FakeRuntime:
    def __init__(
        self,
        *,
        strategy: str,
        fail_question_id: str | None = None,
        interrupt_after: int | None = None,
        seen: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.fail_question_id = fail_question_id
        self.interrupt_after = interrupt_after
        self.seen = seen if seen is not None else []

    def retrieve(self, question: PilotQuestion) -> RetrievalResult:
        if (
            self.interrupt_after is not None
            and len(self.seen) >= self.interrupt_after
        ):
            raise KeyboardInterrupt("simulated interruption")
        self.seen.append(question.question_id)
        if question.question_id == self.fail_question_id:
            raise ValueError("simulated retrieval failure")
        hits = [
            fake_hit(
                strategy=self.strategy,
                question=question,
                rank=rank,
            )
            for rank in range(1, 11)
        ]
        return RetrievalResult(
            hits=hits,
            latency={
                "bm25_ms": 1.0,
                "dense_ms": 2.0,
                "rrf_ms": 3.0,
                "reranker_ms": 4.0,
                "total_ms": 10.5,
                "returned_context_chars": sum(
                    len(hit.context_text) for hit in hits[:5]
                ),
            },
        )


class FakeRuntimeFactory:
    def __init__(
        self,
        *,
        fail_question_id: str | None = None,
        interrupt_config: str | None = None,
        interrupt_after: int | None = None,
    ) -> None:
        self.fail_question_id = fail_question_id
        self.interrupt_config = interrupt_config
        self.interrupt_after = interrupt_after
        self.calls = []
        self.seen_by_config: dict[str, list[str]] = {}

    def __call__(
        self,
        config_id: str,
        strategy: str,
        mode: str,
        config: dict,
    ):
        from experiments.rag_v1_5.runner import PilotConfigRuntime

        self.calls.append((config_id, strategy, mode))
        seen = self.seen_by_config.setdefault(config_id, [])
        runtime = FakeRuntime(
            strategy=strategy,
            fail_question_id=self.fail_question_id,
            interrupt_after=(
                self.interrupt_after
                if config_id == self.interrupt_config
                else None
            ),
            seen=seen,
        )
        return PilotConfigRuntime(
            retrieve=runtime.retrieve,
            index_load_ms=11.0,
            embedding_load_ms=12.0 if "dense" in mode else 0.0,
            reranker_load_ms=(
                13.0 if mode == "hybrid_rerank" else 0.0
            ),
            warmup_ms=14.0,
            index_size_bytes=1024,
        )


class RunnerTests(unittest.TestCase):
    def run_matrix(
        self,
        root: Path,
        *,
        runtime_factory=None,
        validated_inputs: dict | None = None,
        resume_dir: Path | None = None,
        now: datetime = FIXED_NOW,
    ) -> dict:
        from experiments.rag_v1_5.runner import run_pilot_matrix

        validated = validated_inputs or fake_validated_inputs()

        def validator(**kwargs):
            return validated

        return run_pilot_matrix(
            dataset_path=root / "pilot-40.jsonl",
            evidence_groups_path=root / "groups.jsonl",
            pilot_manifest_path=root / "pilot-manifest.json",
            config_path=root / "config.yaml",
            indexes_dir=root / "indexes",
            output_dir=root / "runs",
            resume_dir=resume_dir,
            input_validator=validator,
            runtime_factory=runtime_factory or FakeRuntimeFactory(),
            now=now,
        )

    def test_new_run_writes_fixed_matrix_and_complete_config_artifacts(
        self,
    ) -> None:
        from experiments.rag_v1_5.runner import PILOT_MATRIX

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            factory = FakeRuntimeFactory()
            summary = self.run_matrix(root, runtime_factory=factory)
            matrix_dir = Path(summary["matrix_dir"])
            matrix_config = json.loads(
                (matrix_dir / "matrix-config.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["config_count"], 8)
            self.assertEqual(len(factory.calls), 8)
            self.assertEqual(
                [entry[0] for entry in PILOT_MATRIX],
                [entry["config_id"] for entry in matrix_config["matrix"]],
            )
            self.assertRegex(
                matrix_dir.name,
                r"^pilot-\d{8}T\d{6}Z-[0-9A-F]{8}-[0-9A-F]{8}$",
            )
            for config_id, strategy, mode in PILOT_MATRIX:
                config_dir = matrix_dir / config_id
                for filename in (
                    "run-config.json",
                    "per-question.jsonl",
                    "metrics.json",
                    "latency.json",
                    "errors.jsonl",
                ):
                    self.assertTrue((config_dir / filename).is_file())
                run_config = json.loads(
                    (config_dir / "run-config.json").read_text(
                        encoding="utf-8"
                    )
                )
                records = [
                    json.loads(line)
                    for line in (config_dir / "per-question.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                metrics = json.loads(
                    (config_dir / "metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertRegex(
                    run_config["run_id"],
                    (
                        r"^\d{8}T\d{6}Z-"
                        + re.escape(config_id)
                        + r"-[0-9A-F]{8}$"
                    ),
                )
                self.assertEqual(run_config["config"], matrix_config["config"])
                self.assertEqual(
                    run_config["input_hashes"],
                    matrix_config["input_hashes"],
                )
                self.assertEqual(len(records), 40)
                self.assertTrue(
                    all(
                        set(record["latency"])
                        == {
                            "bm25_ms",
                            "dense_ms",
                            "rrf_ms",
                            "reranker_ms",
                            "total_ms",
                            "returned_context_chars",
                        }
                        for record in records
                    )
                )
                self.assertTrue(
                    all(record["top5_traceability_ok"] for record in records)
                )
                self.assertEqual(metrics["completed_count"], 40)
                self.assertEqual(metrics["error_count"], 0)
                self.assertEqual(metrics["status"], "completed")
                self.assertEqual(
                    metrics["top5_traceability_rate"],
                    1.0,
                )
                self.assertEqual(
                    metrics["c4_parent_recovery_rate"],
                    1.0 if strategy == "c4" else None,
                )

    def test_resume_skips_completed_rows_and_rejects_corruption(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            interrupted = FakeRuntimeFactory(
                interrupt_config="c0-hybrid-rerank",
                interrupt_after=5,
            )
            with self.assertRaises(KeyboardInterrupt):
                self.run_matrix(root, runtime_factory=interrupted)
            matrix_dir = next((root / "runs").iterdir())
            completed_before = [
                json.loads(line)["question_id"]
                for line in (
                    matrix_dir
                    / "c0-hybrid-rerank"
                    / "per-question.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(completed_before), 5)

            resumed = FakeRuntimeFactory()
            summary = self.run_matrix(
                root,
                runtime_factory=resumed,
                resume_dir=matrix_dir,
            )
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(
                set(completed_before).isdisjoint(
                    resumed.seen_by_config["c0-hybrid-rerank"]
                )
            )
            self.assertEqual(
                len(resumed.seen_by_config["c0-hybrid-rerank"]),
                35,
            )

            per_question_path = (
                matrix_dir
                / "c0-hybrid-rerank"
                / "per-question.jsonl"
            )
            first_line = per_question_path.read_text(
                encoding="utf-8"
            ).splitlines()[0]
            with per_question_path.open("a", encoding="utf-8") as handle:
                handle.write(first_line + "\n")
            with self.assertRaisesRegex(ValueError, "重复 question_id"):
                self.run_matrix(
                    root,
                    runtime_factory=FakeRuntimeFactory(),
                    resume_dir=matrix_dir,
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = self.run_matrix(root)
            matrix_dir = Path(summary["matrix_dir"])
            per_question_path = (
                matrix_dir / "c0-hybrid-rerank" / "per-question.jsonl"
            )
            per_question_path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSONL"):
                self.run_matrix(root, resume_dir=matrix_dir)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary = self.run_matrix(root)
            matrix_dir = Path(summary["matrix_dir"])
            matrix_config_path = matrix_dir / "matrix-config.json"
            matrix_config = json.loads(
                matrix_config_path.read_text(encoding="utf-8")
            )
            matrix_config["input_hashes"]["dataset_sha256"] = "F" * 64
            matrix_config_path.write_text(
                json.dumps(matrix_config),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "输入哈希"):
                self.run_matrix(root, resume_dir=matrix_dir)

    def test_new_run_never_overwrites_and_errors_are_not_scored_as_misses(
        self,
    ) -> None:
        questions = approved_questions()
        failed_question = questions[0]
        validated = fake_validated_inputs(questions)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = self.run_matrix(
                root,
                runtime_factory=FakeRuntimeFactory(
                    fail_question_id=failed_question.question_id
                ),
                validated_inputs=validated,
            )
            matrix_dir = Path(first["matrix_dir"])
            metrics = json.loads(
                (
                    matrix_dir
                    / "c0-hybrid-rerank"
                    / "metrics.json"
                ).read_text(encoding="utf-8")
            )
            errors = [
                json.loads(line)
                for line in (
                    matrix_dir
                    / "c0-hybrid-rerank"
                    / "errors.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(first["status"], "failed")
            self.assertEqual(metrics["completed_count"], 39)
            self.assertEqual(metrics["error_count"], 1)
            self.assertEqual(metrics["answerable_question_count"], 31)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["attempt"], 1)
            self.assertEqual(
                metrics["no_answer_score_distribution"]["top1"][
                    "count"
                ],
                8,
            )
            with self.assertRaises(FileExistsError):
                self.run_matrix(
                    root,
                    validated_inputs=validated,
                    now=FIXED_NOW,
                )

    def test_rejects_non_ready_or_incomplete_question_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "ready"):
                self.run_matrix(
                    root,
                    validated_inputs=fake_validated_inputs(
                        status="blocked"
                    ),
                )
            with self.assertRaisesRegex(ValueError, "40"):
                self.run_matrix(
                    root,
                    validated_inputs=fake_validated_inputs(
                        approved_questions()[:-1]
                    ),
                    now=datetime(
                        2026,
                        6,
                        14,
                        15,
                        31,
                        tzinfo=timezone.utc,
                    ),
                )


class RunnerCliTests(unittest.TestCase):
    def test_run_pilot_cli_contract_matches_plan(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        args = build_parser().parse_args(["run-pilot"])

        self.assertEqual(args.command, "run-pilot")
        self.assertEqual(
            args.dataset,
            Path("data/rag_v1_5/evaluation/pilot-40.jsonl"),
        )
        self.assertEqual(
            args.pilot_manifest,
            Path(
                "experiments/rag_v1_5/manifests/"
                "pilot-40-v1.5.0.json"
            ),
        )
        self.assertEqual(
            args.output_dir,
            Path("data/rag_v1_5/runs/pilot"),
        )
        self.assertIsNone(args.resume)


class RuntimeInputGateTests(unittest.TestCase):
    def test_real_gate_rejects_cuda_unavailable_before_runtime(self) -> None:
        from experiments.rag_v1_5.runner import (
            validate_pilot_runtime_inputs,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pilot_manifest_path = root / "pilot.json"
            pilot_manifest_path.write_text(
                json.dumps({"status": "ready"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "CUDA"):
                validate_pilot_runtime_inputs(
                    dataset_path=root / "dataset.jsonl",
                    evidence_groups_path=root / "groups.jsonl",
                    pilot_manifest_path=pilot_manifest_path,
                    config_path=root / "config.yaml",
                    indexes_dir=root / "indexes",
                    repository_root=root,
                    cuda_checker=lambda: False,
                )


if __name__ == "__main__":
    unittest.main()
