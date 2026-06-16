import hashlib
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


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

    def completed_matrix(
        self,
        root: Path,
    ) -> tuple[Path, Path]:
        pilot_manifest_path = root / "pilot-manifest.json"
        validated = fake_validated_inputs()
        hashes = validated["input_hashes"]
        pilot_manifest = {
            "version": "v1.5.0",
            "status": "ready",
            "dataset": {
                "question_count": 40,
                "approved_count": 40,
                "sha256": hashes["dataset_sha256"],
            },
            "inputs": {
                name: {"path": f"{name}.json", "sha256": hashes[key]}
                for name, key in (
                    ("evidence_group", "evidence_group_sha256"),
                    ("review_summary", "review_summary_sha256"),
                    ("quality_gate", "quality_gate_sha256"),
                    ("smoke_manifest", "smoke_manifest_sha256"),
                    ("index_manifest", "index_manifest_sha256"),
                    ("model_manifest", "model_manifest_sha256"),
                    ("config", "config_sha256"),
                    ("evidence", "evidence_sha256"),
                    ("chunk_manifest", "chunk_manifest_sha256"),
                )
            },
            "privacy": {
                "full_corpus_committed": False,
                "full_questions_committed": False,
            },
        }
        pilot_manifest_path.write_text(
            json.dumps(pilot_manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        hashes["pilot_manifest_sha256"] = sha256_file(
            pilot_manifest_path
        )
        summary = self.run_matrix(root, validated_inputs=validated)
        return Path(summary["matrix_dir"]), pilot_manifest_path

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

    def test_freeze_pilot_runs_records_hashes_metrics_and_privacy(self) -> None:
        from experiments.rag_v1_5.runner import (
            PILOT_MATRIX,
            freeze_pilot_runs,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matrix_dir, pilot_manifest_path = self.completed_matrix(root)
            output_path = root / "pilot-runs.json"
            per_question_path = (
                matrix_dir
                / "c0-hybrid-rerank"
                / "per-question.jsonl"
            )
            rows = per_question_path.read_text(
                encoding="utf-8"
            ).splitlines()
            first_row = json.loads(rows[0])
            first_row["manual_comment"] = "PRIVATE_REVIEW_COMMENT"
            rows[0] = json.dumps(first_row, ensure_ascii=False)
            per_question_path.write_text(
                "\n".join(rows) + "\n",
                encoding="utf-8",
            )

            manifest = freeze_pilot_runs(
                run_dir=matrix_dir,
                pilot_manifest_path=pilot_manifest_path,
                output_path=output_path,
            )

            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["config_count"], 8)
            self.assertEqual(
                [record["config_id"] for record in manifest["configs"]],
                [config_id for config_id, _, _ in PILOT_MATRIX],
            )
            self.assertEqual(
                manifest["input_hashes"]["pilot_manifest_sha256"],
                sha256_file(pilot_manifest_path),
            )
            for config in manifest["configs"]:
                self.assertEqual(config["completed_count"], 40)
                self.assertEqual(config["error_count"], 0)
                self.assertEqual(
                    set(config["core_metrics"]),
                    {
                        "recall_at_1",
                        "recall_at_5",
                        "recall_at_10",
                        "hit_at_5",
                        "mrr_at_10",
                        "ndcg_at_10",
                    },
                )
                self.assertIn("total_ms", config["latency"])
                self.assertEqual(config["index_size_bytes"], 1024)
                self.assertEqual(len(config["files"]), 5)
                for filename, file_record in config["files"].items():
                    self.assertEqual(
                        file_record["sha256"],
                        sha256_file(
                            matrix_dir / config["config_id"] / filename
                        ),
                    )

            serialized = output_path.read_text(encoding="utf-8")
            self.assertNotIn(approved_questions()[0].question, serialized)
            self.assertNotIn("child text 1", serialized)
            self.assertNotIn('"hits"', serialized)
            self.assertNotIn("PRIVATE_REVIEW_COMMENT", serialized)
            self.assertFalse(manifest["privacy"]["contains_question_text"])
            self.assertFalse(manifest["privacy"]["contains_hit_text"])
            self.assertFalse(
                manifest["privacy"]["contains_manual_comments"]
            )

    def test_freeze_pilot_runs_rejects_incomplete_or_changed_matrix(self) -> None:
        from experiments.rag_v1_5.runner import freeze_pilot_runs

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matrix_dir, pilot_manifest_path = self.completed_matrix(root)
            output_path = root / "pilot-runs.json"
            metrics_path = (
                matrix_dir / "c0-hybrid-rerank" / "metrics.json"
            )
            original_metrics = metrics_path.read_bytes()
            metrics_path.unlink()
            with self.assertRaises(FileNotFoundError):
                freeze_pilot_runs(
                    run_dir=matrix_dir,
                    pilot_manifest_path=pilot_manifest_path,
                    output_path=output_path,
                )
            metrics_path.write_bytes(original_metrics)

            summary_path = matrix_dir / "matrix-summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["configs"].append(summary["configs"][0])
            summary_path.write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "重复 config"):
                freeze_pilot_runs(
                    run_dir=matrix_dir,
                    pilot_manifest_path=pilot_manifest_path,
                    output_path=output_path,
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matrix_dir, pilot_manifest_path = self.completed_matrix(root)
            matrix_config_path = matrix_dir / "matrix-config.json"
            matrix_config = json.loads(
                matrix_config_path.read_text(encoding="utf-8")
            )
            matrix_config["matrix"][0]["mode"] = "bm25"
            matrix_config_path.write_text(
                json.dumps(matrix_config),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "固定矩阵"):
                freeze_pilot_runs(
                    run_dir=matrix_dir,
                    pilot_manifest_path=pilot_manifest_path,
                    output_path=root / "pilot-runs.json",
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matrix_dir, pilot_manifest_path = self.completed_matrix(root)
            run_config_path = (
                matrix_dir / "c0-hybrid-rerank" / "run-config.json"
            )
            run_config = json.loads(
                run_config_path.read_text(encoding="utf-8")
            )
            run_config["input_hashes"]["dataset_sha256"] = "F" * 64
            run_config_path.write_text(
                json.dumps(run_config),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input hash"):
                freeze_pilot_runs(
                    run_dir=matrix_dir,
                    pilot_manifest_path=pilot_manifest_path,
                    output_path=root / "pilot-runs.json",
                )
            matrix_config = json.loads(
                (matrix_dir / "matrix-config.json").read_text(
                    encoding="utf-8"
                )
            )
            run_config["input_hashes"] = matrix_config["input_hashes"]
            run_config["config"]["bm25"]["top_k"] = 999
            run_config_path.write_text(
                json.dumps(run_config),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "矩阵配置"):
                freeze_pilot_runs(
                    run_dir=matrix_dir,
                    pilot_manifest_path=pilot_manifest_path,
                    output_path=root / "pilot-runs.json",
                )

    def test_freeze_pilot_runs_rejects_result_changes_after_freeze(self) -> None:
        from experiments.rag_v1_5.runner import freeze_pilot_runs

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matrix_dir, pilot_manifest_path = self.completed_matrix(root)
            output_path = root / "pilot-runs.json"
            first = freeze_pilot_runs(
                run_dir=matrix_dir,
                pilot_manifest_path=pilot_manifest_path,
                output_path=output_path,
            )
            second = freeze_pilot_runs(
                run_dir=matrix_dir,
                pilot_manifest_path=pilot_manifest_path,
                output_path=output_path,
            )
            self.assertEqual(first, second)

            metrics_path = (
                matrix_dir / "c0-hybrid-rerank" / "metrics.json"
            )
            metrics_path.write_text(
                metrics_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "已冻结"):
                freeze_pilot_runs(
                    run_dir=matrix_dir,
                    pilot_manifest_path=pilot_manifest_path,
                    output_path=output_path,
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

    def test_freeze_pilot_runs_cli_contract_matches_plan(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        run_dir = Path("data/rag_v1_5/runs/pilot/example")
        args = build_parser().parse_args(
            ["freeze-pilot-runs", "--run-dir", str(run_dir)]
        )

        self.assertEqual(args.command, "freeze-pilot-runs")
        self.assertEqual(args.run_dir, run_dir)
        self.assertEqual(
            args.pilot_manifest,
            Path(
                "experiments/rag_v1_5/manifests/"
                "pilot-40-v1.5.0.json"
            ),
        )
        self.assertEqual(
            args.output,
            Path(
                "experiments/rag_v1_5/manifests/"
                "pilot-runs-v1.5.0.json"
            ),
        )


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
