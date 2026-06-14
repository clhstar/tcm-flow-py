import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.rag_v1_5.schema import PilotQuestion, RetrievalHit


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def make_question(
    question_id: str,
    *,
    answerable: bool = True,
    evidence_id: str = "shl-chapter-01-001",
    clause_id: str = "shl-chapter-01-001",
    support_span: str = "脉浮、头项强痛而恶寒",
) -> PilotQuestion:
    return PilotQuestion(
        question_id=question_id,
        question=f"测试问题 {question_id}",
        question_type=(
            "single_clause_fact" if answerable else "unanswerable"
        ),
        book_scope="shang_han_lun" if answerable else "both",
        answerable=answerable,
        reference_answer=(
            support_span if answerable else "当前两部语料中无答案。"
        ),
        gold_evidence_ids=[evidence_id] if answerable else [],
        gold_clause_ids=[clause_id] if answerable else [],
        graded_relevance={clause_id: 2} if answerable else {},
        support_spans=[support_span] if answerable else [],
        review_status="approved",
    )


def write_questions(path: Path, questions: list[PilotQuestion]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                question.model_dump(mode="json"),
                ensure_ascii=False,
            )
            + "\n"
            for question in questions
        ),
        encoding="utf-8",
    )


class DatasetValidationTests(unittest.TestCase):
    def test_validates_smoke_count_and_evidence_contract(self) -> None:
        from experiments.rag_v1_5.dataset import validate_dataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "smoke-10.jsonl"
            questions = [
                make_question(f"smoke-{index:02d}")
                for index in range(1, 10)
            ]
            questions.append(
                make_question("smoke-10", answerable=False)
            )
            write_questions(dataset_path, questions)

            summary = validate_dataset(
                dataset_path=dataset_path,
                evidence_path=FIXTURES_DIR / "evidence_sample.jsonl",
                profile="smoke",
            )

        self.assertEqual(summary["question_count"], 10)
        self.assertEqual(summary["answerable_count"], 9)
        self.assertEqual(summary["unanswerable_count"], 1)
        self.assertEqual(summary["approved_count"], 10)
        self.assertEqual(len(summary["dataset_sha256"]), 64)

    def test_rejects_invalid_gold_support_and_leaked_ids(self) -> None:
        from experiments.rag_v1_5.dataset import validate_dataset

        base_questions = [
            make_question(f"smoke-{index:02d}")
            for index in range(1, 10)
        ] + [make_question("smoke-10", answerable=False)]
        invalid_cases = {
            "missing evidence": base_questions[0].model_copy(
                update={"gold_evidence_ids": ["missing-evidence"]}
            ),
            "support is not evidence substring": base_questions[0].model_copy(
                update={"support_spans": ["不存在的支持片段"]}
            ),
            "question leaks gold id": base_questions[0].model_copy(
                update={
                    "question": (
                        "请定位 shl-chapter-01-001 的原文内容"
                    )
                }
            ),
        }

        for label, invalid_question in invalid_cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    dataset_path = (
                        Path(temporary_directory) / "smoke-10.jsonl"
                    )
                    questions = list(base_questions)
                    questions[0] = invalid_question
                    write_questions(dataset_path, questions)
                    with self.assertRaises(ValueError):
                        validate_dataset(
                            dataset_path=dataset_path,
                            evidence_path=(
                                FIXTURES_DIR / "evidence_sample.jsonl"
                            ),
                            profile="smoke",
                        )


class SmokeRunTests(unittest.TestCase):
    def test_writes_results_metrics_and_manual_review_csv(self) -> None:
        from experiments.rag_v1_5.dataset import (
            SMOKE_REVIEW_FIELDS,
            run_smoke_dataset,
        )

        questions = [
            make_question(f"smoke-{index:02d}")
            for index in range(1, 10)
        ] + [make_question("smoke-10", answerable=False)]

        def fake_retriever(question: PilotQuestion) -> list[RetrievalHit]:
            clause_id = (
                question.gold_clause_ids[0]
                if question.answerable
                else "unrelated-clause"
            )
            return [
                RetrievalHit(
                    chunk_id=f"chunk-{question.question_id}",
                    strategy="c4",
                    rank=1,
                    text="child text",
                    context_text="full parent context",
                    source_evidence_ids=[
                        (
                            question.gold_evidence_ids[0]
                            if question.answerable
                            else "unrelated-evidence"
                        )
                    ],
                    clause_ids=[clause_id],
                    retrieval_parent_id=clause_id,
                    reranker_score=0.9,
                )
            ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "run"
            review_path = root / "smoke-review.csv"

            summary = run_smoke_dataset(
                questions=questions,
                strategy="c4",
                mode="hybrid_rerank",
                output_dir=output_dir,
                review_csv_path=review_path,
                retriever=fake_retriever,
                provenance={"dataset_sha256": "A" * 64},
            )

            with review_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file_handle:
                rows = list(csv.DictReader(file_handle))
            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            records = [
                json.loads(line)
                for line in (
                    output_dir / "per-question.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["answerable_hit_at_5"], 1.0)
        self.assertEqual(summary["answerable_parent_recovery_rate"], 1.0)
        self.assertEqual(metrics["hit_at_5"], 1.0)
        self.assertEqual(len(records), 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(list(rows[0]), list(SMOKE_REVIEW_FIELDS))
        self.assertEqual(rows[0]["hit_at_5"], "true")
        self.assertEqual(rows[0]["parent_recovery_ok"], "true")


class DatasetCliTests(unittest.TestCase):
    def test_smoke_runtime_requires_ready_matching_quality_gate(
        self,
    ) -> None:
        from experiments.rag_v1_5.cli import (
            validate_smoke_runtime_inputs,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            quality_gate_path = root / "quality-gate.json"
            index_manifest_path = root / "indexes.json"
            indexes_dir = root / "indexes"
            strategy_dir = indexes_dir / "c4"
            strategy_dir.mkdir(parents=True)
            quality_gate_path.write_text(
                '{"status":"ready"}\n',
                encoding="utf-8",
            )
            quality_gate_sha256 = hashlib.sha256(
                quality_gate_path.read_bytes()
            ).hexdigest().upper()
            strategy_manifest_path = strategy_dir / "manifest.json"
            strategy_manifest_path.write_text(
                json.dumps(
                    {"quality_gate_sha256": quality_gate_sha256}
                ),
                encoding="utf-8",
            )
            strategy_manifest_sha256 = hashlib.sha256(
                strategy_manifest_path.read_bytes()
            ).hexdigest().upper()
            index_manifest_path.write_text(
                json.dumps(
                    {
                        "quality_gate_sha256": quality_gate_sha256,
                        "strategies": {
                            "c4": {
                                "manifest_sha256": (
                                    strategy_manifest_sha256
                                )
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = validate_smoke_runtime_inputs(
                quality_gate_path=quality_gate_path,
                index_manifest_path=index_manifest_path,
                indexes_dir=indexes_dir,
                strategy="c4",
            )
            self.assertEqual(
                result["quality_gate_sha256"],
                quality_gate_sha256,
            )

            strategy_manifest_path.write_text(
                '{"quality_gate_sha256":"stale"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_smoke_runtime_inputs(
                    quality_gate_path=quality_gate_path,
                    index_manifest_path=index_manifest_path,
                    indexes_dir=indexes_dir,
                    strategy="c4",
                )

            quality_gate_path.write_text(
                '{"status":"blocked"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_smoke_runtime_inputs(
                    quality_gate_path=quality_gate_path,
                    index_manifest_path=index_manifest_path,
                    indexes_dir=indexes_dir,
                    strategy="c4",
                )

    def test_validate_dataset_cli_accepts_smoke_profile(self) -> None:
        from experiments.rag_v1_5.cli import main

        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "smoke-10.jsonl"
            questions = [
                make_question(f"smoke-{index:02d}")
                for index in range(1, 10)
            ] + [make_question("smoke-10", answerable=False)]
            write_questions(dataset_path, questions)

            exit_code = main(
                [
                    "validate-dataset",
                    "--dataset",
                    str(dataset_path),
                    "--evidence",
                    str(FIXTURES_DIR / "evidence_sample.jsonl"),
                    "--profile",
                    "smoke",
                ]
            )

        self.assertEqual(exit_code, 0)

    def test_run_smoke_cli_contract_matches_plan(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        args = build_parser().parse_args(
            [
                "run-smoke",
                "--dataset",
                "data/rag_v1_5/evaluation/smoke-10.jsonl",
                "--strategy",
                "c4",
                "--mode",
                "hybrid_rerank",
                "--output-dir",
                "data/rag_v1_5/runs/smoke",
            ]
        )

        self.assertEqual(args.command, "run-smoke")
        self.assertEqual(args.strategy, "c4")
        self.assertEqual(args.mode, "hybrid_rerank")
        self.assertEqual(
            args.review_csv,
            Path("data/rag_v1_5/evaluation/smoke-review.csv"),
        )
        self.assertEqual(
            args.quality_gate,
            Path(
                "experiments/rag_v1_5/manifests/"
                "quality-gate-v1.5.0.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
