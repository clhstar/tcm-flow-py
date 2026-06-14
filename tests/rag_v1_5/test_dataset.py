import csv
import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
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


def write_jsonl_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def make_pilot_artifacts() -> tuple[list[dict], list[dict], list[dict]]:
    books = ("shang_han_lun", "jin_gui_yao_lue")
    question_types = (
        "single_clause_fact",
        "formula_composition_or_use",
        "source_location",
        "multi_evidence",
        "unanswerable",
    )
    evidence_records = []
    evidence_ids_by_book = {}
    for book_index, book in enumerate(books, start=1):
        prefix = "shl" if book == "shang_han_lun" else "jgy"
        evidence_ids_by_book[book] = []
        for clause_index in range(1, 3):
            clause_id = f"{prefix}-pilot-clause-{clause_index:03d}"
            evidence_ids_by_book[book].append(clause_id)
            evidence_records.append(
                {
                    "evidence_id": clause_id,
                    "book_id": book,
                    "book_title": f"测试书籍 {book_index}",
                    "volume": "",
                    "chapter_id": f"{prefix}-pilot-chapter",
                    "chapter_title": "测试篇章",
                    "clause_id": clause_id,
                    "clause_number": clause_index,
                    "content_type": "clause",
                    "parent_id": clause_id,
                    "original_text": f"{book} 测试原文 {clause_index}",
                    "normalized_text": (
                        f"{book} 测试原文 {clause_index}"
                    ),
                    "notes": [],
                    "source_file": f"{prefix}.txt",
                    "source_hash": f"{book_index}" * 64,
                    "corpus_version": "v1.5.0",
                }
            )

    questions = []
    groups = []
    question_number = 0
    for book in books:
        book_evidence_ids = evidence_ids_by_book[book]
        for question_type in question_types:
            for cell_index in range(1, 5):
                question_number += 1
                group_id = f"pilot-group-{question_number:02d}"
                question_id = f"pilot-{question_number:02d}"
                answerable = question_type != "unanswerable"
                if question_type == "multi_evidence":
                    gold_evidence_ids = list(book_evidence_ids)
                    gold_clause_ids = list(book_evidence_ids)
                elif answerable:
                    evidence_id = book_evidence_ids[
                        (cell_index - 1) % len(book_evidence_ids)
                    ]
                    gold_evidence_ids = [evidence_id]
                    gold_clause_ids = [evidence_id]
                else:
                    gold_evidence_ids = []
                    gold_clause_ids = []
                questions.append(
                    {
                        "question_id": question_id,
                        "question": (
                            f"测试 {book} {question_type} "
                            f"{cell_index}？"
                        ),
                        "question_type": question_type,
                        "book_scope": book,
                        "answerable": answerable,
                        "reference_answer": (
                            (
                                f"{book} 测试原文 "
                                f"{gold_evidence_ids[0][-3:].lstrip('0')}"
                            )
                            if answerable
                            else "当前两部语料中无答案。"
                        ),
                        "gold_evidence_ids": gold_evidence_ids,
                        "gold_clause_ids": gold_clause_ids,
                        "graded_relevance": {
                            clause_id: 2
                            for clause_id in gold_clause_ids
                        },
                        "support_spans": (
                            [
                                (
                                    f"{book} 测试原文 "
                                    f"{gold_evidence_ids[0][-3:].lstrip('0')}"
                                )
                            ]
                            if answerable
                            else []
                        ),
                        "review_status": "draft",
                        "split": "pilot",
                        "evidence_group_id": group_id,
                        "question_version": 1,
                    }
                )
                groups.append(
                    {
                        "group_id": group_id,
                        "split": "pilot",
                        "book_scope": book,
                        "question_type": question_type,
                        "anchor_evidence_ids": gold_evidence_ids,
                        "anchor_clause_ids": gold_clause_ids,
                        "selection_seed": 20260614,
                        "selection_reason": "测试选择原因",
                        "absence_queries": (
                            ["测试缺失查询一", "测试缺失查询二"]
                            if not answerable
                            else []
                        ),
                    }
                )
    return questions, groups, evidence_records


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


class PilotDatasetValidationTests(unittest.TestCase):
    def validate_pilot(
        self,
        root: Path,
        questions: list[dict],
        groups: list[dict],
        evidence_records: list[dict],
        *,
        filename: str = "pilot-40.jsonl",
        profile: str = "pilot",
    ) -> dict:
        from experiments.rag_v1_5.dataset import validate_dataset

        dataset_path = root / filename
        groups_path = root / "pilot-evidence-groups.jsonl"
        evidence_path = root / "evidence.jsonl"
        write_jsonl_records(dataset_path, questions)
        write_jsonl_records(groups_path, groups)
        write_jsonl_records(evidence_path, evidence_records)
        return validate_dataset(
            dataset_path=dataset_path,
            evidence_path=evidence_path,
            evidence_groups_path=groups_path,
            profile=profile,
        )

    def test_valid_pilot_profile_returns_fixed_counts(self) -> None:
        questions, groups, evidence_records = make_pilot_artifacts()
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = self.validate_pilot(
                Path(temporary_directory),
                questions,
                groups,
                evidence_records,
            )

        self.assertEqual(summary["profile"], "pilot")
        self.assertEqual(summary["question_count"], 40)
        self.assertEqual(summary["answerable_count"], 32)
        self.assertEqual(summary["unanswerable_count"], 8)
        self.assertEqual(
            summary["quota_by_book_and_type"]["shang_han_lun"],
            {
                "single_clause_fact": 4,
                "formula_composition_or_use": 4,
                "source_location": 4,
                "multi_evidence": 4,
                "unanswerable": 4,
            },
        )
        self.assertEqual(
            sum(
                summary["quota_by_book_and_type"][
                    "shang_han_lun"
                ].values()
            ),
            20,
        )
        self.assertEqual(
            sum(
                summary["quota_by_book_and_type"][
                    "jin_gui_yao_lue"
                ].values()
            ),
            20,
        )
        self.assertEqual(summary["duplicate_question_count"], 0)
        self.assertEqual(summary["multi_evidence_count"], 8)
        self.assertEqual(len(summary["evidence_group_sha256"]), 64)

    def test_rejects_incomplete_book_type_quota(self) -> None:
        questions, groups, evidence_records = make_pilot_artifacts()
        questions.pop()
        groups.pop()
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "配额"):
                self.validate_pilot(
                    Path(temporary_directory),
                    questions,
                    groups,
                    evidence_records,
                )

    def test_rejects_normalized_duplicate_question_text(self) -> None:
        questions, groups, evidence_records = make_pilot_artifacts()
        questions[1]["question"] = (
            "  测试   shang_han_lun "
            "single_clause_fact 1?  "
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "重复"):
                self.validate_pilot(
                    Path(temporary_directory),
                    questions,
                    groups,
                    evidence_records,
                )

    def test_rejects_both_book_scope(self) -> None:
        questions, groups, evidence_records = make_pilot_artifacts()
        questions[0]["book_scope"] = "both"
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "both"):
                self.validate_pilot(
                    Path(temporary_directory),
                    questions,
                    groups,
                    evidence_records,
                )

    def test_rejects_invalid_multi_evidence_gold(self) -> None:
        questions, groups, evidence_records = make_pilot_artifacts()
        multi_index = next(
            index
            for index, question in enumerate(questions)
            if question["question_type"] == "multi_evidence"
        )
        invalid_cases = {}

        too_short_questions = deepcopy(questions)
        too_short_groups = deepcopy(groups)
        too_short_questions[multi_index]["gold_evidence_ids"] = [
            too_short_questions[multi_index]["gold_evidence_ids"][0]
        ]
        too_short_questions[multi_index]["gold_clause_ids"] = [
            too_short_questions[multi_index]["gold_clause_ids"][0]
        ]
        too_short_questions[multi_index]["graded_relevance"] = {
            too_short_questions[multi_index]["gold_clause_ids"][0]: 2
        }
        too_short_groups[multi_index]["anchor_evidence_ids"] = list(
            too_short_questions[multi_index]["gold_evidence_ids"]
        )
        too_short_groups[multi_index]["anchor_clause_ids"] = list(
            too_short_questions[multi_index]["gold_clause_ids"]
        )
        invalid_cases["少于 2 个 gold clause"] = (
            too_short_questions,
            too_short_groups,
        )

        cross_book_questions = deepcopy(questions)
        cross_book_groups = deepcopy(groups)
        cross_book_id = next(
            evidence["evidence_id"]
            for evidence in evidence_records
            if evidence["book_id"] == "jin_gui_yao_lue"
        )
        cross_book_questions[multi_index]["gold_evidence_ids"][1] = (
            cross_book_id
        )
        cross_book_questions[multi_index]["gold_clause_ids"][1] = (
            cross_book_id
        )
        cross_book_questions[multi_index]["graded_relevance"] = {
            clause_id: 2
            for clause_id in cross_book_questions[multi_index][
                "gold_clause_ids"
            ]
        }
        cross_book_groups[multi_index]["anchor_evidence_ids"] = list(
            cross_book_questions[multi_index]["gold_evidence_ids"]
        )
        cross_book_groups[multi_index]["anchor_clause_ids"] = list(
            cross_book_questions[multi_index]["gold_clause_ids"]
        )
        invalid_cases["跨书"] = (
            cross_book_questions,
            cross_book_groups,
        )

        for label, (case_questions, case_groups) in invalid_cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaises(ValueError):
                        self.validate_pilot(
                            Path(temporary_directory),
                            case_questions,
                            case_groups,
                            evidence_records,
                        )

    def test_rejects_relevance_key_outside_gold_ids(self) -> None:
        questions, groups, evidence_records = make_pilot_artifacts()
        invalid_keys = (
            "unrelated-evidence",
            groups[0]["group_id"],
        )
        for invalid_key in invalid_keys:
            with self.subTest(invalid_key=invalid_key):
                case_questions = deepcopy(questions)
                case_questions[0]["graded_relevance"] = {
                    invalid_key: 2
                }
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaisesRegex(
                        ValueError,
                        "graded_relevance",
                    ):
                        self.validate_pilot(
                            Path(temporary_directory),
                            case_questions,
                            groups,
                            evidence_records,
                        )

    def test_rejects_unanswerable_gold_support_or_relevance(self) -> None:
        questions, groups, evidence_records = make_pilot_artifacts()
        unanswerable_index = next(
            index
            for index, question in enumerate(questions)
            if not question["answerable"]
        )
        invalid_updates = (
            {"gold_evidence_ids": ["shl-pilot-clause-001"]},
            {"gold_clause_ids": ["shl-pilot-clause-001"]},
            {"support_spans": ["测试支持"]},
            {"graded_relevance": {"shl-pilot-clause-001": 2}},
        )
        for update in invalid_updates:
            with self.subTest(update=update):
                case_questions = deepcopy(questions)
                case_questions[unanswerable_index].update(update)
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaises(ValueError):
                        self.validate_pilot(
                            Path(temporary_directory),
                            case_questions,
                            groups,
                            evidence_records,
                        )

    def test_requires_explicit_pilot_metadata(self) -> None:
        questions, groups, evidence_records = make_pilot_artifacts()
        for missing_field in (
            "evidence_group_id",
            "split",
            "question_version",
        ):
            with self.subTest(missing_field=missing_field):
                case_questions = deepcopy(questions)
                case_questions[0].pop(missing_field)
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaises(ValueError):
                        self.validate_pilot(
                            Path(temporary_directory),
                            case_questions,
                            groups,
                            evidence_records,
                        )

    def test_auto_profile_recognizes_pilot_40_prefix(self) -> None:
        questions, groups, evidence_records = make_pilot_artifacts()
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = self.validate_pilot(
                Path(temporary_directory),
                questions,
                groups,
                evidence_records,
                filename="pilot-40-v1.5.0.jsonl",
                profile="auto",
            )

        self.assertEqual(summary["profile"], "pilot")

    def test_minimal_pilot_fixtures_parse_with_schema(self) -> None:
        from experiments.rag_v1_5.dataset import _read_jsonl
        from experiments.rag_v1_5.schema import PilotEvidenceGroup

        questions = _read_jsonl(
            FIXTURES_DIR / "pilot_questions_sample.jsonl",
            PilotQuestion,
        )
        groups = _read_jsonl(
            FIXTURES_DIR / "pilot_evidence_groups_sample.jsonl",
            PilotEvidenceGroup,
        )

        self.assertEqual(len(questions), 2)
        self.assertEqual(len(groups), 2)


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

    def test_validate_dataset_cli_accepts_pilot_contract(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        args = build_parser().parse_args(
            [
                "validate-dataset",
                "--dataset",
                "data/rag_v1_5/evaluation/pilot-40.jsonl",
                "--profile",
                "pilot",
                "--evidence-groups",
                (
                    "data/rag_v1_5/evaluation/"
                    "pilot-evidence-groups.jsonl"
                ),
            ]
        )

        self.assertEqual(args.profile, "pilot")
        self.assertEqual(
            args.evidence_groups,
            Path(
                "data/rag_v1_5/evaluation/"
                "pilot-evidence-groups.jsonl"
            ),
        )

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
