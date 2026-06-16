import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
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


def make_sampling_evidence_records() -> list[dict]:
    records = []
    for book_index, book in enumerate(
        ("shang_han_lun", "jin_gui_yao_lue"),
        start=1,
    ):
        prefix = "shl" if book == "shang_han_lun" else "jgy"
        for clause_index in range(1, 26):
            clause_id = f"{prefix}-sample-clause-{clause_index:03d}"
            base_record = {
                "book_id": book,
                "book_title": f"测试书籍 {book_index}",
                "volume": "",
                "chapter_id": f"{prefix}-sample-chapter",
                "chapter_title": "测试定位篇章",
                "clause_id": clause_id,
                "clause_number": clause_index,
                "parent_id": clause_id,
                "notes": [],
                "source_file": f"{prefix}.txt",
                "source_hash": f"{book_index}" * 64,
                "corpus_version": "v1.5.0",
            }
            clause_text = (
                f"{book} 第 {clause_index} 条测试原文，"
                "用于稳定候选选择。"
            )
            records.append(
                {
                    **base_record,
                    "evidence_id": clause_id,
                    "content_type": "clause",
                    "original_text": clause_text,
                    "normalized_text": clause_text,
                }
            )
            if 6 <= clause_index <= 10:
                formula_id = f"{clause_id}-formula-01"
                records.append(
                    {
                        **base_record,
                        "evidence_id": formula_id,
                        "content_type": "formula",
                        "parent_id": clause_id,
                        "original_text": f"测试方剂 {clause_index}",
                        "normalized_text": f"测试方剂 {clause_index}",
                    }
                )
                records.append(
                    {
                        **base_record,
                        "evidence_id": f"{formula_id}-ingredients",
                        "content_type": "ingredients",
                        "parent_id": formula_id,
                        "original_text": "测试药物甲、测试药物乙",
                        "normalized_text": "测试药物甲、测试药物乙",
                    }
                )
            if 11 <= clause_index <= 15:
                records.append(
                    {
                        **base_record,
                        "evidence_id": f"{clause_id}-note-01",
                        "content_type": "note",
                        "parent_id": clause_id,
                        "original_text": f"测试校注 {clause_index}",
                        "normalized_text": f"测试校注 {clause_index}",
                    }
                )
    return records


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


class PilotEvidenceSamplingTests(unittest.TestCase):
    def run_sampling(
        self,
        root: Path,
        evidence_records: list[dict],
        *,
        suffix: str,
    ) -> tuple[dict, Path, Path, Path]:
        from experiments.rag_v1_5.dataset import (
            sample_pilot_evidence_groups,
        )

        evidence_path = root / f"evidence-{suffix}.jsonl"
        smoke_path = root / "smoke-10.jsonl"
        output_path = root / f"groups-{suffix}.jsonl"
        exclusions_path = root / f"exclusions-{suffix}.json"
        report_path = root / f"report-{suffix}.json"
        write_jsonl_records(evidence_path, evidence_records)
        smoke_evidence = next(
            record
            for record in evidence_records
            if record["evidence_id"] == "shl-sample-clause-001"
        )
        write_questions(
            smoke_path,
            [
                make_question(
                    "smoke-sampling-01",
                    evidence_id=smoke_evidence["evidence_id"],
                    clause_id=smoke_evidence["clause_id"],
                    support_span=smoke_evidence["normalized_text"],
                )
            ],
        )
        summary = sample_pilot_evidence_groups(
            evidence_path=evidence_path,
            smoke_dataset_path=smoke_path,
            output_path=output_path,
            exclusions_path=exclusions_path,
            candidate_report_path=report_path,
            seed=20260614,
        )
        return summary, output_path, exclusions_path, report_path

    def test_sampling_is_deterministic_and_input_order_independent(
        self,
    ) -> None:
        evidence_records = make_sampling_evidence_records()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = self.run_sampling(
                root,
                evidence_records,
                suffix="first",
            )
            second = self.run_sampling(
                root,
                list(reversed(evidence_records)),
                suffix="second",
            )
            first_summary, first_output, first_exclusions, first_report = (
                first
            )
            _, second_output, second_exclusions, second_report = second

            self.assertEqual(
                first_output.read_bytes(),
                second_output.read_bytes(),
            )
            self.assertEqual(
                first_exclusions.read_bytes(),
                second_exclusions.read_bytes(),
            )
            self.assertEqual(
                first_report.read_bytes(),
                second_report.read_bytes(),
            )

        self.assertEqual(first_summary["group_count"], 40)
        self.assertEqual(first_summary["answerable_group_count"], 32)
        self.assertEqual(first_summary["unanswerable_group_count"], 8)

    def test_sampling_excludes_smoke_and_uses_unique_anchors(self) -> None:
        evidence_records = make_sampling_evidence_records()
        evidence_by_id = {
            record["evidence_id"]: record for record in evidence_records
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, output_path, exclusions_path, _ = self.run_sampling(
                Path(temporary_directory),
                evidence_records,
                suffix="unique",
            )
            groups = [
                json.loads(line)
                for line in output_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            exclusions = json.loads(
                exclusions_path.read_text(encoding="utf-8")
            )

        self.assertEqual(len({group["group_id"] for group in groups}), 40)
        answerable_groups = [
            group
            for group in groups
            if group["question_type"] != "unanswerable"
        ]
        anchor_evidence_ids = [
            evidence_id
            for group in answerable_groups
            for evidence_id in group["anchor_evidence_ids"]
        ]
        anchor_clause_ids = [
            clause_id
            for group in answerable_groups
            for clause_id in group["anchor_clause_ids"]
        ]
        self.assertEqual(
            len(anchor_evidence_ids),
            len(set(anchor_evidence_ids)),
        )
        self.assertEqual(
            len(anchor_clause_ids),
            len(set(anchor_clause_ids)),
        )
        self.assertNotIn(
            "shl-sample-clause-001",
            anchor_evidence_ids,
        )
        self.assertNotIn(
            "shl-sample-clause-001",
            anchor_clause_ids,
        )
        self.assertIn(
            "shl-sample-clause-001",
            exclusions["smoke_gold_evidence_ids"],
        )
        self.assertEqual(
            set(exclusions["pilot_group_ids"]),
            {group["group_id"] for group in groups},
        )
        self.assertEqual(
            set(exclusions["future_formal_excluded_clause_ids"]),
            set(anchor_clause_ids)
            | set(exclusions["smoke_gold_clause_ids"]),
        )

        for group in answerable_groups:
            self.assertTrue(group["anchor_evidence_ids"])
            self.assertTrue(group["anchor_clause_ids"])
            for evidence_id in group["anchor_evidence_ids"]:
                self.assertEqual(
                    evidence_by_id[evidence_id]["book_id"],
                    group["book_scope"],
                )

    def test_sampling_enforces_type_specific_candidate_contracts(
        self,
    ) -> None:
        evidence_records = make_sampling_evidence_records()
        evidence_by_id = {
            record["evidence_id"]: record for record in evidence_records
        }
        clause_book = {
            record["clause_id"]: record["book_id"]
            for record in evidence_records
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            _, output_path, _, report_path = self.run_sampling(
                Path(temporary_directory),
                evidence_records,
                suffix="types",
            )
            groups = [
                json.loads(line)
                for line in output_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )

        formula_types = {"formula", "ingredients", "preparation"}
        for group in groups:
            if group["question_type"] == "formula_composition_or_use":
                self.assertTrue(
                    formula_types
                    & {
                        evidence_by_id[evidence_id]["content_type"]
                        for evidence_id in group[
                            "anchor_evidence_ids"
                        ]
                    }
                )
            elif group["question_type"] == "source_location":
                anchor_records = [
                    evidence_by_id[evidence_id]
                    for evidence_id in group["anchor_evidence_ids"]
                ]
                self.assertTrue(anchor_records)
                self.assertTrue(
                    all(
                        record["content_type"] == "note"
                        for record in anchor_records
                    ),
                    "存在足够 note 候选时必须优先选择 note",
                )
            elif group["question_type"] == "multi_evidence":
                self.assertEqual(
                    len(set(group["anchor_clause_ids"])),
                    2,
                )
                self.assertGreaterEqual(
                    len(set(group["anchor_evidence_ids"])),
                    2,
                )
                self.assertEqual(
                    {
                        clause_book[clause_id]
                        for clause_id in group["anchor_clause_ids"]
                    },
                    {group["book_scope"]},
                )
            elif group["question_type"] == "unanswerable":
                self.assertEqual(group["anchor_evidence_ids"], [])
                self.assertEqual(group["anchor_clause_ids"], [])
                self.assertEqual(group["absence_queries"], [])
                self.assertIn("人工", group["selection_reason"])

        self.assertEqual(report["selected_group_count"], 40)
        self.assertEqual(len(report["strata"]), 10)
        self.assertNotIn("normalized_text", json.dumps(report))
        self.assertNotIn("original_text", json.dumps(report))

    def test_sampling_reports_insufficient_stratum(self) -> None:
        evidence_records = [
            record
            for record in make_sampling_evidence_records()
            if not (
                record["book_id"] == "jin_gui_yao_lue"
                and record["content_type"]
                in {"formula", "ingredients", "preparation"}
            )
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(
                ValueError,
                "jin_gui_yao_lue/formula_composition_or_use",
            ):
                self.run_sampling(
                    Path(temporary_directory),
                    evidence_records,
                    suffix="insufficient",
                )


class PilotReviewWorkflowTests(unittest.TestCase):
    def prepare_review_files(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path, list[dict]]:
        from experiments.rag_v1_5.dataset import prepare_pilot_review

        questions, groups, _ = make_pilot_artifacts()
        draft_path = root / "pilot-40-draft.jsonl"
        groups_path = root / "pilot-evidence-groups.jsonl"
        review_path = root / "pilot-review.csv"
        write_jsonl_records(draft_path, questions)
        write_jsonl_records(groups_path, groups)
        prepare_pilot_review(
            draft_dataset_path=draft_path,
            evidence_groups_path=groups_path,
            review_csv_path=review_path,
            second_review_seed=20260614,
        )
        return draft_path, groups_path, review_path, questions

    def read_review_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            return list(csv.DictReader(file_handle))

    def write_review_rows(
        self,
        path: Path,
        rows: list[dict[str, str]],
        *,
        encoding: str = "utf-8-sig",
    ) -> None:
        with path.open(
            "w",
            encoding=encoding,
            newline="",
        ) as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def approve_rows(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            row["first_status"] = "pass"
            row["first_decision"] = "correct"
            row["first_comment"] = "第一轮通过"
            row["first_reviewer"] = "测试审核者"
            row["first_reviewed_at"] = "2026-06-14"
            if row["second_review_required"] == "true":
                row["second_status"] = "pass"
                row["second_decision"] = "correct"
                row["second_comment"] = "第二轮通过"
                row["second_reviewer"] = "测试复核者"
                row["second_reviewed_at"] = "2026-06-14"

    def test_prepare_review_writes_bom_and_stable_stratified_second_round(
        self,
    ) -> None:
        from experiments.rag_v1_5.dataset import prepare_pilot_review

        questions, groups, _ = make_pilot_artifacts()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            draft_path = root / "pilot-40-draft.jsonl"
            groups_path = root / "pilot-evidence-groups.jsonl"
            first_path = root / "pilot-review-first.csv"
            second_path = root / "pilot-review-second.csv"
            write_jsonl_records(draft_path, questions)
            write_jsonl_records(groups_path, groups)
            first_summary = prepare_pilot_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                review_csv_path=first_path,
                second_review_seed=20260614,
            )
            second_summary = prepare_pilot_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                review_csv_path=second_path,
                second_review_seed=20260614,
            )
            rows = self.read_review_rows(first_path)

            self.assertTrue(first_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

        required_rows = [
            row
            for row in rows
            if row["second_review_required"] == "true"
        ]
        strata = Counter(
            (row["book_scope"], row["question_type"])
            for row in required_rows
        )
        self.assertEqual(len(required_rows), 10)
        self.assertEqual(set(strata.values()), {1})
        self.assertEqual(first_summary["second_review_required_count"], 10)
        self.assertEqual(
            first_summary["second_review_question_ids"],
            second_summary["second_review_question_ids"],
        )

    def test_prepare_review_inherits_only_unchanged_content(self) -> None:
        from experiments.rag_v1_5.dataset import prepare_pilot_review

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            draft_path, groups_path, review_path, questions = (
                self.prepare_review_files(root)
            )
            rows = self.read_review_rows(review_path)
            rows[0]["first_status"] = "pass"
            rows[0]["first_decision"] = "correct"
            rows[0]["first_comment"] = "保留审核"
            rows[0]["first_reviewer"] = "测试审核者"
            rows[0]["first_reviewed_at"] = "2026-06-14"
            original_hash = rows[0]["content_sha256"]
            self.write_review_rows(review_path, rows)

            prepare_pilot_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                review_csv_path=review_path,
                second_review_seed=20260614,
            )
            inherited_rows = self.read_review_rows(review_path)
            self.assertEqual(inherited_rows[0]["first_status"], "pass")
            self.assertEqual(
                inherited_rows[0]["first_comment"],
                "保留审核",
            )

            questions[0]["question"] += " 已修订"
            write_jsonl_records(draft_path, questions)
            prepare_pilot_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                review_csv_path=review_path,
                second_review_seed=20260614,
            )
            reset_rows = self.read_review_rows(review_path)

        self.assertNotEqual(reset_rows[0]["content_sha256"], original_hash)
        self.assertEqual(reset_rows[0]["first_status"], "pending")
        self.assertEqual(reset_rows[0]["first_decision"], "")
        self.assertEqual(reset_rows[0]["first_reviewer"], "")

    def test_import_review_normalizes_legacy_encodings_and_approves_all(
        self,
    ) -> None:
        from experiments.rag_v1_5.dataset import import_pilot_review

        cases = (
            ("cp936", "中文审核"),
            ("gb18030", "中文审核𠮷"),
        )
        for encoding, comment in cases:
            with self.subTest(encoding=encoding):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    draft_path, groups_path, review_path, _ = (
                        self.prepare_review_files(root)
                    )
                    output_path = root / "pilot-40.jsonl"
                    summary_path = root / "pilot-review-summary.json"
                    rows = self.read_review_rows(review_path)
                    self.approve_rows(rows)
                    rows[0]["first_comment"] = comment
                    for row in rows:
                        row["answerable"] = row["answerable"].upper()
                        row["second_review_required"] = (
                            row["second_review_required"].upper()
                        )
                    self.write_review_rows(
                        review_path,
                        rows,
                        encoding=encoding,
                    )

                    summary = import_pilot_review(
                        draft_dataset_path=draft_path,
                        evidence_groups_path=groups_path,
                        reviewed_csv_path=review_path,
                        output_dataset_path=output_path,
                        summary_path=summary_path,
                    )
                    approved = [
                        json.loads(line)
                        for line in output_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    encoding_summary = summary["encoding"]
                    backup_path = Path(
                        encoding_summary["backup_path"]
                    )

                    self.assertEqual(summary["status"], "ready")
                    self.assertEqual(summary["first_review_pass_count"], 40)
                    self.assertEqual(
                        summary["second_review_pass_count"],
                        10,
                    )
                    self.assertEqual(summary["revision_count"], 0)
                    self.assertEqual(summary["rejected_count"], 0)
                    self.assertEqual(len(approved), 40)
                    self.assertTrue(
                        all(
                            record["review_status"] == "approved"
                            for record in approved
                        )
                    )
                    self.assertEqual(
                        encoding_summary["detected_encoding"],
                        encoding,
                    )
                    self.assertTrue(encoding_summary["converted"])
                    self.assertTrue(
                        encoding_summary["unicode_equivalent"]
                    )
                    self.assertTrue(backup_path.is_file())
                    self.assertEqual(
                        encoding_summary["original_sha256"],
                        encoding_summary["backup_sha256"],
                    )
                    self.assertTrue(
                        review_path.read_bytes().startswith(
                            b"\xef\xbb\xbf"
                        )
                    )

    def test_import_review_rejects_immutable_changes(self) -> None:
        from experiments.rag_v1_5.dataset import import_pilot_review

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            draft_path, groups_path, review_path, _ = (
                self.prepare_review_files(root)
            )
            rows = self.read_review_rows(review_path)
            self.approve_rows(rows)
            rows[0]["question"] += " 被修改"
            self.write_review_rows(review_path, rows)

            with self.assertRaisesRegex(ValueError, "不允许修改"):
                import_pilot_review(
                    draft_dataset_path=draft_path,
                    evidence_groups_path=groups_path,
                    reviewed_csv_path=review_path,
                    output_dataset_path=root / "pilot-40.jsonl",
                    summary_path=root / "summary.json",
                )

    def test_import_review_blocks_incomplete_or_failed_rounds(self) -> None:
        from experiments.rag_v1_5.dataset import import_pilot_review

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            draft_path, groups_path, review_path, _ = (
                self.prepare_review_files(root)
            )
            output_path = root / "pilot-40.jsonl"
            summary_path = root / "summary.json"

            pending_summary = import_pilot_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                reviewed_csv_path=review_path,
                output_dataset_path=output_path,
                summary_path=summary_path,
            )
            self.assertEqual(pending_summary["status"], "blocked")
            self.assertEqual(
                pending_summary["first_review_pending_count"],
                40,
            )
            self.assertFalse(output_path.exists())

            rows = self.read_review_rows(review_path)
            for row in rows:
                row["first_status"] = "pass"
                row["first_decision"] = "correct"
                row["first_reviewer"] = "测试审核者"
                row["first_reviewed_at"] = "2026-06-14"
            self.write_review_rows(review_path, rows)
            second_pending_summary = import_pilot_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                reviewed_csv_path=review_path,
                output_dataset_path=output_path,
                summary_path=summary_path,
            )
            self.assertEqual(second_pending_summary["status"], "blocked")
            self.assertEqual(
                second_pending_summary["second_review_pending_count"],
                10,
            )
            self.assertFalse(output_path.exists())

            self.approve_rows(rows)
            rows[0]["first_status"] = "fail"
            rows[0]["first_decision"] = "gold_id_error"
            self.write_review_rows(review_path, rows)
            failed_summary = import_pilot_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                reviewed_csv_path=review_path,
                output_dataset_path=output_path,
                summary_path=summary_path,
            )

        self.assertEqual(failed_summary["status"], "blocked")
        self.assertEqual(failed_summary["first_review_fail_count"], 1)
        self.assertEqual(failed_summary["rejected_count"], 1)
        self.assertFalse(output_path.exists())


class PilotManifestFreezeTests(unittest.TestCase):
    def prepare_freeze_files(self, root: Path) -> dict[str, Path]:
        questions, groups, evidence_records = make_pilot_artifacts()
        for question in questions:
            question["review_status"] = "approved"
        questions[0]["question_version"] = 2

        dataset_path = root / "pilot-40.jsonl"
        groups_path = root / "pilot-evidence-groups.jsonl"
        evidence_path = root / "evidence.jsonl"
        review_summary_path = root / "pilot-review-summary.json"
        exclusions_path = root / "pilot-exclusions.json"
        chunk_manifest_path = root / "chunks-v1.5.0.json"
        quality_gate_path = root / "quality-gate-v1.5.0.json"
        index_manifest_path = root / "indexes-v1.5.0.json"
        model_manifest_path = root / "models-v1.5.0.json"
        config_path = root / "retrieval-pilot.yaml"
        smoke_manifest_path = root / "smoke-10-v1.5.0.json"
        manifest_path = root / "pilot-40-v1.5.0.json"

        write_jsonl_records(dataset_path, questions)
        write_jsonl_records(groups_path, groups)
        write_jsonl_records(evidence_path, evidence_records)
        dataset_sha256 = hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest().upper()
        groups_sha256 = hashlib.sha256(
            groups_path.read_bytes()
        ).hexdigest().upper()
        evidence_sha256 = hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest().upper()

        review_summary_path.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "question_count": 40,
                    "first_review_pass_count": 40,
                    "first_review_pending_count": 0,
                    "first_review_fail_count": 0,
                    "second_review_required_count": 10,
                    "second_review_pass_count": 10,
                    "second_review_pending_count": 0,
                    "second_review_fail_count": 0,
                    "revision_count": 1,
                    "rejected_count": 0,
                    "output_dataset_sha256": dataset_sha256,
                    "evidence_group_sha256": groups_sha256,
                    "review_csv_sha256": "A" * 64,
                }
            ),
            encoding="utf-8",
        )
        exclusions_path.write_text(
            json.dumps(
                {
                    "version": "v1.5.0",
                    "pilot_group_ids": [
                        group["group_id"] for group in reversed(groups)
                    ],
                    "pilot_anchor_evidence_ids": sorted(
                        {
                            evidence_id
                            for group in groups
                            for evidence_id in group[
                                "anchor_evidence_ids"
                            ]
                        }
                    ),
                    "pilot_anchor_clause_ids": sorted(
                        {
                            clause_id
                            for group in groups
                            for clause_id in group[
                                "anchor_clause_ids"
                            ]
                        }
                    ),
                }
            ),
            encoding="utf-8",
        )
        chunk_manifest_path.write_text(
            json.dumps(
                {
                    "version": "v1.5.0",
                    "evidence_sha256": evidence_sha256,
                }
            ),
            encoding="utf-8",
        )
        chunk_sha256 = hashlib.sha256(
            chunk_manifest_path.read_bytes()
        ).hexdigest().upper()
        quality_gate_path.write_text(
            json.dumps(
                {
                    "version": "v1.5.0",
                    "status": "ready",
                    "evidence_sha256": evidence_sha256,
                    "chunk_manifest_sha256": chunk_sha256,
                }
            ),
            encoding="utf-8",
        )
        quality_gate_sha256 = hashlib.sha256(
            quality_gate_path.read_bytes()
        ).hexdigest().upper()
        model_manifest_path.write_text(
            json.dumps({"version": "v1.5.0"}),
            encoding="utf-8",
        )
        model_sha256 = hashlib.sha256(
            model_manifest_path.read_bytes()
        ).hexdigest().upper()
        index_manifest_path.write_text(
            json.dumps(
                {
                    "version": "v1.5.0",
                    "chunk_manifest_sha256": chunk_sha256,
                    "quality_gate_sha256": quality_gate_sha256,
                    "model_manifest_sha256": model_sha256,
                }
            ),
            encoding="utf-8",
        )
        index_sha256 = hashlib.sha256(
            index_manifest_path.read_bytes()
        ).hexdigest().upper()
        config_path.write_text("version: v1.5.0\n", encoding="utf-8")
        config_sha256 = hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest().upper()
        smoke_manifest_path.write_text(
            json.dumps(
                {
                    "version": "v1.5.0",
                    "status": "passed",
                    "inputs": {
                        "config_sha256": config_sha256,
                        "evidence_sha256": evidence_sha256,
                        "index_manifest_sha256": index_sha256,
                        "model_manifest_sha256": model_sha256,
                        "quality_gate_sha256": quality_gate_sha256,
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "dataset_path": dataset_path,
            "evidence_groups_path": groups_path,
            "review_summary_path": review_summary_path,
            "exclusions_path": exclusions_path,
            "manifest_path": manifest_path,
            "evidence_path": evidence_path,
            "chunk_manifest_path": chunk_manifest_path,
            "quality_gate_path": quality_gate_path,
            "index_manifest_path": index_manifest_path,
            "model_manifest_path": model_manifest_path,
            "config_path": config_path,
            "smoke_manifest_path": smoke_manifest_path,
        }

    def test_freezes_ready_manifest_with_complete_private_safe_hash_chain(
        self,
    ) -> None:
        from experiments.rag_v1_5.dataset import freeze_pilot_manifest

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.prepare_freeze_files(
                Path(temporary_directory)
            )
            first = freeze_pilot_manifest(**paths)
            second = freeze_pilot_manifest(**paths)
            serialized = json.dumps(first, ensure_ascii=False)

        self.assertEqual(first["status"], "ready")
        self.assertEqual(first["dataset"]["question_count"], 40)
        self.assertEqual(first["dataset"]["answerable_count"], 32)
        self.assertEqual(first["dataset"]["unanswerable_count"], 8)
        self.assertEqual(first["dataset"]["approved_count"], 40)
        self.assertEqual(first["review"]["first_pass_count"], 40)
        self.assertEqual(first["review"]["second_pass_count"], 10)
        self.assertEqual(first["review"]["revision_count"], 1)
        self.assertEqual(first["review"]["rejected_count"], 0)
        self.assertEqual(
            set(first["inputs"]),
            {
                "evidence_group",
                "review_summary",
                "review_csv_sha256",
                "exclusions",
                "evidence",
                "chunk_manifest",
                "quality_gate",
                "index_manifest",
                "model_manifest",
                "config",
                "smoke_manifest",
            },
        )
        self.assertTrue(
            all(
                len(value["sha256"]) == 64
                for value in first["inputs"].values()
                if isinstance(value, dict)
            )
        )
        self.assertNotIn("测试 shang_han_lun", serialized)
        self.assertNotIn("reference_answer", serialized)
        self.assertNotIn("support_spans", serialized)
        self.assertNotIn("first_comment", serialized)
        first_without_time = dict(first)
        second_without_time = dict(second)
        first_without_time.pop("frozen_at")
        second_without_time.pop("frozen_at")
        self.assertEqual(first_without_time, second_without_time)

    def test_rejects_incomplete_review_hash_mismatch_and_bad_quota(
        self,
    ) -> None:
        from experiments.rag_v1_5.dataset import freeze_pilot_manifest

        cases = ("missing", "pending", "hash", "quota")
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    paths = self.prepare_freeze_files(
                        Path(temporary_directory)
                    )
                    if case == "missing":
                        paths["review_summary_path"].unlink()
                        expected_error = FileNotFoundError
                    else:
                        summary = json.loads(
                            paths["review_summary_path"].read_text(
                                encoding="utf-8"
                            )
                        )
                        if case == "pending":
                            summary["first_review_pass_count"] = 39
                            summary["first_review_pending_count"] = 1
                        elif case == "hash":
                            summary["output_dataset_sha256"] = "B" * 64
                        else:
                            dataset_records = [
                                json.loads(line)
                                for line in paths[
                                    "dataset_path"
                                ].read_text(
                                    encoding="utf-8"
                                ).splitlines()
                            ]
                            write_jsonl_records(
                                paths["dataset_path"],
                                dataset_records[:-1],
                            )
                            summary["output_dataset_sha256"] = (
                                hashlib.sha256(
                                    paths["dataset_path"].read_bytes()
                                ).hexdigest().upper()
                            )
                        paths["review_summary_path"].write_text(
                            json.dumps(summary),
                            encoding="utf-8",
                        )
                        expected_error = ValueError

                    with self.assertRaises(expected_error):
                        freeze_pilot_manifest(**paths)

    def test_rejects_exclusion_mismatch_and_ready_manifest_input_change(
        self,
    ) -> None:
        from experiments.rag_v1_5.dataset import freeze_pilot_manifest

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.prepare_freeze_files(
                Path(temporary_directory)
            )
            exclusions = json.loads(
                paths["exclusions_path"].read_text(encoding="utf-8")
            )
            exclusions["pilot_group_ids"].pop()
            paths["exclusions_path"].write_text(
                json.dumps(exclusions),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "排除清单"):
                freeze_pilot_manifest(**paths)

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.prepare_freeze_files(
                Path(temporary_directory)
            )
            freeze_pilot_manifest(**paths)
            smoke_manifest = json.loads(
                paths["smoke_manifest_path"].read_text(encoding="utf-8")
            )
            smoke_manifest["generated_at"] = "changed"
            paths["smoke_manifest_path"].write_text(
                json.dumps(smoke_manifest),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                freeze_pilot_manifest(**paths)


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

    def test_sample_pilot_evidence_cli_contract_matches_plan(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        args = build_parser().parse_args(
            [
                "sample-pilot-evidence",
                "--seed",
                "20260614",
            ]
        )

        self.assertEqual(args.command, "sample-pilot-evidence")
        self.assertEqual(args.seed, 20260614)
        self.assertEqual(
            args.output,
            Path(
                "data/rag_v1_5/evaluation/"
                "pilot-evidence-groups.jsonl"
            ),
        )
        self.assertEqual(
            args.exclusions,
            Path(
                "data/rag_v1_5/evaluation/"
                "pilot-exclusions.json"
            ),
        )

    def test_pilot_review_cli_contract_matches_plan(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        prepare_args = build_parser().parse_args(
            ["prepare-pilot-review"]
        )
        import_args = build_parser().parse_args(
            ["import-pilot-review"]
        )

        self.assertEqual(
            prepare_args.dataset,
            Path("data/rag_v1_5/evaluation/pilot-40-draft.jsonl"),
        )
        self.assertEqual(
            prepare_args.review_csv,
            Path("data/rag_v1_5/evaluation/pilot-review.csv"),
        )
        self.assertEqual(prepare_args.second_review_seed, 20260614)
        self.assertEqual(
            import_args.output,
            Path("data/rag_v1_5/evaluation/pilot-40.jsonl"),
        )
        self.assertEqual(
            import_args.summary,
            Path(
                "data/rag_v1_5/evaluation/"
                "pilot-review-summary.json"
            ),
        )

    def test_freeze_pilot_cli_contract_matches_plan(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        args = build_parser().parse_args(["freeze-pilot-dataset"])

        self.assertEqual(args.command, "freeze-pilot-dataset")
        self.assertEqual(
            args.dataset,
            Path("data/rag_v1_5/evaluation/pilot-40.jsonl"),
        )
        self.assertEqual(
            args.review_summary,
            Path(
                "data/rag_v1_5/evaluation/"
                "pilot-review-summary.json"
            ),
        )
        self.assertEqual(
            args.manifest,
            Path(
                "experiments/rag_v1_5/manifests/"
                "pilot-40-v1.5.0.json"
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
