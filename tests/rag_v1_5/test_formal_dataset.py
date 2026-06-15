import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


BOOKS = ("shang_han_lun", "jin_gui_yao_lue")
SPLITS = ("formal_dev", "formal_test")
FORMAL_PER_BOOK_SPLIT = {
    "single_clause_fact": 30,
    "formula_composition_or_use": 20,
    "source_location": 10,
    "multi_evidence": 20,
    "unanswerable": 20,
}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def make_evidence(
    *,
    evidence_id: str,
    clause_id: str,
    book: str,
    text: str,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "book_id": book,
        "book_title": (
            "伤寒论" if book == "shang_han_lun" else "金匮要略方论"
        ),
        "volume": "",
        "chapter_id": f"{book}-chapter",
        "chapter_title": "测试篇章",
        "clause_id": clause_id,
        "clause_number": 1,
        "content_type": "clause",
        "parent_id": clause_id,
        "original_text": text,
        "normalized_text": text,
        "notes": [],
        "source_file": f"{book}.txt",
        "source_hash": "A" * 64,
        "corpus_version": "v1.5.0",
    }


def make_formal_artifacts() -> tuple[list[dict], list[dict], list[dict]]:
    questions = []
    groups = []
    evidence = []
    question_number = 0
    clause_number = 0

    for book in BOOKS:
        for split in SPLITS:
            for question_type, count in FORMAL_PER_BOOK_SPLIT.items():
                for cell_index in range(1, count + 1):
                    question_number += 1
                    group_id = f"formal-group-{question_number:03d}"
                    question_id = f"formal-{question_number:03d}"
                    answerable = question_type != "unanswerable"
                    anchor_evidence_ids = []
                    anchor_clause_ids = []
                    support_spans = []
                    if answerable:
                        anchor_count = (
                            2 if question_type == "multi_evidence" else 1
                        )
                        for anchor_index in range(anchor_count):
                            clause_number += 1
                            clause_id = (
                                f"{book}-{split}-clause-{clause_number:03d}"
                            )
                            evidence_id = clause_id
                            text = (
                                f"{book} {split} {question_type} "
                                f"{cell_index} 证据 {anchor_index + 1}"
                            )
                            evidence.append(
                                make_evidence(
                                    evidence_id=evidence_id,
                                    clause_id=clause_id,
                                    book=book,
                                    text=text,
                                )
                            )
                            anchor_evidence_ids.append(evidence_id)
                            anchor_clause_ids.append(clause_id)
                            support_spans.append(text)

                    groups.append(
                        {
                            "group_id": group_id,
                            "split": split,
                            "book_scope": book,
                            "question_type": question_type,
                            "anchor_evidence_ids": anchor_evidence_ids,
                            "anchor_clause_ids": anchor_clause_ids,
                            "selection_seed": 20260614,
                            "selection_reason": "固定配额测试",
                            "absence_queries": (
                                ["缺失查询一", "缺失查询二"]
                                if not answerable
                                else []
                            ),
                        }
                    )
                    questions.append(
                        {
                            "question_id": question_id,
                            "question": (
                                f"{book} {split} {question_type} "
                                f"{cell_index} 的测试问题？"
                            ),
                            "question_type": question_type,
                            "book_scope": book,
                            "answerable": answerable,
                            "reference_answer": (
                                "；".join(support_spans)
                                if answerable
                                else "当前指定古籍范围内无答案。"
                            ),
                            "gold_evidence_ids": anchor_evidence_ids,
                            "gold_clause_ids": anchor_clause_ids,
                            "graded_relevance": {
                                clause_id: 2
                                for clause_id in anchor_clause_ids
                            },
                            "support_spans": support_spans,
                            "review_status": "approved",
                            "split": split,
                            "evidence_group_id": group_id,
                            "question_version": 1,
                        }
                    )

    return questions, groups, evidence


class FormalDatasetValidationTests(unittest.TestCase):
    def validate(
        self,
        root: Path,
        questions: list[dict],
        groups: list[dict],
        evidence: list[dict],
        *,
        exclusions: dict | None = None,
        prior_questions: list[dict] | None = None,
    ) -> dict:
        from experiments.rag_v1_5.formal_dataset import (
            validate_formal_dataset,
        )

        dataset_path = root / "formal-400-draft.jsonl"
        groups_path = root / "formal-evidence-groups.jsonl"
        evidence_path = root / "evidence.jsonl"
        exclusions_path = root / "formal-exclusions.json"
        prior_path = root / "prior.jsonl"
        write_jsonl(dataset_path, questions)
        write_jsonl(groups_path, groups)
        write_jsonl(evidence_path, evidence)
        exclusions_path.write_text(
            json.dumps(
                exclusions
                or {
                    "prior_group_ids": [],
                    "prior_evidence_ids": [],
                    "prior_clause_ids": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_jsonl(prior_path, prior_questions or [])
        return validate_formal_dataset(
            dataset_path=dataset_path,
            evidence_path=evidence_path,
            evidence_groups_path=groups_path,
            exclusions_path=exclusions_path,
            prior_dataset_paths=(prior_path,),
        )

    def test_valid_formal_dataset_has_fixed_400_question_contract(
        self,
    ) -> None:
        questions, groups, evidence = make_formal_artifacts()
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = self.validate(
                Path(temporary_directory),
                questions,
                groups,
                evidence,
            )

        self.assertEqual(summary["question_count"], 400)
        self.assertEqual(summary["answerable_count"], 320)
        self.assertEqual(summary["unanswerable_count"], 80)
        self.assertEqual(
            summary["split_counts"],
            {"formal_dev": 200, "formal_test": 200},
        )
        self.assertEqual(
            summary["book_counts"],
            {"shang_han_lun": 200, "jin_gui_yao_lue": 200},
        )
        self.assertEqual(summary["prior_overlap_count"], 0)
        self.assertEqual(summary["cross_split_clause_overlap_count"], 0)
        self.assertEqual(summary["duplicate_question_count"], 0)
        self.assertEqual(summary["status"], "ready")

    def test_rejects_duplicate_question_id_group_id_and_text(self) -> None:
        questions, groups, evidence = make_formal_artifacts()
        cases = {}

        duplicate_question_id = deepcopy(questions)
        duplicate_question_id[1]["question_id"] = (
            duplicate_question_id[0]["question_id"]
        )
        cases["question_id"] = (duplicate_question_id, groups)

        duplicate_group_id = deepcopy(groups)
        duplicate_group_id[1]["group_id"] = duplicate_group_id[0]["group_id"]
        cases["group_id"] = (questions, duplicate_group_id)

        duplicate_question = deepcopy(questions)
        duplicate_question[1]["question"] = (
            "  "
            + duplicate_question[0]["question"].replace("？", "?")
            + "  "
        )
        cases["问题文本"] = (duplicate_question, groups)

        for label, (case_questions, case_groups) in cases.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaisesRegex(ValueError, label):
                        self.validate(
                            Path(temporary_directory),
                            case_questions,
                            case_groups,
                            evidence,
                        )

    def test_rejects_cross_split_clause_and_answerable_anchor_reuse(
        self,
    ) -> None:
        questions, groups, evidence = make_formal_artifacts()

        cross_split_questions = deepcopy(questions)
        cross_split_groups = deepcopy(groups)
        dev_index = next(
            index
            for index, question in enumerate(cross_split_questions)
            if question["split"] == "formal_dev" and question["answerable"]
        )
        test_index = next(
            index
            for index, question in enumerate(cross_split_questions)
            if question["split"] == "formal_test" and question["answerable"]
        )
        reused_clause = cross_split_questions[dev_index]["gold_clause_ids"][0]
        reused_evidence = cross_split_questions[dev_index][
            "gold_evidence_ids"
        ][0]
        cross_split_questions[test_index]["gold_clause_ids"][0] = (
            reused_clause
        )
        cross_split_questions[test_index]["gold_evidence_ids"][0] = (
            reused_evidence
        )
        cross_split_questions[test_index]["graded_relevance"] = {
            reused_clause: 2
        }
        cross_split_questions[test_index]["support_spans"] = [
            cross_split_questions[dev_index]["support_spans"][0]
        ]
        cross_split_questions[test_index]["reference_answer"] = (
            cross_split_questions[dev_index]["reference_answer"]
        )
        cross_split_groups[test_index]["anchor_clause_ids"][0] = reused_clause
        cross_split_groups[test_index]["anchor_evidence_ids"][0] = (
            reused_evidence
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "dev/test"):
                self.validate(
                    Path(temporary_directory),
                    cross_split_questions,
                    cross_split_groups,
                    evidence,
                )

        same_split_questions = deepcopy(questions)
        same_split_groups = deepcopy(groups)
        first_index = next(
            index
            for index, question in enumerate(same_split_questions)
            if question["split"] == "formal_dev" and question["answerable"]
        )
        second_index = next(
            index
            for index, question in enumerate(same_split_questions)
            if (
                index > first_index
                and question["split"] == "formal_dev"
                and question["answerable"]
                and question["question_type"] != "multi_evidence"
            )
        )
        reused_clause = same_split_questions[first_index]["gold_clause_ids"][0]
        reused_evidence = same_split_questions[first_index][
            "gold_evidence_ids"
        ][0]
        same_split_questions[second_index]["gold_clause_ids"] = [
            reused_clause
        ]
        same_split_questions[second_index]["gold_evidence_ids"] = [
            reused_evidence
        ]
        same_split_questions[second_index]["graded_relevance"] = {
            reused_clause: 2
        }
        same_split_questions[second_index]["support_spans"] = [
            same_split_questions[first_index]["support_spans"][0]
        ]
        same_split_questions[second_index]["reference_answer"] = (
            same_split_questions[first_index]["reference_answer"]
        )
        same_split_groups[second_index]["anchor_clause_ids"] = [reused_clause]
        same_split_groups[second_index]["anchor_evidence_ids"] = [
            reused_evidence
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "anchor"):
                self.validate(
                    Path(temporary_directory),
                    same_split_questions,
                    same_split_groups,
                    evidence,
                )

    def test_rejects_prior_overlap_and_multi_evidence_scope_leaks(
        self,
    ) -> None:
        questions, groups, evidence = make_formal_artifacts()
        answerable = next(
            question for question in questions if question["answerable"]
        )
        prior_question = {
            **answerable,
            "question_id": "pilot-overlap",
            "question": "历史问题",
            "split": "pilot",
            "evidence_group_id": "pilot-group",
        }
        exclusions = {
            "pilot_group_ids": ["pilot-group"],
            "pilot_anchor_evidence_ids": answerable["gold_evidence_ids"],
            "pilot_anchor_clause_ids": answerable["gold_clause_ids"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "Smoke/Pilot"):
                self.validate(
                    Path(temporary_directory),
                    questions,
                    groups,
                    evidence,
                    exclusions=exclusions,
                    prior_questions=[prior_question],
                )

        multi_index = next(
            index
            for index, question in enumerate(questions)
            if question["question_type"] == "multi_evidence"
        )
        cross_book_questions = deepcopy(questions)
        cross_book_groups = deepcopy(groups)
        foreign_evidence = next(
            item
            for item in evidence
            if item["book_id"] != questions[multi_index]["book_scope"]
        )
        cross_book_questions[multi_index]["gold_evidence_ids"][1] = (
            foreign_evidence["evidence_id"]
        )
        cross_book_questions[multi_index]["gold_clause_ids"][1] = (
            foreign_evidence["clause_id"]
        )
        cross_book_questions[multi_index]["graded_relevance"] = {
            clause_id: 2
            for clause_id in cross_book_questions[multi_index][
                "gold_clause_ids"
            ]
        }
        cross_book_questions[multi_index]["support_spans"][1] = (
            foreign_evidence["normalized_text"]
        )
        cross_book_groups[multi_index]["anchor_evidence_ids"][1] = (
            foreign_evidence["evidence_id"]
        )
        cross_book_groups[multi_index]["anchor_clause_ids"][1] = (
            foreign_evidence["clause_id"]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "同一本书"):
                self.validate(
                    Path(temporary_directory),
                    cross_book_questions,
                    cross_book_groups,
                    evidence,
                )

    def test_rejects_unanswerable_gold_and_missing_formal_metadata(
        self,
    ) -> None:
        questions, groups, evidence = make_formal_artifacts()
        unanswerable_index = next(
            index
            for index, question in enumerate(questions)
            if not question["answerable"]
        )
        invalid_unanswerable = deepcopy(questions)
        invalid_unanswerable[unanswerable_index]["gold_clause_ids"] = [
            evidence[0]["clause_id"]
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                self.validate(
                    Path(temporary_directory),
                    invalid_unanswerable,
                    groups,
                    evidence,
                )

        for field in ("split", "evidence_group_id", "question_version"):
            with self.subTest(field=field):
                missing_metadata = deepcopy(questions)
                missing_metadata[0].pop(field)
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaisesRegex(ValueError, "元数据"):
                        self.validate(
                            Path(temporary_directory),
                            missing_metadata,
                            groups,
                            evidence,
                        )


class FormalDatasetCliTests(unittest.TestCase):
    def test_validate_formal_dataset_cli_contract(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        args = build_parser().parse_args(
            [
                "validate-formal-dataset",
                "--dataset",
                "data/rag_v1_5/formal/evaluation/formal-400-draft.jsonl",
                "--evidence-groups",
                (
                    "data/rag_v1_5/formal/evaluation/"
                    "formal-evidence-groups.jsonl"
                ),
                "--exclusions",
                (
                    "data/rag_v1_5/formal/evaluation/"
                    "formal-exclusions.json"
                ),
            ]
        )

        self.assertEqual(args.command, "validate-formal-dataset")
        self.assertEqual(len(args.prior_dataset), 2)


if __name__ == "__main__":
    unittest.main()
