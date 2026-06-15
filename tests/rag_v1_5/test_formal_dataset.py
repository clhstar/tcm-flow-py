import csv
import json
import hashlib
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml


BOOKS = ("shang_han_lun", "jin_gui_yao_lue")
SPLITS = ("formal_dev", "formal_test")
FORMAL_PER_BOOK_SPLIT = {
    "single_clause_fact": 30,
    "formula_composition_or_use": 20,
    "source_location": 10,
    "multi_evidence": 20,
    "unanswerable": 20,
}
FORMAL_CONFIG_IDS = (
    "b1-c0-bm25",
    "b2-c0-dense",
    "b3-c0-hybrid",
    "b4-c0-hybrid-rerank",
    "c1-hybrid-rerank",
    "c2-hybrid-rerank",
    "c3-hybrid-rerank",
    "p-c4-hybrid-rerank",
    "p-no-parent",
    "p-no-structure",
    "p-no-bm25",
    "p-no-dense",
    "p-no-reranker",
    "p-no-title",
)


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
    clause_number: int = 1,
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
        "clause_number": clause_number,
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
                                    clause_number=clause_number,
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
                                [
                                    f"{split} 缺失查询 {cell_index}",
                                    f"{split} 缺失查询 {cell_index} 相关",
                                ]
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


def make_formal_sampling_evidence() -> list[dict]:
    records = []
    for book_index, book in enumerate(BOOKS, start=1):
        for clause_index in range(1, 241):
            chapter_number = ((clause_index - 1) // 30) + 1
            clause_id = f"{book}-sample-{clause_index:03d}"
            base = {
                "book_id": book,
                "book_title": f"测试书籍 {book_index}",
                "volume": "",
                "chapter_id": f"{book}-chapter-{chapter_number:02d}",
                "chapter_title": f"测试篇章 {chapter_number}",
                "clause_id": clause_id,
                "clause_number": clause_index,
                "parent_id": clause_id,
                "notes": [],
                "source_file": f"{book}.txt",
                "source_hash": f"{book_index}" * 64,
                "corpus_version": "v1.5.0",
            }
            clause_text = f"{book} 第 {clause_index} 条稳定抽样原文"
            records.append(
                {
                    **base,
                    "evidence_id": clause_id,
                    "content_type": "clause",
                    "original_text": clause_text,
                    "normalized_text": clause_text,
                }
            )
            if clause_index <= 60:
                formula_id = f"{clause_id}-formula-01"
                records.extend(
                    [
                        {
                            **base,
                            "evidence_id": formula_id,
                            "content_type": "formula",
                            "parent_id": clause_id,
                            "original_text": f"测试方剂 {clause_index}",
                            "normalized_text": f"测试方剂 {clause_index}",
                        },
                        {
                            **base,
                            "evidence_id": f"{formula_id}-ingredients",
                            "content_type": "ingredients",
                            "parent_id": formula_id,
                            "original_text": "测试药物甲、测试药物乙",
                            "normalized_text": "测试药物甲、测试药物乙",
                        },
                        {
                            **base,
                            "evidence_id": f"{formula_id}-preparation",
                            "content_type": "preparation",
                            "parent_id": formula_id,
                            "original_text": "测试煎服方法",
                            "normalized_text": "测试煎服方法",
                        },
                    ]
                )
            if 61 <= clause_index <= 110:
                records.append(
                    {
                        **base,
                        "evidence_id": f"{clause_id}-note-01",
                        "content_type": "note",
                        "parent_id": clause_id,
                        "original_text": f"测试校注 {clause_index}",
                        "normalized_text": f"测试校注 {clause_index}",
                    }
                )
    return records


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
        self.assertEqual(summary["answerable_anchor_clause_count"], 400)
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

    def test_sample_formal_evidence_cli_contract(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        args = build_parser().parse_args(["sample-formal-evidence"])

        self.assertEqual(args.command, "sample-formal-evidence")
        self.assertEqual(args.seed, 20260614)
        self.assertEqual(
            args.output,
            Path(
                "data/rag_v1_5/formal/evaluation/"
                "formal-evidence-groups.jsonl"
            ),
        )

    def test_freeze_formal_prereg_cli_contract(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        args = build_parser().parse_args(["freeze-formal-prereg"])

        self.assertEqual(args.command, "freeze-formal-prereg")
        self.assertEqual(
            args.config,
            Path("experiments/rag_v1_5/configs/formal-400.yaml"),
        )
        self.assertEqual(
            args.output,
            Path(
                "experiments/rag_v1_5/manifests/"
                "formal-prereg-v1.5.0.json"
            ),
        )

    def test_formal_authoring_cli_contract(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        prepare_args = build_parser().parse_args(
            ["prepare-formal-authoring"]
        )
        draft_args = build_parser().parse_args(
            ["draft-formal-authoring"]
        )
        import_args = build_parser().parse_args(
            ["import-formal-authoring"]
        )

        self.assertEqual(
            prepare_args.output,
            Path(
                "data/rag_v1_5/formal/evaluation/"
                "formal-authoring.csv"
            ),
        )
        self.assertEqual(
            draft_args.authoring_csv,
            Path(
                "data/rag_v1_5/formal/evaluation/"
                "formal-authoring.csv"
            ),
        )
        self.assertEqual(
            import_args.output,
            Path(
                "data/rag_v1_5/formal/evaluation/"
                "formal-400-draft.jsonl"
            ),
        )


class FormalEvidenceSamplingTests(unittest.TestCase):
    def run_sampling(
        self,
        root: Path,
        evidence_records: list[dict],
        *,
        suffix: str,
    ) -> tuple[dict, Path, Path, Path]:
        from experiments.rag_v1_5.formal_dataset import (
            sample_formal_evidence_groups,
        )

        evidence_path = root / f"evidence-{suffix}.jsonl"
        smoke_path = root / "smoke-10.jsonl"
        pilot_path = root / "pilot-40.jsonl"
        pilot_exclusions_path = root / "pilot-exclusions.json"
        output_path = root / f"groups-{suffix}.jsonl"
        exclusions_path = root / f"exclusions-{suffix}.json"
        report_path = root / f"report-{suffix}.json"
        write_jsonl(evidence_path, evidence_records)

        prior_clause = min(
            evidence_records,
            key=lambda record: record["evidence_id"],
        )
        prior_question = {
            "question_id": "prior-001",
            "question": "历史测试问题？",
            "question_type": "single_clause_fact",
            "book_scope": prior_clause["book_id"],
            "answerable": True,
            "reference_answer": prior_clause["normalized_text"],
            "gold_evidence_ids": [prior_clause["evidence_id"]],
            "gold_clause_ids": [prior_clause["clause_id"]],
            "graded_relevance": {prior_clause["clause_id"]: 2},
            "support_spans": [prior_clause["normalized_text"]],
            "review_status": "approved",
            "split": "smoke",
            "evidence_group_id": None,
            "question_version": 1,
        }
        write_jsonl(smoke_path, [prior_question])
        write_jsonl(pilot_path, [])
        pilot_exclusions_path.write_text(
            json.dumps(
                {
                    "future_formal_excluded_evidence_ids": [
                        prior_clause["evidence_id"]
                    ],
                    "future_formal_excluded_clause_ids": [
                        prior_clause["clause_id"]
                    ],
                    "pilot_group_ids": [],
                }
            ),
            encoding="utf-8",
        )

        summary = sample_formal_evidence_groups(
            evidence_path=evidence_path,
            smoke_dataset_path=smoke_path,
            pilot_dataset_path=pilot_path,
            pilot_exclusions_path=pilot_exclusions_path,
            output_path=output_path,
            exclusions_path=exclusions_path,
            candidate_report_path=report_path,
            seed=20260614,
        )
        return summary, output_path, exclusions_path, report_path

    def test_sampling_is_deterministic_and_meets_fixed_contract(
        self,
    ) -> None:
        evidence_records = make_formal_sampling_evidence()
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
            first_summary, first_groups, first_exclusions, first_report = (
                first
            )
            _, second_groups, second_exclusions, second_report = second

            for first_path, second_path in (
                (first_groups, second_groups),
                (first_exclusions, second_exclusions),
                (first_report, second_report),
            ):
                self.assertEqual(
                    hashlib.sha256(first_path.read_bytes()).hexdigest(),
                    hashlib.sha256(second_path.read_bytes()).hexdigest(),
                )

            groups = [
                json.loads(line)
                for line in first_groups.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            exclusions = json.loads(
                first_exclusions.read_text(encoding="utf-8")
            )

        self.assertEqual(first_summary["status"], "ready")
        self.assertEqual(first_summary["group_count"], 400)
        self.assertEqual(first_summary["answerable_group_count"], 320)
        self.assertEqual(first_summary["unanswerable_group_count"], 80)
        answerable_groups = [
            group
            for group in groups
            if group["question_type"] != "unanswerable"
        ]
        anchor_clause_ids = [
            clause_id
            for group in answerable_groups
            for clause_id in group["anchor_clause_ids"]
        ]
        self.assertEqual(len(anchor_clause_ids), 400)
        self.assertEqual(len(set(anchor_clause_ids)), 400)
        dev_clauses = {
            clause_id
            for group in answerable_groups
            if group["split"] == "formal_dev"
            for clause_id in group["anchor_clause_ids"]
        }
        test_clauses = {
            clause_id
            for group in answerable_groups
            if group["split"] == "formal_test"
            for clause_id in group["anchor_clause_ids"]
        }
        self.assertFalse(dev_clauses & test_clauses)
        self.assertNotIn(
            min(
                make_formal_sampling_evidence(),
                key=lambda record: record["evidence_id"],
            )["clause_id"],
            anchor_clause_ids,
        )
        self.assertEqual(
            exclusions["prior_clause_ids"],
            [
                min(
                    make_formal_sampling_evidence(),
                    key=lambda record: record["evidence_id"],
                )["clause_id"]
            ],
        )
        for group in groups:
            if group["question_type"] == "unanswerable":
                self.assertEqual(len(group["absence_queries"]), 2)

    def test_candidate_audit_blocks_insufficient_formula_stratum(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_dataset import (
            audit_formal_candidates,
        )

        evidence_records = [
            record
            for record in make_formal_sampling_evidence()
            if not (
                record["book_id"] == "jin_gui_yao_lue"
                and record["content_type"]
                in {"formula", "ingredients", "preparation"}
            )
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence_path = root / "evidence.jsonl"
            exclusions_path = root / "pilot-exclusions.json"
            write_jsonl(evidence_path, evidence_records)
            exclusions_path.write_text(
                json.dumps(
                    {
                        "future_formal_excluded_evidence_ids": [],
                        "future_formal_excluded_clause_ids": [],
                    }
                ),
                encoding="utf-8",
            )
            report = audit_formal_candidates(
                evidence_path=evidence_path,
                prior_exclusions_path=exclusions_path,
            )

        self.assertEqual(report["status"], "blocked")
        self.assertIn(
            "jin_gui_yao_lue/formal_dev/"
            "formula_composition_or_use",
            report["blocked_strata"],
        )


def make_formal_config() -> dict:
    matrix = []
    definitions = (
        ("b1-c0-bm25", "B1", "c0", "bm25", "parent", "with_titles"),
        ("b2-c0-dense", "B2", "c0", "dense", "parent", "with_titles"),
        ("b3-c0-hybrid", "B3", "c0", "hybrid", "parent", "with_titles"),
        (
            "b4-c0-hybrid-rerank",
            "B4/C0",
            "c0",
            "hybrid_rerank",
            "parent",
            "with_titles",
        ),
        (
            "c1-hybrid-rerank",
            "C1",
            "c1",
            "hybrid_rerank",
            "parent",
            "with_titles",
        ),
        (
            "c2-hybrid-rerank",
            "C2",
            "c2",
            "hybrid_rerank",
            "parent",
            "with_titles",
        ),
        (
            "c3-hybrid-rerank",
            "C3",
            "c3",
            "hybrid_rerank",
            "parent",
            "with_titles",
        ),
        (
            "p-c4-hybrid-rerank",
            "P/C4",
            "c4",
            "hybrid_rerank",
            "parent",
            "with_titles",
        ),
        (
            "p-no-parent",
            "P-Parent",
            "c4",
            "hybrid_rerank",
            "child",
            "with_titles",
        ),
        (
            "p-no-structure",
            "P-Structure",
            "c5",
            "hybrid_rerank",
            "parent",
            "with_titles",
        ),
        (
            "p-no-bm25",
            "P-BM25",
            "c4",
            "dense_rerank",
            "parent",
            "with_titles",
        ),
        (
            "p-no-dense",
            "P-Dense",
            "c4",
            "bm25_rerank",
            "parent",
            "with_titles",
        ),
        (
            "p-no-reranker",
            "P-Reranker",
            "c4",
            "hybrid",
            "parent",
            "with_titles",
        ),
        (
            "p-no-title",
            "P-Title",
            "c4",
            "hybrid_rerank",
            "parent",
            "without_titles",
        ),
    )
    for (
        config_id,
        paper_role,
        strategy,
        mode,
        context_policy,
        metadata_policy,
    ) in definitions:
        matrix.append(
            {
                "config_id": config_id,
                "paper_role": paper_role,
                "strategy": strategy,
                "mode": mode,
                "context_policy": context_policy,
                "metadata_policy": metadata_policy,
            }
        )
    return {
        "version": "v1.5.0",
        "seed": 20260614,
        "dataset": {"dev_count": 200, "test_count": 200},
        "retrieval": {
            "bm25_top_k": 20,
            "dense_top_k": 20,
            "rrf_k": 60,
            "reranker_candidate_k": 40,
            "result_top_k": 10,
            "primary_report_top_k": 5,
        },
        "embedding": {
            "model": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "device": "cuda",
            "use_fp16": True,
            "batch_size": 4,
            "max_length": 1024,
            "normalize": True,
        },
        "bm25": {
            "tokenizer": "jieba",
            "hmm": False,
            "top_k": 20,
        },
        "dense": {"top_k": 20},
        "rrf": {"k": 60},
        "reranker": {
            "model": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "device": "cuda",
            "use_fp16": True,
            "batch_size": 2,
            "max_length": 1024,
            "candidate_k": 40,
            "top_k": 10,
            "normalize_score": True,
        },
        "evaluation": {
            "top_ks": [1, 5, 10],
            "primary_granularity": "clause",
        },
        "statistics": {
            "bootstrap_seed": 20260614,
            "bootstrap_resamples": 10000,
            "confidence_level": 0.95,
            "strata": ["book_scope", "question_type"],
            "primary_metrics": [
                "recall_at_5",
                "mrr_at_10",
                "ndcg_at_10",
            ],
        },
        "quota_per_book_split": FORMAL_PER_BOOK_SPLIT,
        "matrix": matrix,
        "comparisons": {
            "primary": {
                "a": "p-c4-hybrid-rerank",
                "b": "b4-c0-hybrid-rerank",
            },
            "ablations": [
                {"a": "p-c4-hybrid-rerank", "b": config_id}
                for config_id in (
                    "p-no-parent",
                    "p-no-structure",
                    "p-no-bm25",
                    "p-no-dense",
                    "p-no-reranker",
                    "p-no-title",
                )
            ],
        },
        "provenance": {"pilot_config_sha256": "B" * 64},
    }


class FormalPreregistrationTests(unittest.TestCase):
    def prepare_files(self, root: Path) -> dict[str, Path]:
        from experiments.rag_v1_5.formal_dataset import _sha256_file

        _, groups, _ = make_formal_artifacts()
        config_path = root / "formal-400.yaml"
        groups_path = root / "formal-evidence-groups.jsonl"
        exclusions_path = root / "formal-exclusions.json"
        pilot_manifest_path = root / "pilot-40-v1.5.0.json"
        pilot_runs_path = root / "pilot-runs-v1.5.0.json"
        output_path = root / "formal-prereg-v1.5.0.json"

        config = make_formal_config()
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        write_jsonl(groups_path, groups)
        exclusions_path.write_text(
            json.dumps(
                {
                    "prior_group_ids": ["pilot-group"],
                    "prior_evidence_ids": ["pilot-evidence"],
                    "prior_clause_ids": ["pilot-clause"],
                }
            ),
            encoding="utf-8",
        )
        pilot_manifest_path.write_text(
            json.dumps(
                {
                    "version": "v1.5.0",
                    "status": "ready",
                    "inputs": {
                        "config": {"sha256": "B" * 64},
                        "model_manifest": {"sha256": "C" * 64},
                    },
                }
            ),
            encoding="utf-8",
        )
        pilot_runs_path.write_text(
            json.dumps(
                {
                    "version": "v1.5.0",
                    "status": "ready",
                    "config_count": 8,
                    "completed_config_count": 8,
                    "failed_config_count": 0,
                    "input_hashes": {
                        "pilot_manifest_sha256": _sha256_file(
                            pilot_manifest_path
                        )
                    },
                }
            ),
            encoding="utf-8",
        )
        return {
            "config_path": config_path,
            "evidence_groups_path": groups_path,
            "exclusions_path": exclusions_path,
            "pilot_manifest_path": pilot_manifest_path,
            "pilot_runs_manifest_path": pilot_runs_path,
            "output_path": output_path,
        }

    def test_freezes_private_safe_idempotent_preregistration(self) -> None:
        from experiments.rag_v1_5.formal_dataset import (
            freeze_formal_preregistration,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.prepare_files(Path(temporary_directory))
            first = freeze_formal_preregistration(**paths)
            second = freeze_formal_preregistration(**paths)
            serialized = json.dumps(first, ensure_ascii=False)

        self.assertEqual(first["status"], "ready")
        self.assertEqual(len(first["matrix"]), 14)
        self.assertEqual(
            {row["config_id"] for row in first["matrix"]},
            set(FORMAL_CONFIG_IDS),
        )
        self.assertEqual(len(first["comparisons"]["ablations"]), 6)
        self.assertEqual(first["dataset"]["question_count"], 400)
        self.assertEqual(first["dataset"]["dev_count"], 200)
        self.assertEqual(first["dataset"]["test_count"], 200)
        self.assertEqual(first["statistics"]["bootstrap_resamples"], 10000)
        self.assertEqual(first, second)
        self.assertNotIn("测试问题", serialized)
        self.assertNotIn("reference_answer", serialized)
        self.assertNotIn("support_spans", serialized)
        self.assertNotIn("original_text", serialized)

    def test_rejects_incomplete_matrix_and_ready_input_change(self) -> None:
        from experiments.rag_v1_5.formal_dataset import (
            freeze_formal_preregistration,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.prepare_files(Path(temporary_directory))
            config = yaml.safe_load(
                paths["config_path"].read_text(encoding="utf-8")
            )
            config["matrix"].pop()
            paths["config_path"].write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "14"):
                freeze_formal_preregistration(**paths)

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.prepare_files(Path(temporary_directory))
            freeze_formal_preregistration(**paths)
            exclusions = json.loads(
                paths["exclusions_path"].read_text(encoding="utf-8")
            )
            exclusions["prior_clause_ids"].append("changed")
            paths["exclusions_path"].write_text(
                json.dumps(exclusions),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                freeze_formal_preregistration(**paths)

    def test_rejects_config_missing_runtime_retrieval_fields(self) -> None:
        from experiments.rag_v1_5.formal_dataset import (
            freeze_formal_preregistration,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = self.prepare_files(Path(temporary_directory))
            config = yaml.safe_load(
                paths["config_path"].read_text(encoding="utf-8")
            )
            config.pop("bm25")
            paths["config_path"].write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "运行字段"):
                freeze_formal_preregistration(**paths)


class FormalAuthoringWorkflowTests(unittest.TestCase):
    def prepare_authoring_files(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path, list[dict]]:
        from experiments.rag_v1_5.formal_dataset import (
            prepare_formal_authoring_csv,
        )

        _, groups, evidence = make_formal_artifacts()
        groups_path = root / "formal-evidence-groups.jsonl"
        evidence_path = root / "evidence.jsonl"
        authoring_path = root / "formal-authoring.csv"
        write_jsonl(groups_path, groups)
        write_jsonl(evidence_path, evidence)
        summary = prepare_formal_authoring_csv(
            evidence_groups_path=groups_path,
            evidence_path=evidence_path,
            output_csv_path=authoring_path,
        )
        self.assertEqual(summary["row_count"], 400)
        return groups_path, evidence_path, authoring_path, evidence

    def read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            return list(csv.DictReader(file_handle))

    def write_rows(
        self,
        path: Path,
        rows: list[dict[str, str]],
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def complete_rows(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            context = json.loads(row["evidence_context"])
            row["question"] = f"请回答 {row['question_id']} 对应的问题？"
            if row["question_type"] == "unanswerable":
                row["reference_answer"] = "当前指定古籍范围内无答案。"
                row["gold_evidence_ids"] = "[]"
                row["gold_clause_ids"] = "[]"
                row["graded_relevance"] = "{}"
                row["support_spans"] = "[]"
            else:
                support_spans = [
                    item["normalized_text"] for item in context
                ]
                row["reference_answer"] = "；".join(support_spans)
                row["support_spans"] = json.dumps(
                    support_spans,
                    ensure_ascii=False,
                )
            row["question_version"] = "1"

    def test_prepare_and_import_complete_formal_authoring_csv(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_dataset import (
            FORMAL_AUTHORING_FIELDS,
            import_formal_authoring_csv,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            groups_path, evidence_path, authoring_path, _ = (
                self.prepare_authoring_files(root)
            )
            self.assertTrue(
                authoring_path.read_bytes().startswith(b"\xef\xbb\xbf")
            )
            rows = self.read_rows(authoring_path)
            self.assertEqual(len(rows), 400)
            self.assertEqual(list(rows[0]), list(FORMAL_AUTHORING_FIELDS))
            self.assertTrue(rows[0]["evidence_context"])
            self.assertEqual(rows[0]["question"], "")
            self.complete_rows(rows)
            self.write_rows(authoring_path, rows)
            output_path = root / "formal-400-draft.jsonl"

            summary = import_formal_authoring_csv(
                authoring_csv_path=authoring_path,
                evidence_groups_path=groups_path,
                evidence_path=evidence_path,
                output_dataset_path=output_path,
            )
            records = [
                json.loads(line)
                for line in output_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(summary["status"], "draft")
        self.assertEqual(summary["question_count"], 400)
        self.assertEqual(summary["answerable_count"], 320)
        self.assertEqual(summary["unanswerable_count"], 80)
        self.assertEqual(len(records), 400)
        self.assertTrue(
            all(record["review_status"] == "draft" for record in records)
        )

    def test_draft_formal_authoring_csv_is_complete_and_importable(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_dataset import (
            draft_formal_authoring_csv,
            import_formal_authoring_csv,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            groups_path, evidence_path, authoring_path, _ = (
                self.prepare_authoring_files(root)
            )

            first = draft_formal_authoring_csv(
                authoring_csv_path=authoring_path,
                evidence_groups_path=groups_path,
                evidence_path=evidence_path,
            )
            first_bytes = authoring_path.read_bytes()
            second = draft_formal_authoring_csv(
                authoring_csv_path=authoring_path,
                evidence_groups_path=groups_path,
                evidence_path=evidence_path,
            )
            rows = self.read_rows(authoring_path)

            self.assertEqual(authoring_path.read_bytes(), first_bytes)
            self.assertEqual(first["row_count"], 400)
            self.assertEqual(first["drafted_count"], 400)
            self.assertEqual(first["newly_drafted_count"], 400)
            self.assertEqual(second["newly_drafted_count"], 0)
            self.assertEqual(second["preserved_count"], 400)
            self.assertEqual(first["answerable_count"], 320)
            self.assertEqual(first["unanswerable_count"], 80)
            self.assertEqual(
                len({row["question"] for row in rows}),
                400,
            )
            self.assertTrue(
                all(row["question"] for row in rows)
            )
            self.assertTrue(
                all(row["reference_answer"] for row in rows)
            )

            answerable_rows = [
                row
                for row in rows
                if row["question_type"] != "unanswerable"
            ]
            unanswerable_rows = [
                row
                for row in rows
                if row["question_type"] == "unanswerable"
            ]
            self.assertTrue(
                all(
                    json.loads(row["support_spans"])
                    for row in answerable_rows
                )
            )
            self.assertTrue(
                all(
                    json.loads(row["support_spans"]) == []
                    for row in unanswerable_rows
                )
            )
            self.assertTrue(
                all(
                    json.loads(row["gold_evidence_ids"]) == []
                    for row in unanswerable_rows
                )
            )

            output_path = root / "formal-400-draft.jsonl"
            imported = import_formal_authoring_csv(
                authoring_csv_path=authoring_path,
                evidence_groups_path=groups_path,
                evidence_path=evidence_path,
                output_dataset_path=output_path,
            )

        self.assertEqual(imported["status"], "draft")
        self.assertEqual(imported["question_count"], 400)

    def test_draft_templates_avoid_location_leakage_and_disambiguate_notes(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_dataset import (
            _build_formal_authoring_rows,
            _draft_formal_row,
        )
        from experiments.rag_v1_5.schema import (
            EvidenceUnit,
            FormalEvidenceGroup,
        )

        clause_a = make_evidence(
            evidence_id="shl-chapter-02-052",
            clause_id="shl-chapter-02-052",
            book="shang_han_lun",
            text="脉浮而数者，可发汗，宜麻黄汤。方十八。",
            clause_number=52,
        )
        clause_b = make_evidence(
            evidence_id="shl-chapter-08-354",
            clause_id="shl-chapter-08-354",
            book="shang_han_lun",
            text="大汗，若大下利而厥冷者，四逆汤主之。方六。",
            clause_number=354,
        )
        note_a = deepcopy(clause_a)
        note_a.update(
            {
                "evidence_id": "shl-chapter-02-052-note-01",
                "content_type": "note",
                "normalized_text": "用前第五方。",
                "original_text": "用前第五方。",
            }
        )
        note_b = deepcopy(clause_b)
        note_b.update(
            {
                "evidence_id": "shl-chapter-08-354-note-01",
                "content_type": "note",
                "normalized_text": "用前第五方。",
                "original_text": "用前第五方。",
            }
        )
        groups = [
            FormalEvidenceGroup(
                group_id=(
                    "formal-shang_han_lun-formal_dev-"
                    "source_location-01"
                ),
                split="formal_dev",
                book_scope="shang_han_lun",
                question_type="source_location",
                anchor_evidence_ids=[note_a["evidence_id"]],
                anchor_clause_ids=[note_a["clause_id"]],
                selection_seed=20260614,
                selection_reason="测试重复校注",
                absence_queries=[],
            ),
            FormalEvidenceGroup(
                group_id=(
                    "formal-shang_han_lun-formal_test-"
                    "source_location-01"
                ),
                split="formal_test",
                book_scope="shang_han_lun",
                question_type="source_location",
                anchor_evidence_ids=[note_b["evidence_id"]],
                anchor_clause_ids=[note_b["clause_id"]],
                selection_seed=20260614,
                selection_reason="测试重复校注",
                absence_queries=[],
            ),
        ]
        evidence = [
            EvidenceUnit.model_validate(record)
            for record in (clause_a, clause_b, note_a, note_b)
        ]
        rows = _build_formal_authoring_rows(
            groups=groups,
            evidence_by_id={item.evidence_id: item for item in evidence},
        )

        for row in rows:
            _draft_formal_row(row)

        self.assertNotEqual(rows[0]["question"], rows[1]["question"])
        self.assertIn("脉浮而数者", rows[0]["question"])
        self.assertIn("大汗，若大下利", rows[1]["question"])
        self.assertNotIn("第52条", rows[0]["question"])
        self.assertNotIn("第354条", rows[1]["question"])
        self.assertIn("第52条", rows[0]["reference_answer"])
        self.assertIn("第354条", rows[1]["reference_answer"])

    def test_draft_support_spans_preserve_exact_evidence_whitespace(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_dataset import (
            _formal_support_spans,
        )

        evidence_text = (
            "大承气汤\n"
            "大黄（酒洗，四两） 芒硝（三合）\n"
            "上二味，以水煎服。"
        )
        spans = _formal_support_spans(
            [{"normalized_text": evidence_text}]
        )

        self.assertEqual(spans, [evidence_text])

    def test_import_rejects_immutable_duplicate_and_clinical_edits(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_dataset import (
            import_formal_authoring_csv,
        )

        cases = ("immutable", "duplicate", "clinical", "out_of_scope_gold")
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    groups_path, evidence_path, authoring_path, _ = (
                        self.prepare_authoring_files(root)
                    )
                    rows = self.read_rows(authoring_path)
                    self.complete_rows(rows)
                    if case == "immutable":
                        rows[0]["split"] = "formal_test"
                        expected = "不可编辑"
                    elif case == "duplicate":
                        rows[1]["question"] = rows[0]["question"]
                        expected = "重复"
                    elif case == "clinical":
                        rows[0]["question"] = (
                            "患者应该服用什么方剂治疗当前症状？"
                        )
                        expected = "临床"
                    else:
                        rows[0]["gold_evidence_ids"] = json.dumps(
                            [json.loads(rows[1]["anchor_evidence_ids"])[0]]
                        )
                        expected = "anchor"
                    self.write_rows(authoring_path, rows)

                    with self.assertRaisesRegex(ValueError, expected):
                        import_formal_authoring_csv(
                            authoring_csv_path=authoring_path,
                            evidence_groups_path=groups_path,
                            evidence_path=evidence_path,
                            output_dataset_path=(
                                root / "formal-400-draft.jsonl"
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
