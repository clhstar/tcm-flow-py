import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.rag_v1_5.answer_review import (
    REVIEW_FIELDS,
    build_review_sample,
    import_formal_answer_review,
)


class AnswerReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.questions = []
        answerable_types = (
            "single_clause_fact",
            "formula_composition_or_use",
            "source_location",
            "multi_evidence",
        )
        for book_scope in (
            "shang_han_lun",
            "jin_gui_yao_lue",
        ):
            for question_type in answerable_types:
                for index in range(10):
                    self.questions.append(
                        {
                            "question_id": (
                                f"{book_scope}-{question_type}-{index}"
                            ),
                            "book_scope": book_scope,
                            "question_type": question_type,
                            "question": "测试问题",
                            "reference_answer": "测试答案",
                            "answerable": True,
                        }
                    )
            for index in range(20):
                self.questions.append(
                    {
                        "question_id": (
                            f"{book_scope}-unanswerable-{index}"
                        ),
                        "book_scope": book_scope,
                        "question_type": "unanswerable",
                        "question": "无答案问题",
                        "reference_answer": "无答案",
                        "answerable": False,
                    }
                )
        self.answers = []
        for question in self.questions:
            for method in ("B0", "B4", "P", "P-no-parent"):
                self.answers.append(
                    {
                        "question_id": question["question_id"],
                        "method": method,
                        "repeat_index": 0,
                        "answer": "测试答案",
                        "evidence": [],
                    }
                )

    def test_review_sample_is_blinded_and_stratified(self):
        rows = build_review_sample(
            questions=self.questions,
            answers=self.answers,
            seed=20260616,
            answerable_count=80,
            unanswerable_count=40,
            canonical_repeat_index=0,
        )

        self.assertEqual(len(rows), 520)
        main_rows = [
            row
            for row in rows
            if row["review_track"] == "main"
        ]
        self.assertEqual(len(main_rows), 360)
        self.assertNotIn("method", rows[0])
        self.assertIn(
            main_rows[0]["blind_method"],
            {"A", "B", "C"},
        )
        self.assertEqual(
            sum(
                row["second_review_required"] == "yes"
                for row in rows
            ),
            52,
        )

    def test_parent_ablation_review_has_paired_160_rows(self):
        rows = build_review_sample(
            questions=self.questions,
            answers=self.answers,
            seed=20260616,
            answerable_count=80,
            unanswerable_count=40,
            canonical_repeat_index=0,
        )
        parent_rows = [
            row
            for row in rows
            if row["review_track"] == "parent_ablation"
        ]

        self.assertEqual(len(parent_rows), 160)

    def _write_review_csv(self, path, rows):
        with path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=REVIEW_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    def _completed_review_row(self):
        return {
            "review_id": "main-q-1-A",
            "review_track": "main",
            "question_id": "q-1",
            "book_scope": "shang_han_lun",
            "question_type": "single_clause_fact",
            "question": "测试问题",
            "reference_answer": "测试答案",
            "evidence": "[]",
            "answer": "测试答案",
            "blind_method": "A",
            "answer_correct": "yes",
            "evidence_support": "not_applicable",
            "citation_correct": "not_applicable",
            "appropriate_refusal": "not_applicable",
            "hallucination": "no",
            "review_status": "pass",
            "review_comment": "",
            "second_review_required": "yes",
            "second_review_status": "completed",
        }

    def test_import_rejects_modified_private_source_fields(self):
        source_path = self.root / "source.json"
        review_path = self.root / "review.csv"
        second_path = self.root / "second.csv"
        summary_path = self.root / "summary.json"
        source_row = self._completed_review_row()
        source_path.write_text(
            json.dumps({"rows": [source_row]}, ensure_ascii=False),
            encoding="utf-8",
        )
        modified = {**source_row, "question": "被修改的问题"}
        self._write_review_csv(review_path, [modified])
        self._write_review_csv(second_path, [source_row])

        with self.assertRaises(ValueError):
            import_formal_answer_review(
                source_snapshot_path=source_path,
                reviewed_csv_path=review_path,
                second_reviewed_csv_path=second_path,
                summary_path=summary_path,
            )

    def test_import_preserves_second_review_disagreement(self):
        source_path = self.root / "source.json"
        review_path = self.root / "review.csv"
        second_path = self.root / "second.csv"
        summary_path = self.root / "summary.json"
        source_row = self._completed_review_row()
        source_path.write_text(
            json.dumps({"rows": [source_row]}, ensure_ascii=False),
            encoding="utf-8",
        )
        second_row = {
            **source_row,
            "answer_correct": "partial",
        }
        self._write_review_csv(review_path, [source_row])
        self._write_review_csv(second_path, [second_row])

        summary = import_formal_answer_review(
            source_snapshot_path=source_path,
            reviewed_csv_path=review_path,
            second_reviewed_csv_path=second_path,
            summary_path=summary_path,
        )

        self.assertEqual(summary["status"], "needs_adjudication")
        self.assertEqual(summary["disagreement_count"], 1)

    def test_adjudication_json_preserves_reviews_and_sets_final_labels(
        self,
    ):
        source_path = self.root / "source.json"
        review_path = self.root / "review.csv"
        second_path = self.root / "second.csv"
        summary_path = self.root / "summary.json"
        source_row = self._completed_review_row()
        source_path.write_text(
            json.dumps({"rows": [source_row]}, ensure_ascii=False),
            encoding="utf-8",
        )
        final_labels = {
            "answer_correct": "partial",
            "evidence_support": "not_applicable",
            "citation_correct": "not_applicable",
            "appropriate_refusal": "not_applicable",
            "hallucination": "no",
            "review_status": "pass",
        }
        primary_row = {
            **source_row,
            "second_review_status": "adjudicated",
            "review_comment": json.dumps(final_labels),
        }
        second_row = {
            **source_row,
            "answer_correct": "partial",
        }
        self._write_review_csv(review_path, [primary_row])
        self._write_review_csv(second_path, [second_row])

        summary = import_formal_answer_review(
            source_snapshot_path=source_path,
            reviewed_csv_path=review_path,
            second_reviewed_csv_path=second_path,
            summary_path=summary_path,
        )

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(
            summary["metrics"]["answer_correct_yes_rate"],
            0.0,
        )
