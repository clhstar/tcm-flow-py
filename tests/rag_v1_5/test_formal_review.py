import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tests.rag_v1_5.test_formal_dataset import (
    make_formal_artifacts,
    write_jsonl,
)


class FormalReviewWorkflowTests(unittest.TestCase):
    def prepare_files(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path, list[dict]]:
        from experiments.rag_v1_5.formal_review import (
            prepare_formal_review,
        )

        questions, groups, _ = make_formal_artifacts()
        for question in questions:
            question["review_status"] = "draft"
        draft_path = root / "formal-400-draft.jsonl"
        groups_path = root / "formal-evidence-groups.jsonl"
        review_path = root / "formal-review.csv"
        write_jsonl(draft_path, questions)
        write_jsonl(groups_path, groups)
        prepare_formal_review(
            draft_dataset_path=draft_path,
            evidence_groups_path=groups_path,
            review_csv_path=review_path,
            second_review_seed=20260614,
        )
        return draft_path, groups_path, review_path, questions

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
            row["first_comment"] = "首审通过"
            row["first_reviewer"] = "测试审核者"
            row["first_reviewed_at"] = "2026-06-15"
            if row["second_review_required"] == "true":
                row["second_status"] = "pass"
                row["second_decision"] = "correct"
                row["second_comment"] = "复核通过"
                row["second_reviewer"] = "测试复核者"
                row["second_reviewed_at"] = "2026-06-15"

    def test_prepare_review_writes_bom_and_stratifies_forty_rows(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_review import (
            prepare_formal_review,
        )

        questions, groups, _ = make_formal_artifacts()
        for question in questions:
            question["review_status"] = "draft"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            draft_path = root / "formal-400-draft.jsonl"
            groups_path = root / "formal-evidence-groups.jsonl"
            first_path = root / "formal-review-first.csv"
            second_path = root / "formal-review-second.csv"
            write_jsonl(draft_path, questions)
            write_jsonl(groups_path, groups)
            first = prepare_formal_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                review_csv_path=first_path,
                second_review_seed=20260614,
            )
            second = prepare_formal_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                review_csv_path=second_path,
                second_review_seed=20260614,
            )
            rows = self.read_rows(first_path)

            self.assertTrue(
                first_path.read_bytes().startswith(b"\xef\xbb\xbf")
            )
            self.assertEqual(
                first_path.read_bytes(),
                second_path.read_bytes(),
            )

        required = [
            row
            for row in rows
            if row["second_review_required"] == "true"
        ]
        strata = Counter(
            (
                row["book_scope"],
                row["split"],
                row["question_type"],
            )
            for row in required
        )
        self.assertEqual(len(rows), 400)
        self.assertEqual(len(required), 40)
        self.assertEqual(len(strata), 20)
        self.assertEqual(set(strata.values()), {2})
        self.assertEqual(first["second_review_required_count"], 40)
        self.assertEqual(
            first["second_review_question_ids"],
            second["second_review_question_ids"],
        )

    def test_prepare_review_inherits_only_unchanged_content(self) -> None:
        from experiments.rag_v1_5.formal_review import (
            prepare_formal_review,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            draft_path, groups_path, review_path, questions = (
                self.prepare_files(root)
            )
            rows = self.read_rows(review_path)
            rows[0]["first_status"] = "pass"
            rows[0]["first_decision"] = "correct"
            rows[0]["first_comment"] = "保留审核"
            rows[0]["first_reviewer"] = "测试审核者"
            rows[0]["first_reviewed_at"] = "2026-06-15"
            old_hash = rows[0]["content_sha256"]
            self.write_rows(review_path, rows)

            prepare_formal_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                review_csv_path=review_path,
            )
            inherited = self.read_rows(review_path)
            self.assertEqual(inherited[0]["first_status"], "pass")

            questions[0]["question"] += " 修订"
            questions[0]["question_version"] += 1
            write_jsonl(draft_path, questions)
            prepare_formal_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                review_csv_path=review_path,
            )
            reset = self.read_rows(review_path)

        self.assertNotEqual(reset[0]["content_sha256"], old_hash)
        self.assertEqual(reset[0]["first_status"], "pending")
        self.assertEqual(reset[0]["first_reviewer"], "")

    def test_import_review_blocks_pending_and_approves_complete_rounds(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_review import (
            import_formal_review,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            draft_path, groups_path, review_path, _ = (
                self.prepare_files(root)
            )
            output_path = root / "formal-400.jsonl"
            summary_path = root / "formal-review-summary.json"

            pending = import_formal_review(
                draft_dataset_path=draft_path,
                evidence_groups_path=groups_path,
                reviewed_csv_path=review_path,
                output_dataset_path=output_path,
                summary_path=summary_path,
            )
            self.assertEqual(pending["status"], "blocked")
            self.assertEqual(
                pending["first_review_pending_count"],
                400,
            )
            self.assertEqual(
                pending["second_review_pending_count"],
                40,
            )
            self.assertFalse(output_path.exists())

            rows = self.read_rows(review_path)
            self.approve_rows(rows)
            self.write_rows(review_path, rows)
            ready = import_formal_review(
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

        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["first_review_pass_count"], 400)
        self.assertEqual(ready["second_review_pass_count"], 40)
        self.assertEqual(len(approved), 400)
        self.assertTrue(
            all(
                record["review_status"] == "approved"
                for record in approved
            )
        )

    def test_import_review_normalizes_legacy_encodings(self) -> None:
        from experiments.rag_v1_5.formal_review import (
            import_formal_review,
        )

        cases = (
            ("cp936", "中文审核"),
            ("gb18030", "中文审核𠮷"),
        )
        for encoding, comment in cases:
            with self.subTest(encoding=encoding):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    draft_path, groups_path, review_path, _ = (
                        self.prepare_files(root)
                    )
                    rows = self.read_rows(review_path)
                    self.approve_rows(rows)
                    rows[0]["first_comment"] = comment
                    self.write_rows(
                        review_path,
                        rows,
                        encoding=encoding,
                    )
                    summary = import_formal_review(
                        draft_dataset_path=draft_path,
                        evidence_groups_path=groups_path,
                        reviewed_csv_path=review_path,
                        output_dataset_path=root / "formal-400.jsonl",
                        summary_path=root / "summary.json",
                    )
                    self.assertEqual(
                        summary["encoding"]["detected_encoding"],
                        encoding,
                    )
                    self.assertTrue(
                        summary["encoding"]["converted"]
                    )
                    self.assertTrue(
                        summary["encoding"]["unicode_equivalent"]
                    )
                    self.assertTrue(
                        review_path.read_bytes().startswith(
                            b"\xef\xbb\xbf"
                        )
                    )

    def test_import_review_rejects_immutable_edits_and_duplicate_ids(
        self,
    ) -> None:
        from experiments.rag_v1_5.formal_review import (
            import_formal_review,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            draft_path, groups_path, review_path, _ = (
                self.prepare_files(root)
            )
            rows = self.read_rows(review_path)
            self.approve_rows(rows)
            rows[0]["question"] += " 非法修改"
            self.write_rows(review_path, rows)
            with self.assertRaisesRegex(ValueError, "不允许修改"):
                import_formal_review(
                    draft_dataset_path=draft_path,
                    evidence_groups_path=groups_path,
                    reviewed_csv_path=review_path,
                    output_dataset_path=root / "formal-400.jsonl",
                    summary_path=root / "summary.json",
                )

            rows = self.read_rows(review_path)
            rows[0]["question"] = rows[1]["question"]
            rows[0]["question_id"] = rows[1]["question_id"]
            self.write_rows(review_path, rows)
            with self.assertRaisesRegex(ValueError, "重复 question_id"):
                import_formal_review(
                    draft_dataset_path=draft_path,
                    evidence_groups_path=groups_path,
                    reviewed_csv_path=review_path,
                    output_dataset_path=root / "formal-400.jsonl",
                    summary_path=root / "summary.json",
                )


class FormalReviewCliTests(unittest.TestCase):
    def test_formal_review_cli_contract(self) -> None:
        from experiments.rag_v1_5.cli import build_parser

        prepare_args = build_parser().parse_args(
            ["prepare-formal-review"]
        )
        import_args = build_parser().parse_args(
            ["import-formal-review"]
        )

        self.assertEqual(
            prepare_args.review_csv,
            Path(
                "data/rag_v1_5/formal/evaluation/"
                "formal-review.csv"
            ),
        )
        self.assertEqual(
            prepare_args.second_review_seed,
            20260614,
        )
        self.assertEqual(
            import_args.output,
            Path(
                "data/rag_v1_5/formal/evaluation/"
                "formal-400.jsonl"
            ),
        )
        self.assertEqual(
            import_args.summary,
            Path(
                "data/rag_v1_5/formal/evaluation/"
                "formal-review-summary.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
