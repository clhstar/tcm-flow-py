import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.rag_v1_6.public_tcm_qg_formal_review import (
    import_public_tcm_qg_formal_answer_review,
    prepare_public_tcm_qg_formal_answer_review,
    review_csv_columns,
)
from experiments.rag_v1_6.cli import build_parser


class PublicTcmQgFormalReviewTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _dataset_rows(self) -> list[dict]:
        rows = []
        for index in range(1, 4):
            answer = f"答案{index}"
            source_text = f"证据文本包含{answer}。"
            rows.append(
                {
                    "qa_id": f"q-{index}",
                    "source_doc_id": f"doc-{index}",
                    "split": "test",
                    "question": f"问题{index}？",
                    "answer": answer,
                    "source_text": source_text,
                    "answer_start": 6,
                    "answer_end": 6 + len(answer),
                    "review_status": "approved",
                    "question_version": 1,
                }
            )
        return rows

    def _answer_rows(self, qa_ids: list[str]) -> list[dict]:
        rows = []
        for qa_id in qa_ids:
            doc_id = qa_id.replace("q", "doc")
            for method in ("B0", "B4", "P", "P-no-parent"):
                rows.append(
                    {
                        "qa_id": qa_id,
                        "source_doc_id": doc_id,
                        "split": "test",
                        "method": method,
                        "repeat_index": 0,
                        "answer": f"{method}回答",
                        "abstain": False,
                        "citations": [] if method == "B0" else ["E1"],
                        "retrieval_supported": method != "B0",
                        "latency_ms": 1.0,
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "model_name": "fake",
                    }
                )
        return rows

    def _retrieval_row(self, qa_id: str, config_id: str) -> dict:
        doc_id = qa_id.replace("q", "doc")
        return {
            "qa_id": qa_id,
            "source_doc_id": doc_id,
            "split": "test",
            "config_id": config_id,
            "method_role": config_id,
            "doc_recall_at_5": 1.0,
            "doc_mrr_at_10": 1.0,
            "answer_span_hit_at_5": 1.0,
            "answer_span_coverage_at_5": 1.0,
            "top5_traceability_ok": True,
            "latency": 0.01,
            "hits": [
                {
                    "chunk_id": (
                        f"p-shared-child-{qa_id}"
                        if config_id.startswith("p-")
                        else f"{config_id}-{qa_id}"
                    ),
                    "strategy": "child" if config_id.startswith("p-") else "b4",
                    "rank": 1,
                    "source_doc_id": doc_id,
                    "parent_id": f"{doc_id}-parent",
                    "text": f"证据文本包含答案{qa_id[-1]}。",
                    "context_text": f"证据文本包含答案{qa_id[-1]}。",
                    "start_index": 0,
                    "char_count": 10,
                    "context_start_index": 0,
                    "context_char_count": 10,
                    "bm25_score": 1.0,
                    "dense_score": 1.0,
                    "rrf_score": 1.0,
                    "reranker_score": 1.0,
                    "score": 1.0,
                }
            ],
        }

    def test_review_csv_columns_include_blind_fields(self):
        columns = review_csv_columns()

        for column in (
            "review_id",
            "blind_method_id",
            "question",
            "answer",
            "citations",
            "evidence",
            "answer_correct",
            "evidence_supported",
            "citation_correct",
            "hallucination",
            "answer_completeness",
            "clinical_safety_issue",
            "reviewer_comment",
        ):
            self.assertIn(column, columns)

    def test_cli_parses_formal_answer_review_and_summary_commands(self):
        parser = build_parser()

        for command in (
            "summarize-public-tcm-qg-formal-answer-test",
            "prepare-public-tcm-qg-formal-answer-review",
            "import-public-tcm-qg-formal-answer-review",
            "freeze-public-tcm-qg-formal-answer-runs",
        ):
            args = parser.parse_args([command])
            self.assertEqual(args.command, command)

        import_args = parser.parse_args(["import-public-tcm-qg-formal-answer-review"])
        self.assertTrue(str(import_args.parent_ablation_reviewed_csv).endswith(
            "formal-answer-review-parent-ablation.csv"
        ))

    def test_prepare_review_package_writes_blinded_rows_and_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.jsonl"
            retrieval = root / "retrieval"
            run = root / "answer-run"
            output = root / "review"
            qa_ids = ["q-1", "q-2", "q-3"]
            self._write_jsonl(dataset, self._dataset_rows())
            run.mkdir()
            (run / "matrix-summary.json").write_text(
                json.dumps({"status": "completed", "split": "test"}),
                encoding="utf-8",
            )
            self._write_jsonl(run / "per-answer.jsonl", self._answer_rows(qa_ids))
            (retrieval / "matrix-config.json").parent.mkdir(parents=True, exist_ok=True)
            (retrieval / "matrix-config.json").write_text(
                json.dumps({"status": "completed", "split": "test"}),
                encoding="utf-8",
            )
            (retrieval / "matrix-summary.json").write_text(
                json.dumps({"status": "completed", "split": "test"}),
                encoding="utf-8",
            )
            for config_id in (
                "b4-public-hybrid-rerank",
                "p-public-hybrid-rerank",
                "p-public-no-parent",
            ):
                self._write_jsonl(
                    retrieval / config_id / "per-question.jsonl",
                    [self._retrieval_row(qa_id, config_id) for qa_id in qa_ids],
                )

            manifest = prepare_public_tcm_qg_formal_answer_review(
                answer_run_dir=run,
                dataset_path=dataset,
                retrieval_matrix_dir=retrieval,
                output_dir=output,
                main_review_questions=2,
                second_review_rate=0.5,
                parent_ablation_focus_questions=1,
                seed=20260616,
            )

            self.assertEqual(manifest["main_review_rows"], 8)
            self.assertEqual(manifest["second_review_rows"], 4)
            self.assertEqual(manifest["parent_ablation_rows"], 2)
            self.assertTrue(manifest["blind_key_written"])
            with (output / "formal-answer-review-main.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertIn("blind_method_id", rows[0])
            self.assertNotIn("method", rows[0])
            self.assertTrue((output / "formal-answer-review-blind-key.csv").is_file())

    def test_import_review_summary_records_second_review_disagreement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            review_path = root / "review.csv"
            second_path = root / "second.csv"
            parent_path = root / "parent.csv"
            output_path = root / "review-summary.json"
            row = {
                "review_id": "main-q-1-A",
                "blind_method_id": "A",
                "question": "问题？",
                "answer": "回答",
                "citations": "E1",
                "evidence": "证据",
                "answer_correct": "yes",
                "evidence_supported": "yes",
                "citation_correct": "yes",
                "hallucination": "no",
                "answer_completeness": "complete",
                "clinical_safety_issue": "no",
                "reviewer_comment": "",
            }
            disagreement = {**row, "answer_correct": "no"}
            for path, rows in (
                (review_path, [row]),
                (second_path, [disagreement]),
                (parent_path, [row]),
            ):
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=review_csv_columns())
                    writer.writeheader()
                    writer.writerows(rows)

            summary = import_public_tcm_qg_formal_answer_review(
                reviewed_csv_path=review_path,
                second_reviewed_csv_path=second_path,
                parent_ablation_reviewed_csv_path=parent_path,
                output_path=output_path,
            )

            self.assertEqual(summary["status"], "ready")
            self.assertTrue(summary["answer_review_completed"])
            self.assertEqual(summary["reviewed_count"], 1)
            self.assertEqual(summary["second_review_count"], 1)
            self.assertEqual(summary["parent_ablation_reviewed_count"], 1)
            self.assertEqual(summary["disagreement_count"], 1)
            self.assertTrue(output_path.is_file())

    def test_import_review_accepts_gb18030_csv_saved_by_spreadsheet(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            review_path = root / "review.csv"
            second_path = root / "second.csv"
            output_path = root / "review-summary.json"
            row = {
                "review_id": "main-q-1-A",
                "blind_method_id": "A",
                "question": "问题？",
                "answer": "回答",
                "citations": "E1",
                "evidence": "证据",
                "answer_correct": "yes",
                "evidence_supported": "yes",
                "citation_correct": "yes",
                "hallucination": "no",
                "answer_completeness": "complete",
                "clinical_safety_issue": "no",
                "reviewer_comment": "可接受",
            }
            for path in (review_path, second_path):
                with path.open("w", encoding="gb18030", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=review_csv_columns())
                    writer.writeheader()
                    writer.writerow(row)

            summary = import_public_tcm_qg_formal_answer_review(
                reviewed_csv_path=review_path,
                second_reviewed_csv_path=second_path,
                output_path=output_path,
            )

            self.assertEqual(summary["status"], "ready")
            self.assertEqual(summary["reviewed_count"], 1)

    def test_import_review_normalizes_numeric_review_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            review_path = root / "review.csv"
            second_path = root / "second.csv"
            output_path = root / "review-summary.json"
            row = {
                "review_id": "main-q-1-A",
                "blind_method_id": "A",
                "question": "问题？",
                "answer": "回答",
                "citations": "E1",
                "evidence": "证据",
                "answer_correct": "1",
                "evidence_supported": "1",
                "citation_correct": "0",
                "hallucination": "0",
                "answer_completeness": "3",
                "clinical_safety_issue": "0",
                "reviewer_comment": "",
            }
            for path in (review_path, second_path):
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=review_csv_columns())
                    writer.writeheader()
                    writer.writerow(row)

            summary = import_public_tcm_qg_formal_answer_review(
                reviewed_csv_path=review_path,
                second_reviewed_csv_path=second_path,
                output_path=output_path,
            )

            self.assertEqual(summary["metrics"]["answer_correct_rate"], 1.0)
            self.assertEqual(summary["metrics"]["citation_correct_rate"], 0.0)
            self.assertEqual(summary["metrics"]["hallucination_rate"], 0.0)
            self.assertEqual(
                summary["metrics"]["answer_completeness_distribution"],
                {
                    "score_3": 1,
                    "score_2": 0,
                    "score_1": 0,
                    "score_0": 0,
                },
            )


if __name__ == "__main__":
    unittest.main()
