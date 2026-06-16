import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.rag_v1_6.public_tcm_qg_formal_metrics import (
    classify_success_gate,
    freeze_public_tcm_qg_formal_answer_runs,
    summarize_public_tcm_qg_formal_answer_test,
)


class PublicTcmQgFormalMetricsTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _dataset_rows(self) -> list[dict]:
        source_text = "无症状胆囊结石可不作治疗。"
        return [
            {
                "qa_id": "q-1",
                "source_doc_id": "doc-1",
                "split": "test",
                "question": "什么可不作治疗？",
                "answer": "无症状胆囊结石",
                "source_text": source_text,
                "answer_start": 0,
                "answer_end": 7,
                "review_status": "approved",
                "question_version": 1,
            }
        ]

    def _retrieval_row(self, *, config_id: str, contains_answer: bool) -> dict:
        context = "无症状胆囊结石可不作治疗。" if contains_answer else "有症状者需要治疗。"
        strategy = "child" if config_id.startswith("p-") else "b4"
        return {
            "qa_id": "q-1",
            "source_doc_id": "doc-1",
            "split": "test",
            "config_id": config_id,
            "method_role": config_id,
            "doc_recall_at_5": 1.0,
            "doc_mrr_at_10": 1.0,
            "answer_span_hit_at_5": 1.0 if contains_answer else 0.0,
            "answer_span_coverage_at_5": 1.0 if contains_answer else 0.0,
            "top5_traceability_ok": True,
            "latency": 0.01,
            "hits": [
                {
                    "chunk_id": (
                        "p-shared-child-q-1"
                        if config_id.startswith("p-")
                        else f"{config_id}-chunk"
                    ),
                    "strategy": strategy,
                    "rank": 1,
                    "source_doc_id": "doc-1",
                    "parent_id": "doc-1-parent",
                    "text": context,
                    "context_text": context,
                    "start_index": 0,
                    "char_count": len(context),
                    "context_start_index": 0,
                    "context_char_count": len(context),
                    "bm25_score": 1.0,
                    "dense_score": 1.0,
                    "rrf_score": 1.0,
                    "reranker_score": 1.0,
                    "score": 1.0,
                }
            ],
        }

    def test_success_gate_classifies_three_outcomes(self):
        strong = classify_success_gate(
            {
                "P-B4": {
                    "char_f1_delta": 0.01,
                    "char_f1_ci_lower": 0.001,
                    "citation_recall_delta": 0.0,
                    "unsupported_answer_rate_delta": -0.01,
                },
                "P-P-no-parent": {
                    "char_f1_delta": 0.02,
                    "char_f1_ci_lower": 0.01,
                    "citation_recall_delta": 0.03,
                    "citation_recall_ci_lower": 0.01,
                },
            }
        )
        parent_only = classify_success_gate(
            {
                "P-B4": {
                    "char_f1_delta": -0.01,
                    "char_f1_ci_lower": -0.02,
                    "citation_recall_delta": -0.01,
                    "unsupported_answer_rate_delta": 0.0,
                },
                "P-P-no-parent": {
                    "char_f1_delta": 0.02,
                    "char_f1_ci_lower": 0.01,
                    "citation_recall_delta": 0.03,
                    "citation_recall_ci_lower": 0.01,
                },
            }
        )
        failed = classify_success_gate(
            {
                "P-B4": {
                    "char_f1_delta": -0.01,
                    "char_f1_ci_lower": -0.02,
                    "citation_recall_delta": -0.01,
                    "unsupported_answer_rate_delta": 0.0,
                },
                "P-P-no-parent": {
                    "char_f1_delta": 0.0,
                    "char_f1_ci_lower": -0.01,
                    "citation_recall_delta": 0.0,
                    "citation_recall_ci_lower": -0.01,
                },
            }
        )

        self.assertEqual(strong["success_gate"], "strong_success")
        self.assertEqual(parent_only["success_gate"], "parent_ablation_only")
        self.assertEqual(failed["success_gate"], "failed")

    def test_summarize_formal_answer_test_writes_metrics_and_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.jsonl"
            retrieval = root / "retrieval"
            run = root / "answer-run"
            config = root / "formal.yaml"
            self._write_jsonl(dataset, self._dataset_rows())
            config.write_text(
                yaml.safe_dump(
                    {
                        "retrieval": {"answer_context_top_k": 5},
                        "statistics": {
                            "bootstrap_seed": 20260616,
                            "bootstrap_resamples": 50,
                            "confidence_level": 0.95,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            retrieval.mkdir(parents=True, exist_ok=True)
            (retrieval / "matrix-config.json").write_text(
                json.dumps({"status": "completed", "split": "test"}),
                encoding="utf-8",
            )
            (retrieval / "matrix-summary.json").write_text(
                json.dumps({"status": "completed", "split": "test"}),
                encoding="utf-8",
            )
            for config_id, contains_answer in (
                ("b4-public-hybrid-rerank", True),
                ("p-public-hybrid-rerank", True),
                ("p-public-no-parent", False),
            ):
                self._write_jsonl(
                    retrieval / config_id / "per-question.jsonl",
                    [self._retrieval_row(config_id=config_id, contains_answer=contains_answer)],
                )
            run.mkdir()
            (run / "matrix-summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "split": "test",
                        "expected_runs": 4,
                        "completed_count": 4,
                        "error_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            self._write_jsonl(
                run / "per-answer.jsonl",
                [
                    {
                        "qa_id": "q-1",
                        "source_doc_id": "doc-1",
                        "split": "test",
                        "method": "B0",
                        "repeat_index": 0,
                        "answer": "不知道",
                        "abstain": False,
                        "citations": [],
                        "retrieval_supported": False,
                        "latency_ms": 1.0,
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "model_name": "fake",
                    },
                    {
                        "qa_id": "q-1",
                        "source_doc_id": "doc-1",
                        "split": "test",
                        "method": "B4",
                        "repeat_index": 0,
                        "answer": "无症状胆囊结石",
                        "abstain": False,
                        "citations": ["E1"],
                        "retrieval_supported": True,
                        "latency_ms": 1.0,
                        "input_tokens": 10,
                        "output_tokens": 7,
                        "model_name": "fake",
                    },
                    {
                        "qa_id": "q-1",
                        "source_doc_id": "doc-1",
                        "split": "test",
                        "method": "P",
                        "repeat_index": 0,
                        "answer": "无症状胆囊结石",
                        "abstain": False,
                        "citations": ["E1"],
                        "retrieval_supported": True,
                        "latency_ms": 1.0,
                        "input_tokens": 10,
                        "output_tokens": 7,
                        "model_name": "fake",
                    },
                    {
                        "qa_id": "q-1",
                        "source_doc_id": "doc-1",
                        "split": "test",
                        "method": "P-no-parent",
                        "repeat_index": 0,
                        "answer": "需要治疗",
                        "abstain": False,
                        "citations": ["E1"],
                        "retrieval_supported": True,
                        "latency_ms": 1.0,
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "model_name": "fake",
                    },
                ],
            )

            summary = summarize_public_tcm_qg_formal_answer_test(
                run_dir=run,
                dataset_path=dataset,
                retrieval_matrix_dir=retrieval,
                config_path=config,
            )

            self.assertEqual(summary["status"], "ready")
            self.assertEqual(summary["success_gate"]["success_gate"], "parent_ablation_only")
            self.assertTrue((run / "automatic-metrics.json").is_file())
            self.assertTrue((run / "paired-bootstrap.json").is_file())
            self.assertTrue((run / "success-gate.json").is_file())
            self.assertTrue((run / "per-question-metrics.jsonl").is_file())
            per_question = (run / "per-question-metrics.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("answer_text", per_question)
            self.assertNotIn("question_text", per_question)

    def test_freeze_answer_runs_excludes_review_comment_key_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run = root / "answer-run"
            run.mkdir()
            for filename, payload in (
                ("matrix-summary.json", {"status": "completed", "split": "test"}),
                ("per-answer.jsonl", None),
                ("automatic-metrics.json", {"status": "ready", "by_method": {}}),
                ("paired-bootstrap.json", {"status": "ready", "comparisons": []}),
                ("success-gate.json", {"status": "ready", "success_gate": "failed"}),
            ):
                path = run / filename
                if payload is None:
                    path.write_text('{"qa_id":"q-1"}\n', encoding="utf-8")
                else:
                    path.write_text(json.dumps(payload), encoding="utf-8")
            review = root / "review-summary.json"
            review.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "answer_review_completed": True,
                        "reviewed_count": 1,
                        "second_review_count": 1,
                        "disagreement_count": 0,
                        "metrics": {},
                    }
                ),
                encoding="utf-8",
            )
            prereg = root / "answer-prereg.json"
            prereg.write_text("{}", encoding="utf-8")
            retrieval = root / "retrieval-runs.json"
            retrieval.write_text("{}", encoding="utf-8")
            output = root / "answer-runs.json"

            manifest = freeze_public_tcm_qg_formal_answer_runs(
                answer_run_dir=run,
                review_summary_path=review,
                output_path=output,
                answer_prereg_path=prereg,
                retrieval_runs_manifest_path=retrieval,
            )

            serialized = json.dumps(manifest, ensure_ascii=False)
            self.assertEqual(manifest["status"], "ready")
            self.assertNotIn("reviewer_comment", serialized)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
