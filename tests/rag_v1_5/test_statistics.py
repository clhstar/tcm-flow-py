import json
import tempfile
import unittest
from pathlib import Path


class PairedStratifiedBootstrapTests(unittest.TestCase):
    def rows(self, delta: float = 0.0):
        a = []
        b = []
        for index in range(8):
            base = float(index % 4) / 4.0
            common = {
                "question_id": f"q-{index}",
                "book_scope": "book-a" if index < 4 else "book-b",
                "question_type": (
                    "single_clause_fact"
                    if index % 2 == 0
                    else "multi_evidence"
                ),
            }
            a.append(
                {
                    **common,
                    "recall_at_5": base + delta,
                    "mrr_at_10": base + delta,
                    "ndcg_at_10": base + delta,
                }
            )
            b.append(
                {
                    **common,
                    "recall_at_5": base,
                    "mrr_at_10": base,
                    "ndcg_at_10": base,
                }
            )
        return a, b

    def test_is_deterministic_and_zero_for_identical_inputs(self) -> None:
        from experiments.rag_v1_5.statistics import (
            paired_stratified_bootstrap,
        )

        a, b = self.rows()
        first = paired_stratified_bootstrap(
            per_question_a=a,
            per_question_b=b,
            metric_fields=(
                "recall_at_5",
                "mrr_at_10",
                "ndcg_at_10",
            ),
            strata_fields=("book_scope", "question_type"),
            resamples=1000,
            seed=20260614,
            confidence_level=0.95,
        )
        second = paired_stratified_bootstrap(
            per_question_a=a,
            per_question_b=b,
            metric_fields=(
                "recall_at_5",
                "mrr_at_10",
                "ndcg_at_10",
            ),
            strata_fields=("book_scope", "question_type"),
            resamples=1000,
            seed=20260614,
            confidence_level=0.95,
        )

        self.assertEqual(first, second)
        for metric in first["metrics"].values():
            self.assertEqual(metric["delta"], 0.0)
            self.assertEqual(metric["ci_lower"], 0.0)
            self.assertEqual(metric["ci_upper"], 0.0)
            self.assertTrue(metric["inconclusive_at_95pct"])

    def test_positive_delta_and_contract_validation(self) -> None:
        from experiments.rag_v1_5.statistics import (
            paired_stratified_bootstrap,
        )

        a, b = self.rows(delta=0.1)
        result = paired_stratified_bootstrap(
            per_question_a=a,
            per_question_b=b,
            metric_fields=("recall_at_5",),
            strata_fields=("book_scope", "question_type"),
            resamples=1000,
            seed=20260614,
            confidence_level=0.95,
        )

        self.assertAlmostEqual(
            result["metrics"]["recall_at_5"]["delta"],
            0.1,
        )
        self.assertGreater(
            result["metrics"]["recall_at_5"]["ci_lower"],
            0.0,
        )
        self.assertFalse(
            result["metrics"]["recall_at_5"][
                "inconclusive_at_95pct"
            ]
        )

        b[0]["question_id"] = "other"
        with self.assertRaises(ValueError):
            paired_stratified_bootstrap(
                per_question_a=a,
                per_question_b=b,
                metric_fields=("recall_at_5",),
                strata_fields=("book_scope", "question_type"),
                resamples=1000,
                seed=20260614,
                confidence_level=0.95,
            )

    def test_summarizes_completed_formal_test_matrix(self) -> None:
        from experiments.rag_v1_5.formal_runner import (
            FORMAL_RETRIEVAL_MATRIX,
        )
        from experiments.rag_v1_5.statistics import (
            summarize_formal_test,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matrix = [
                {
                    "config_id": row.config_id,
                    "paper_role": row.paper_role,
                    "strategy": row.strategy,
                    "mode": row.mode,
                    "context_policy": row.context_policy,
                    "metadata_policy": row.metadata_policy,
                }
                for row in FORMAL_RETRIEVAL_MATRIX
            ]
            (root / "matrix-config.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "split": "formal_test",
                        "matrix": matrix,
                    }
                ),
                encoding="utf-8",
            )
            (root / "matrix-summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "split": "formal_test",
                        "config_count": 14,
                        "completed_config_count": 14,
                        "failed_config_count": 0,
                        "configs": [
                            {
                                "config_id": row.config_id,
                                "latency": {
                                    "returned_context_chars": {
                                        "mean": 50.0,
                                        "median": 48.0,
                                        "p95": 60.0,
                                    }
                                },
                                "index_size_bytes": 100,
                            }
                            for row in FORMAL_RETRIEVAL_MATRIX
                        ],
                    }
                ),
                encoding="utf-8",
            )
            for config_index, row in enumerate(
                FORMAL_RETRIEVAL_MATRIX
            ):
                config_dir = root / row.config_id
                config_dir.mkdir()
                records = []
                for index in range(8):
                    records.append(
                        {
                            "question_id": f"q-{index}",
                            "answerable": True,
                            "book_scope": (
                                "book-a" if index < 4 else "book-b"
                            ),
                            "question_type": (
                                "single_clause_fact"
                                if index % 2 == 0
                                else "multi_evidence"
                            ),
                            "recall_at_5": (
                                0.5 + config_index / 100.0
                            ),
                            "mrr_at_10": (
                                0.4 + config_index / 100.0
                            ),
                            "ndcg_at_10": (
                                0.45 + config_index / 100.0
                            ),
                        }
                    )
                (config_dir / "per-question.jsonl").write_text(
                    "".join(
                        json.dumps(record) + "\n"
                        for record in records
                    ),
                    encoding="utf-8",
                )
                (config_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "by_book": {},
                            "by_question_type": {},
                            "no_answer_score_distribution": {},
                        }
                    ),
                    encoding="utf-8",
                )
            prereg_path = root / "prereg.json"
            prereg_path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "statistics": {
                            "bootstrap_seed": 20260614,
                            "bootstrap_resamples": 100,
                            "confidence_level": 0.95,
                            "strata": [
                                "book_scope",
                                "question_type",
                            ],
                            "primary_metrics": [
                                "recall_at_5",
                                "mrr_at_10",
                                "ndcg_at_10",
                            ],
                        },
                        "comparisons": {
                            "primary": {
                                "a": "p-c4-hybrid-rerank",
                                "b": "b4-c0-hybrid-rerank",
                            },
                            "ablations": [
                                {
                                    "a": "p-c4-hybrid-rerank",
                                    "b": config_id,
                                }
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
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "formal-statistics.json"
            summary = summarize_formal_test(
                run_dir=root,
                prereg_manifest_path=prereg_path,
                output_path=output_path,
            )

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(len(summary["absolute"]), 14)
        self.assertEqual(len(summary["paired_comparisons"]), 7)
        self.assertEqual(
            summary["details"]["p-c4-hybrid-rerank"][
                "returned_context_chars"
            ]["mean"],
            50.0,
        )


if __name__ == "__main__":
    unittest.main()
