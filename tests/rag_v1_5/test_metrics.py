import importlib
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from experiments.rag_v1_5.schema import PilotQuestion, RetrievalHit


def question(
    question_id: str,
    gold_clause_ids: list[str],
    *,
    answerable: bool = True,
    question_type: str = "single_clause_fact",
    book_scope: str = "shang_han_lun",
) -> PilotQuestion:
    return PilotQuestion(
        question_id=question_id,
        question=f"Question {question_id}",
        question_type=question_type,
        book_scope=book_scope,
        answerable=answerable,
        reference_answer="answer" if answerable else "无答案",
        gold_evidence_ids=(
            [f"evidence-{clause_id}" for clause_id in gold_clause_ids]
            if answerable
            else []
        ),
        gold_clause_ids=gold_clause_ids if answerable else [],
        graded_relevance=(
            {clause_id: 2 for clause_id in gold_clause_ids}
            if answerable
            else {}
        ),
        support_spans=["support"] if answerable else [],
        review_status="approved",
    )


def hit(
    chunk_id: str,
    clause_ids: list[str],
    rank: int,
    *,
    score: float = 0.8,
    strategy: str = "c4",
    parent_id: str | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        strategy=strategy,
        rank=rank,
        text=f"text:{chunk_id}",
        context_text=f"context:{chunk_id}",
        source_evidence_ids=[f"evidence:{chunk_id}"],
        clause_ids=clause_ids,
        retrieval_parent_id=parent_id,
        reranker_score=score,
    )


class MetricsTests(unittest.TestCase):
    def metrics_module(self):
        module_spec = importlib.util.find_spec(
            "experiments.rag_v1_5.metrics"
        )
        self.assertIsNotNone(module_spec, "metrics module is not implemented")
        return importlib.import_module("experiments.rag_v1_5.metrics")

    def test_hand_calculated_retrieval_metrics(self) -> None:
        metrics = self.metrics_module()
        questions = [
            question("q1", ["A"], book_scope="shang_han_lun"),
            question(
                "q2",
                ["D"],
                question_type="source_location",
                book_scope="jin_gui_yao_lue",
            ),
            question(
                "q3",
                ["F", "G"],
                question_type="multi_evidence",
                book_scope="jin_gui_yao_lue",
            ),
            question(
                "q4",
                [],
                answerable=False,
                question_type="unanswerable",
                book_scope="both",
            ),
        ]
        rankings = {
            "q1": [
                hit("q1-a", ["A"], 1, parent_id="A"),
                hit("q1-b", ["B"], 2, parent_id="B"),
                hit("q1-c", ["C"], 3, parent_id="C"),
            ],
            "q2": [
                hit("q2-b", ["B"], 1, parent_id="B"),
                hit("q2-d", ["D"], 2, parent_id="D"),
                hit("q2-e", ["E"], 3, parent_id="E"),
            ],
            "q3": [
                hit("q3-f", ["F"], 1, parent_id="F"),
                hit("q3-x", ["X"], 2, parent_id="X"),
                hit("q3-g", ["G"], 3, parent_id="G"),
            ],
            "q4": [
                hit("q4-z", ["Z"], 1, score=0.95, parent_id="Z"),
                hit("q4-y", ["Y"], 2, score=0.70, parent_id="Y"),
            ],
        }

        result = metrics.evaluate_rankings(questions, rankings)

        self.assertEqual(result["answerable_question_count"], 3)
        self.assertAlmostEqual(result["recall_at_1"], 0.5)
        self.assertAlmostEqual(result["recall_at_5"], 1.0)
        self.assertAlmostEqual(result["recall_at_10"], 1.0)
        self.assertAlmostEqual(result["hit_at_5"], 1.0)
        self.assertAlmostEqual(result["mrr_at_10"], (1 + 0.5 + 1) / 3)
        q2_ndcg = 1 / math.log2(3)
        q3_ndcg = (3 + 3 / math.log2(4)) / (
            3 + 3 / math.log2(3)
        )
        self.assertAlmostEqual(
            result["ndcg_at_10"],
            (1 + q2_ndcg + q3_ndcg) / 3,
        )
        self.assertEqual(result["c4_parent_recovery_rate"], 1.0)
        self.assertEqual(
            result["no_answer_scores"]["top1"],
            [0.95],
        )
        self.assertEqual(
            result["no_answer_scores"]["top5"],
            [0.95, 0.70],
        )
        self.assertEqual(
            result["by_question_type"]["multi_evidence"][
                "answerable_question_count"
            ],
            1,
        )
        self.assertEqual(
            result["by_book"]["jin_gui_yao_lue"][
                "answerable_question_count"
            ],
            2,
        )

    def test_hit_clause_ids_uses_clause_granularity(self) -> None:
        metrics = self.metrics_module()
        retrieval_hit = hit(
            "chunk",
            ["clause-a", "clause-b"],
            1,
            strategy="c0",
        )
        self.assertEqual(
            metrics.hit_clause_ids(retrieval_hit),
            {"clause-a", "clause-b"},
        )

    def test_summarizes_latency_and_index_size(self) -> None:
        metrics = self.metrics_module()
        latency = metrics.summarize_latency(
            [
                {
                    "bm25_ms": 1.0,
                    "dense_ms": 2.0,
                    "rrf_ms": 3.0,
                    "reranker_ms": 4.0,
                    "total_ms": 10.0,
                    "returned_context_chars": 100,
                },
                {
                    "bm25_ms": 3.0,
                    "dense_ms": 4.0,
                    "rrf_ms": 5.0,
                    "reranker_ms": 6.0,
                    "total_ms": 18.0,
                    "returned_context_chars": 200,
                },
            ]
        )
        self.assertEqual(latency["bm25_ms"]["count"], 2)
        self.assertEqual(latency["bm25_ms"]["mean"], 2.0)
        self.assertEqual(latency["bm25_ms"]["median"], 2.0)
        self.assertEqual(latency["bm25_ms"]["p95"], 3.0)
        self.assertEqual(latency["bm25_ms"]["max"], 3.0)
        self.assertEqual(
            latency["returned_context_chars"]["mean"],
            150.0,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "a.bin").write_bytes(b"123")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.bin").write_bytes(b"12345")
            self.assertEqual(metrics.index_size_bytes(root), 8)

    def test_summarizes_score_distribution_without_retaining_raw_scores(
        self,
    ) -> None:
        metrics = self.metrics_module()

        summary = metrics.summarize_score_distribution(
            [0.1, 0.4, 0.9, 0.2]
        )
        empty = metrics.summarize_score_distribution([])

        self.assertEqual(
            summary,
            {
                "count": 4,
                "min": 0.1,
                "median": 0.30000000000000004,
                "max": 0.9,
            },
        )
        self.assertEqual(
            empty,
            {
                "count": 0,
                "min": None,
                "median": None,
                "max": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
