import unittest

from experiments.rag_v1_5.answer_metrics import (
    char_f1,
    citation_metrics,
    paired_bootstrap,
    refusal_metrics,
    summarize_answer_rows,
)


class AnswerMetricsTests(unittest.TestCase):
    def test_char_f1_normalizes_chinese_punctuation(self):
        self.assertEqual(
            char_f1("桂枝、芍药", "桂枝芍药"),
            1.0,
        )

    def test_citation_metrics_use_gold_clause_coverage(self):
        result = citation_metrics(
            citations=["E1", "E2"],
            evidence={
                "E1": {"clause_ids": ["gold-1"]},
                "E2": {"clause_ids": ["other"]},
            },
            gold_clause_ids=["gold-1"],
        )
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)

    def test_refusal_metrics_distinguish_correct_refusal(self):
        result = refusal_metrics(
            answerable=False,
            abstain=True,
        )
        self.assertEqual(result["correct"], 1)

    def test_summary_keeps_unsupported_answers_separate_from_f1(self):
        questions = {
            "q-1": {
                "answerable": True,
                "reference_answer": "桂枝芍药",
                "gold_clause_ids": ["gold-1"],
            },
            "q-2": {
                "answerable": False,
                "reference_answer": "无答案",
                "gold_clause_ids": [],
            },
        }
        answers = []
        for method in ("B0", "B4", "P", "P-no-parent"):
            for repeat_index in range(3):
                answers.append(
                    {
                        "question_id": "q-1",
                        "method": method,
                        "repeat_index": repeat_index,
                        "answer": "桂枝、芍药",
                        "abstain": False,
                        "citations": (
                            [] if method == "B0" else ["E1"]
                        ),
                        "latency_ms": 10.0,
                        "input_tokens": 5,
                        "output_tokens": 2,
                    }
                )
                answers.append(
                    {
                        "question_id": "q-2",
                        "method": method,
                        "repeat_index": repeat_index,
                        "answer": (
                            "错误回答"
                            if method == "B0"
                            else "在指定古籍证据范围内未找到可靠答案。"
                        ),
                        "abstain": method != "B0",
                        "citations": [],
                        "latency_ms": 10.0,
                        "input_tokens": 5,
                        "output_tokens": 2,
                    }
                )
        retrieval = {
            method: {
                "q-1": {
                    "evidence": [
                        {
                            "label": "E1",
                            "clause_ids": ["gold-1"],
                        }
                    ]
                },
                "q-2": {"evidence": []},
            }
            for method in ("B4", "P", "P-no-parent")
        }

        result = summarize_answer_rows(
            questions=questions,
            answers=answers,
            retrieval=retrieval,
        )

        self.assertEqual(result["by_method"]["B4"]["char_f1"], 1.0)
        self.assertEqual(
            result["by_method"]["P"]["citation_recall"],
            1.0,
        )
        self.assertEqual(
            result["by_method"]["B0"]["unsupported_answer_rate"],
            1.0,
        )
        self.assertEqual(
            result["by_method"]["P"]["answer_stability"],
            1.0,
        )

    def test_paired_bootstrap_resamples_question_ids(self):
        rows = []
        for question_id, a_value, b_value in (
            ("q-1", 1.0, 0.0),
            ("q-2", 0.5, 0.5),
        ):
            for repeat_index in range(3):
                rows.extend(
                    [
                        {
                            "question_id": question_id,
                            "method": "A",
                            "repeat_index": repeat_index,
                            "score": a_value,
                        },
                        {
                            "question_id": question_id,
                            "method": "B",
                            "repeat_index": repeat_index,
                            "score": b_value,
                        },
                    ]
                )

        result = paired_bootstrap(
            rows=rows,
            method_a="A",
            method_b="B",
            metric="score",
            seed=20260616,
            resamples=100,
            confidence_level=0.95,
        )

        self.assertEqual(result["question_count"], 2)
        self.assertEqual(result["delta"], 0.5)
