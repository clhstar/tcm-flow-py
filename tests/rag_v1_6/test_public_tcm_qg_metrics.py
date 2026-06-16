import unittest

from experiments.rag_v1_6.public_tcm_qg_metrics import (
    rouge_l_f1,
    success_gate,
    summarize_public_answer_rows,
)


class PublicTcmQgMetricsTests(unittest.TestCase):
    def test_rouge_l_f1_rewards_subsequence_overlap(self):
        self.assertGreater(
            rouge_l_f1(
                "应区别不同情况分别处理",
                "胆囊结石应区别不同情况分别处理",
            ),
            0.7,
        )

    def test_success_gate_requires_parent_ablation_improvement(self):
        result = success_gate(
            comparisons={
                "P-B4": {"char_f1_delta": 0.03, "char_f1_ci_low": 0.01},
                "P-P-no-parent": {
                    "char_f1_delta": 0.02,
                    "char_f1_ci_low": 0.005,
                },
            },
            by_method={
                "P": {"citation_recall": 0.90, "unsupported_answer_rate": 0.01},
                "P-no-parent": {
                    "citation_recall": 0.88,
                    "unsupported_answer_rate": 0.02,
                },
            },
        )

        self.assertTrue(result["public_tcm_qg_success"])

    def test_citation_support_allows_duplicate_public_document(self):
        result = summarize_public_answer_rows(
            questions={
                "q1": {
                    "answer": "无症状胆囊结石可不作治疗",
                    "source_doc_id": "gold-doc",
                }
            },
            answers=[
                {
                    "qa_id": "q1",
                    "source_doc_id": "gold-doc",
                    "split": "test",
                    "method": "P",
                    "repeat_index": 0,
                    "answer": "无症状胆囊结石可不作治疗",
                    "abstain": False,
                    "citations": ["E1"],
                    "retrieval_supported": True,
                    "latency_ms": 1.0,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "model_name": "proxy",
                }
            ],
            retrieval={
                "P": {
                    "q1": {
                        "evidence": [
                            {
                                "label": "E1",
                                "source_doc_id": "duplicate-doc",
                                "text": "无症状胆囊结石可不作治疗。",
                            }
                        ]
                    }
                }
            },
        )

        self.assertEqual(result["by_method"]["P"]["citation_recall"], 1.0)
        self.assertEqual(result["by_method"]["P"]["unsupported_answer_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
