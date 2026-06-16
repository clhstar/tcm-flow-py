import unittest

from experiments.rag_v1_6.public_tcm_qg_answer import build_public_tcm_qg_prompt
from experiments.rag_v1_6.public_tcm_qg_runner import (
    answer_span_hit,
    freeze_public_tcm_qg_runs,
    public_tcm_qg_matrix,
)


class PublicTcmQgRunnerTests(unittest.TestCase):
    def test_matrix_contains_only_public_main_methods(self):
        self.assertEqual(
            [row["config_id"] for row in public_tcm_qg_matrix()],
            [
                "b4-public-bm25-rerank",
                "p-public-bm25-rerank",
                "p-public-no-parent",
            ],
        )

    def test_answer_span_hit_uses_context_span(self):
        self.assertTrue(
            answer_span_hit(
                answer_start=20,
                answer_end=25,
                hits=[
                    {
                        "context_start_index": 0,
                        "context_char_count": 100,
                    }
                ],
            )
        )


class PublicTcmQgAnswerTests(unittest.TestCase):
    def test_prompt_requires_public_evidence_only_answer(self):
        prompt = build_public_tcm_qg_prompt(
            question="什么类型的胆囊结石可不作治疗？",
            method="P",
            evidence=[
                {
                    "label": "E1",
                    "source_doc_id": "1240",
                    "text": "无症状胆囊结石可不作治疗。",
                }
            ],
        )

        self.assertIn("provided public document evidence", prompt)
        self.assertIn("E1", prompt)


class PublicTcmQgFreezeTests(unittest.TestCase):
    def test_runs_manifest_excludes_raw_text_and_answers(self):
        manifest = freeze_public_tcm_qg_runs(
            metrics={
                "status": "ready",
                "by_method": {"P": {"char_f1": 0.7}},
                "success_gate": {"public_tcm_qg_success": True},
            },
            output_path=None,
        )
        serialized = str(manifest)

        self.assertEqual(manifest["status"], "ready")
        self.assertNotIn("source_text", serialized)
        self.assertNotIn("reference_answer", serialized)
        self.assertNotIn("answer_text", serialized)
        self.assertNotIn("question_text", serialized)


if __name__ == "__main__":
    unittest.main()
