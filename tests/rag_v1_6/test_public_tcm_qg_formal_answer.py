import unittest
import tempfile
from pathlib import Path

from experiments.rag_v1_6.common import atomic_write_json, write_jsonl
from experiments.rag_v1_6.public_tcm_qg_formal_answer import (
    build_public_tcm_qg_formal_prompt,
    estimate_public_tcm_qg_formal_answer_cost,
    freeze_public_tcm_qg_formal_answer_dev,
    freeze_public_tcm_qg_formal_answer_prereg,
    parse_formal_answer_json,
    run_public_tcm_qg_formal_answer_matrix,
)


class PublicTcmQgFormalAnswerTests(unittest.TestCase):
    def test_parse_formal_answer_json_accepts_code_fence(self):
        parsed = parse_formal_answer_json(
            '```json\n{"answer":"可不作治疗","abstain":false,"citations":["E1"]}\n```',
            method="P",
            evidence_labels={"E1"},
        )

        self.assertEqual(parsed["answer"], "可不作治疗")
        self.assertFalse(parsed["abstain"])
        self.assertEqual(parsed["citations"], ["E1"])

    def test_parse_formal_answer_json_extracts_object_from_preface(self):
        parsed = parse_formal_answer_json(
            '结果如下：{"answer":"可不作治疗","abstain":false,"citations":["E1"]}',
            method="P",
            evidence_labels={"E1"},
        )

        self.assertEqual(parsed["answer"], "可不作治疗")

    def test_cost_estimate_counts_questions_methods_and_repeats(self):
        estimate = estimate_public_tcm_qg_formal_answer_cost(
            question_count=2,
            methods=["B0", "B4", "P", "P-no-parent"],
            repeats=1,
            prompt_token_estimate_per_call=100,
            completion_token_estimate_per_call=32,
            model_name="formal-model",
            base_url_origin="https://api.example.com",
        )

        self.assertEqual(estimate["expected_calls"], 8)
        self.assertGreater(estimate["estimated_prompt_tokens"], 0)
        self.assertGreater(estimate["estimated_completion_tokens"], 0)
        self.assertIn("formal-model", estimate["estimated_cost_by_model"])

    def test_prompt_contract_for_evidence_methods_and_b0(self):
        prompt = build_public_tcm_qg_formal_prompt(
            question="什么类型的胆囊结石可不作治疗？",
            method="P",
            evidence=[
                {
                    "label": "E1",
                    "source_doc_id": "d1",
                    "text": "无症状胆囊结石可不作治疗。",
                }
            ],
        )
        self.assertIn("只能依据给定公开文档证据回答", prompt)
        self.assertIn("证据直接支持时必须回答", prompt)
        self.assertIn("证据不足时拒答", prompt)
        self.assertIn("输出 JSON", prompt)
        self.assertIn("不要输出解释文字", prompt)
        self.assertIn("citations 只能使用 E1-E5", prompt)

        b0_prompt = build_public_tcm_qg_formal_prompt(
            question="什么类型的胆囊结石可不作治疗？",
            method="B0",
            evidence=[],
        )
        self.assertNotIn("citations", b0_prompt)
        self.assertIn("不使用外部检索证据", b0_prompt)

    def test_answer_prereg_manifest_excludes_private_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retrieval_run = root / "retrieval"
            retrieval_run.mkdir()
            atomic_write_json(
                retrieval_run / "matrix-summary.json",
                {"status": "completed", "split": "test"},
            )
            output = root / "answer-prereg.json"

            manifest = freeze_public_tcm_qg_formal_answer_prereg(
                output_path=output,
                retrieval_test_run_dir=retrieval_run,
                answer_methods=["B0", "B4", "P", "P-no-parent"],
                temperature=0,
                repeats=1,
                model_name="deepseek-chat",
                base_url_origin="https://api.deepseek.com",
            )

            serialized = str(manifest)
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(
                manifest["stage"],
                "public_tcm_qg_formal_answer_preregistered",
            )
            self.assertIn("prompt_sha256", manifest)
            self.assertIn("retrieval_test_matrix_sha256", manifest["inputs"])
            for forbidden in (
                "source_text",
                "question_text",
                "reference_answer",
                "answer_text",
                "evidence_text",
            ):
                self.assertNotIn(forbidden, serialized)
            self.assertTrue(output.is_file())

    def test_run_answer_matrix_writes_records_and_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            text = "无症状胆囊结石可不作治疗。"
            answer_start = text.index("无症状胆囊结石可不作治疗")
            write_jsonl(
                dataset,
                [
                    {
                        "qa_id": "q1",
                        "source_doc_id": "d1",
                        "split": "dev",
                        "question": "什么胆囊结石可不作治疗？",
                        "answer": "无症状胆囊结石可不作治疗",
                        "source_text": text,
                        "answer_start": answer_start,
                        "answer_end": answer_start + len("无症状胆囊结石可不作治疗"),
                        "review_status": "approved",
                        "question_version": 1,
                    }
                ],
            )
            retrieval_dir = root / "retrieval"
            retrieval_dir.mkdir()
            atomic_write_json(
                retrieval_dir / "matrix-config.json",
                {"status": "completed", "split": "dev"},
            )
            atomic_write_json(
                retrieval_dir / "matrix-summary.json",
                {"status": "completed", "split": "dev"},
            )
            for config_id, method_role in (
                ("b4-public-hybrid-rerank", "B4"),
                ("p-public-hybrid-rerank", "P"),
                ("p-public-no-parent", "P-no-parent"),
            ):
                config_dir = retrieval_dir / config_id
                config_dir.mkdir()
                write_jsonl(
                    config_dir / "per-question.jsonl",
                    [
                        {
                            "qa_id": "q1",
                            "source_doc_id": "d1",
                            "split": "dev",
                            "config_id": config_id,
                            "method_role": method_role,
                            "hits": [
                                {
                                    "chunk_id": "c1",
                                    "source_doc_id": "d1",
                                    "parent_id": "p1",
                                    "context_text": text,
                                    "context_start_index": 0,
                                    "context_char_count": len(text),
                                }
                            ],
                        }
                    ],
                )
            prereg = root / "answer-prereg.json"
            atomic_write_json(
                prereg,
                {
                    "status": "ready",
                    "answer_methods": ["B0", "B4", "P", "P-no-parent"],
                    "repeats": 1,
                    "model_name": "fake-model",
                    "base_url_origin": "https://api.example.com",
                    "prompt_sha256": "ABC",
                },
            )

            class FakeClient:
                calls = 0

                def invoke_json(self, *, prompt, method):
                    FakeClient.calls += 1
                    return {
                        "content": '{"answer":"无症状胆囊结石可不作治疗","abstain":false,"citations":["E1"]}',
                        "input_tokens": len(prompt),
                        "output_tokens": 10,
                    }

            first = run_public_tcm_qg_formal_answer_matrix(
                split="dev",
                dataset_path=dataset,
                retrieval_matrix_dir=retrieval_dir,
                answer_prereg_path=prereg,
                output_dir=root / "answer",
                client_factory=lambda: FakeClient(),
                max_workers=1,
            )
            second = run_public_tcm_qg_formal_answer_matrix(
                split="dev",
                dataset_path=dataset,
                retrieval_matrix_dir=retrieval_dir,
                answer_prereg_path=prereg,
                output_dir=root / "answer",
                resume_dir=Path(first["run_dir"]),
                client_factory=lambda: FakeClient(),
                max_workers=1,
            )

            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["completed_count"], 4)
            self.assertEqual(first["json_parse_error_rate"], 0)
            self.assertEqual(second["completed_count"], 4)
            self.assertEqual(FakeClient.calls, 4)

    def test_freeze_dev_requires_completed_zero_parse_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            write_jsonl(run_dir / "per-answer.jsonl", [{"qa_id": "q1"}])
            atomic_write_json(
                run_dir / "matrix-summary.json",
                {
                    "status": "completed",
                    "split": "dev",
                    "error_count": 0,
                    "json_parse_error_rate": 0,
                    "question_count": 1,
                    "expected_runs": 4,
                    "completed_count": 4,
                    "input_hashes": {"answer_prereg_sha256": "ABC"},
                },
            )
            output = root / "dev-freeze.json"

            manifest = freeze_public_tcm_qg_formal_answer_dev(
                run_dir=run_dir,
                output_path=output,
            )

            self.assertEqual(manifest["status"], "ready")
            self.assertTrue(manifest["answer_dev_frozen"])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
