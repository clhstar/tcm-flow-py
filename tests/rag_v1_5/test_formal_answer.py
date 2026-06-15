import unittest

from pydantic import ValidationError

from experiments.rag_v1_5.schema import (
    FormalAnswerOutput,
    FormalAnswerRunRecord,
)


class FormalAnswerSchemaTests(unittest.TestCase):
    def test_output_requires_consistent_abstention_and_citations(self):
        valid = FormalAnswerOutput(
            answer="桂枝汤。",
            abstain=False,
            citations=["E1"],
        )
        self.assertEqual(valid.citations, ["E1"])

        with self.assertRaises(ValidationError):
            FormalAnswerOutput(
                answer="未找到。",
                abstain=True,
                citations=["E1"],
            )

    def test_run_record_has_frozen_method_repeat_and_usage(self):
        record = FormalAnswerRunRecord(
            question_id="formal-q-001",
            split="formal_dev",
            method="P",
            repeat_index=0,
            answer="桂枝汤。",
            abstain=False,
            citations=["E1"],
            retrieval_gate_abstain=False,
            evidence_ids=["E1"],
            latency_ms=123.0,
            input_tokens=100,
            output_tokens=10,
            model_name="frozen-model",
        )
        self.assertEqual(record.method, "P")
