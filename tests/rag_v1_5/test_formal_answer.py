import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from experiments.rag_v1_5.schema import (
    FormalAnswerOutput,
    FormalAnswerRunRecord,
)


class FormalAnswerSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

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

    def test_freeze_answer_prereg_records_only_hashes_and_model_identity(
        self,
    ):
        from experiments.rag_v1_5.formal_answer import (
            freeze_formal_answer_prereg,
        )

        config_path = self.root / "formal-answer.yaml"
        formal_manifest_path = self.root / "formal-400.json"
        formal_runs_manifest_path = self.root / "formal-runs.json"
        dev_run_dir = self.root / "dev-run"
        test_run_dir = self.root / "test-run"
        output_path = self.root / "answer-prereg.json"
        config_path.write_text(
            """
version: v1.5.0
model:
  env_model_key: OPENAI_MODEL
  env_base_url_key: OPENAI_BASE_URL
generation:
  repeats: 3
  answer_methods: [B0, B4, P, P-no-parent]
""".lstrip(),
            encoding="utf-8",
        )
        formal_manifest_path.write_text(
            '{"status":"ready"}\n',
            encoding="utf-8",
        )
        formal_runs_manifest_path.write_text(
            '{"status":"ready"}\n',
            encoding="utf-8",
        )
        dev_run_dir.mkdir()
        test_run_dir.mkdir()
        (dev_run_dir / "matrix-config.json").write_text(
            '{"split":"formal_dev"}\n',
            encoding="utf-8",
        )
        (test_run_dir / "matrix-config.json").write_text(
            '{"split":"formal_test"}\n',
            encoding="utf-8",
        )

        manifest = freeze_formal_answer_prereg(
            config_path=config_path,
            formal_manifest_path=formal_manifest_path,
            formal_runs_manifest_path=formal_runs_manifest_path,
            dev_run_dir=dev_run_dir,
            test_run_dir=test_run_dir,
            output_path=output_path,
            env={
                "OPENAI_MODEL": "frozen-model",
                "OPENAI_BASE_URL": "https://example.invalid/v1",
            },
        )
        serialized = json.dumps(manifest, ensure_ascii=False)

        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(manifest["model"]["name"], "frozen-model")
        self.assertNotIn("api_key", serialized.lower())
        self.assertNotIn("reference_answer", serialized)
