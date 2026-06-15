import hashlib
import json
import tempfile
import threading
import time
import unittest
import warnings
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

    def _write_answer_matrix_fixture(self, *, split):
        dataset_path = self.root / f"{split}.jsonl"
        matrix_dir = self.root / f"{split}-retrieval"
        config_path = self.root / "formal-answer.yaml"
        formal_manifest_path = self.root / "formal-400.json"
        formal_runs_manifest_path = self.root / "formal-runs.json"
        answer_prereg_path = self.root / "answer-prereg.json"
        config_path.write_text(
            """
version: v1.5.0
seed: 20260616
model:
  env_model_key: OPENAI_MODEL
  env_base_url_key: OPENAI_BASE_URL
  temperature: 0.2
  max_tokens: 512
  timeout_seconds: 120
  max_retries: 2
  structured_output_method: json_mode
generation:
  repeats: 3
  context_top_k: 5
  answer_methods: [B0, B4, P, P-no-parent]
execution:
  max_workers: 2
""".lstrip(),
            encoding="utf-8",
        )
        dataset_path.write_text(
            json.dumps(
                {
                    "question_id": "q-1",
                    "question": "测试问题",
                    "reference_answer": "测试答案",
                    "answerable": True,
                    "gold_clause_ids": ["gold-1"],
                    "review_status": "approved",
                    "split": split,
                },
                ensure_ascii=False,
            )
            + "\n",
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
        matrix_dir.mkdir()
        matrix_config = {
            "status": "completed",
            "split": split,
            "input_hashes": {
                "dataset_sha256": hashlib.sha256(
                    dataset_path.read_bytes()
                ).hexdigest().upper(),
                "formal_manifest_sha256": hashlib.sha256(
                    formal_manifest_path.read_bytes()
                ).hexdigest().upper(),
            },
        }
        matrix_config_path = matrix_dir / "matrix-config.json"
        matrix_config_path.write_text(
            json.dumps(matrix_config),
            encoding="utf-8",
        )
        for config_id, context_text in (
            ("b4-c0-hybrid-rerank", "b4 text"),
            ("p-c4-hybrid-rerank", "parent text"),
            ("p-no-parent", "child text"),
        ):
            config_dir = matrix_dir / config_id
            config_dir.mkdir()
            (config_dir / "per-question.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "q-1",
                        "config_id": config_id,
                        "hits": [
                            {
                                "chunk_id": "chunk-1",
                                "clause_ids": ["gold-1"],
                                "context_text": context_text,
                                "text": "child text",
                                "retrieval_parent_id": "parent-1",
                                "reranker_score": 0.9,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        matrix_hash_key = (
            "dev_matrix_config_sha256"
            if split == "formal_dev"
            else "test_matrix_config_sha256"
        )
        answer_prereg = {
            "status": "ready",
            "model": {
                "name": "frozen-model",
                "base_url_origin": "example.invalid",
            },
            "inputs": {
                "config_sha256": hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest().upper(),
                matrix_hash_key: hashlib.sha256(
                    matrix_config_path.read_bytes()
                ).hexdigest().upper(),
                "formal_manifest_sha256": hashlib.sha256(
                    formal_manifest_path.read_bytes()
                ).hexdigest().upper(),
                "formal_runs_manifest_sha256": hashlib.sha256(
                    formal_runs_manifest_path.read_bytes()
                ).hexdigest().upper(),
            },
            "methods": ["B0", "B4", "P", "P-no-parent"],
            "repeats": 3,
        }
        answer_prereg_path.write_text(
            json.dumps(answer_prereg),
            encoding="utf-8",
        )
        return {
            "dataset_path": dataset_path,
            "matrix_dir": matrix_dir,
            "config_path": config_path,
            "formal_manifest_path": formal_manifest_path,
            "formal_runs_manifest_path": formal_runs_manifest_path,
            "answer_prereg_path": answer_prereg_path,
        }

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

    def test_loads_only_frozen_b4_p_and_child_contexts(self):
        from experiments.rag_v1_5.formal_answer import (
            load_frozen_answer_inputs,
        )

        dataset_path = self.root / "formal-400.jsonl"
        matrix_dir = self.root / "matrix"
        answer_prereg_path = self.root / "answer-prereg.json"
        dataset_path.write_text(
            json.dumps(
                {
                    "question_id": "q-1",
                    "question": "测试问题",
                    "reference_answer": "测试答案",
                    "answerable": True,
                    "gold_clause_ids": ["gold-1"],
                    "review_status": "approved",
                    "split": "formal_dev",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        matrix_dir.mkdir()
        matrix_config = {
            "status": "completed",
            "split": "formal_dev",
            "input_hashes": {
                "dataset_sha256": hashlib.sha256(
                    dataset_path.read_bytes()
                ).hexdigest().upper(),
                "formal_manifest_sha256": "A" * 64,
            },
        }
        (matrix_dir / "matrix-config.json").write_text(
            json.dumps(matrix_config),
            encoding="utf-8",
        )
        for config_id, context_text in (
            ("b4-c0-hybrid-rerank", "b4 text"),
            ("p-c4-hybrid-rerank", "parent text"),
            ("p-no-parent", "child text"),
        ):
            config_dir = matrix_dir / config_id
            config_dir.mkdir()
            record = {
                "question_id": "q-1",
                "config_id": config_id,
                "hits": [
                    {
                        "chunk_id": "chunk-1",
                        "clause_ids": ["gold-1"],
                        "context_text": context_text,
                        "text": "child text",
                        "retrieval_parent_id": "parent-1",
                        "reranker_score": 0.9,
                    }
                ],
            }
            (config_dir / "per-question.jsonl").write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        answer_prereg = {
            "status": "ready",
            "inputs": {
                "dev_matrix_config_sha256": hashlib.sha256(
                    (matrix_dir / "matrix-config.json").read_bytes()
                ).hexdigest().upper(),
                "formal_manifest_sha256": "A" * 64,
                "formal_runs_manifest_sha256": "B" * 64,
            },
        }
        answer_prereg_path.write_text(
            json.dumps(answer_prereg),
            encoding="utf-8",
        )

        loaded = load_frozen_answer_inputs(
            dataset_path=dataset_path,
            matrix_dir=matrix_dir,
            answer_prereg_path=answer_prereg_path,
            split="formal_dev",
        )

        self.assertEqual(
            set(loaded["retrieval"]),
            {"B4", "P", "P-no-parent"},
        )
        self.assertEqual(
            loaded["retrieval"]["P"]["q-1"]["evidence"][0]["label"],
            "E1",
        )
        self.assertEqual(
            loaded["retrieval"]["P-no-parent"]["q-1"]["evidence"][0][
                "text"
            ],
            "child text",
        )

    def test_parent_context_deduplicates_by_retrieval_parent_id(self):
        from experiments.rag_v1_5.formal_answer import build_evidence_items

        record = {
            "config_id": "p-c4-hybrid-rerank",
            "hits": [
                {
                    "chunk_id": "chunk-1",
                    "clause_ids": ["c-1"],
                    "context_text": "parent one",
                    "text": "child one",
                    "retrieval_parent_id": "parent-1",
                    "reranker_score": 0.9,
                },
                {
                    "chunk_id": "chunk-2",
                    "clause_ids": ["c-2"],
                    "context_text": "parent one",
                    "text": "child two",
                    "retrieval_parent_id": "parent-1",
                    "reranker_score": 0.8,
                },
                {
                    "chunk_id": "chunk-3",
                    "clause_ids": ["c-3"],
                    "context_text": "parent two",
                    "text": "child three",
                    "retrieval_parent_id": "parent-2",
                    "reranker_score": 0.7,
                },
            ],
        }

        items = build_evidence_items(record)

        self.assertEqual(
            [item["label"] for item in items],
            ["E1", "E2"],
        )
        self.assertEqual(len(items), 2)

    def test_prompt_requires_evidence_only_answer_and_citations(self):
        from experiments.rag_v1_5.formal_answer import (
            build_answer_messages,
        )

        messages = build_answer_messages(
            question="桂枝汤由哪些药组成？",
            evidence=[
                {
                    "label": "E1",
                    "text": "桂枝、芍药、生姜...",
                }
            ],
            method="P",
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertIn("只能依据给定证据", serialized)
        self.assertIn("citations", serialized)
        self.assertIn("E1", serialized)

    def test_b0_prompt_has_no_evidence_labels(self):
        from experiments.rag_v1_5.formal_answer import (
            build_answer_messages,
        )

        messages = build_answer_messages(
            question="桂枝汤由哪些药组成？",
            evidence=[],
            method="B0",
        )
        self.assertNotIn(
            "[E1]",
            json.dumps(messages, ensure_ascii=False),
        )

    def test_model_adapter_parses_structured_output_and_usage(self):
        from experiments.rag_v1_5.formal_answer import FormalAnswerModel

        class FakeRaw:
            response_metadata = {
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                },
                "system_fingerprint": "fp-test",
            }

        class FakeStructuredModel:
            def invoke(self, messages):
                self.messages = messages
                return {
                    "parsed": {
                        "answer": "桂枝汤。",
                        "abstain": False,
                        "citations": ["E1"],
                    },
                    "raw": FakeRaw(),
                }

        class FakeChatModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.structured = FakeStructuredModel()

            def with_structured_output(
                self,
                schema,
                *,
                method,
                include_raw,
            ):
                self.schema = schema
                self.method = method
                self.include_raw = include_raw
                return self.structured

        created = []

        def fake_factory(**kwargs):
            model = FakeChatModel(**kwargs)
            created.append(model)
            return model

        adapter = FormalAnswerModel(
            config={
                "model": {
                    "env_model_key": "OPENAI_MODEL",
                    "env_base_url_key": "OPENAI_BASE_URL",
                    "temperature": 0.2,
                    "max_tokens": 512,
                    "timeout_seconds": 120,
                    "max_retries": 2,
                    "structured_output_method": "json_mode",
                }
            },
            env={
                "OPENAI_MODEL": "frozen-model",
                "OPENAI_BASE_URL": "https://example.invalid/v1",
            },
            chat_model_factory=fake_factory,
        )
        parsed, metadata = adapter.invoke(
            [{"role": "user", "content": "test"}]
        )

        self.assertEqual(parsed.answer, "桂枝汤。")
        self.assertEqual(metadata["input_tokens"], 12)
        self.assertEqual(metadata["output_tokens"], 3)
        self.assertEqual(
            created[0].kwargs["model"],
            "frozen-model",
        )

    def test_calibrates_threshold_on_dev_only_with_stable_tie_break(
        self,
    ):
        from experiments.rag_v1_5.formal_answer import calibrate_threshold

        rows = [
            {"answerable": True, "score": 0.9},
            {"answerable": True, "score": 0.8},
            {"answerable": False, "score": 0.4},
            {"answerable": False, "score": 0.2},
        ]

        result = calibrate_threshold(rows)

        self.assertEqual(result["objective"], "balanced_accuracy")
        self.assertGreater(result["threshold"], 0.4)
        self.assertLessEqual(result["threshold"], 0.8)

    def test_test_run_requires_frozen_dev(self):
        from experiments.rag_v1_5.formal_answer import (
            run_formal_answer_matrix,
        )

        with self.assertRaises(ValueError):
            run_formal_answer_matrix(
                split="formal_test",
                dev_freeze_path=self.root / "missing.json",
                output_dir=self.root / "test",
                model_factory=lambda **_: None,
            )

    def test_abstention_output_is_canonicalized_before_validation(self):
        from experiments.rag_v1_5.formal_answer import (
            ABSTAIN_ANSWER,
            canonicalize_answer_output,
        )
        from experiments.rag_v1_5.schema import FormalAnswerOutput

        output = canonicalize_answer_output(
            FormalAnswerOutput(
                answer="证据不足，无法回答。",
                abstain=True,
                citations=[],
            )
        )

        self.assertEqual(output.answer, ABSTAIN_ANSWER)
        self.assertEqual(output.citations, [])

    def test_dev_matrix_writes_all_methods_and_repeats(self):
        from experiments.rag_v1_5.formal_answer import (
            run_formal_answer_matrix,
        )
        from experiments.rag_v1_5.schema import FormalAnswerOutput

        fixture = self._write_answer_matrix_fixture(
            split="formal_dev"
        )

        class FakeModel:
            model_name = "frozen-model"

            def invoke(self, messages):
                nonlocal active_calls, max_active_calls
                with call_lock:
                    active_calls += 1
                    max_active_calls = max(
                        max_active_calls,
                        active_calls,
                    )
                time.sleep(0.01)
                with call_lock:
                    active_calls -= 1
                cited = "证据：" in messages[-1]["content"]
                return (
                    FormalAnswerOutput(
                        answer="测试答案",
                        abstain=False,
                        citations=["E1"] if cited else [],
                    ),
                    {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "system_fingerprint": "fp-test",
                    },
                )

        call_lock = threading.Lock()
        active_calls = 0
        max_active_calls = 0
        summary = run_formal_answer_matrix(
            split="formal_dev",
            output_dir=self.root / "answer-dev",
            model_factory=lambda **_: FakeModel(),
            **fixture,
        )

        self.assertEqual(summary["expected_runs"], 12)
        self.assertEqual(summary["completed_count"], 12)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["status"], "completed")
        records = (
            Path(summary["run_dir"]) / "per-answer.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(records), 12)
        self.assertGreaterEqual(max_active_calls, 2)

    def test_answer_cli_contracts(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from experiments.rag_v1_5.cli import build_parser

        dev = build_parser().parse_args(["run-formal-answer-dev"])
        freeze = build_parser().parse_args(
            ["freeze-formal-answer-dev"]
        )
        test = build_parser().parse_args(["run-formal-answer-test"])
        summarize = build_parser().parse_args(
            [
                "summarize-formal-answer-test",
                "--run-dir",
                "answer-test",
            ]
        )
        prepare_review = build_parser().parse_args(
            [
                "prepare-formal-answer-review",
                "--run-dir",
                "answer-test",
            ]
        )
        import_review = build_parser().parse_args(
            ["import-formal-answer-review"]
        )
        self.assertEqual(dev.command, "run-formal-answer-dev")
        self.assertEqual(
            freeze.command,
            "freeze-formal-answer-dev",
        )
        self.assertEqual(test.command, "run-formal-answer-test")
        self.assertEqual(
            summarize.command,
            "summarize-formal-answer-test",
        )
        self.assertEqual(
            prepare_review.command,
            "prepare-formal-answer-review",
        )
        self.assertEqual(
            import_review.command,
            "import-formal-answer-review",
        )
