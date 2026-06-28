import copy
import unittest

from app.runtime.runs.context import (
    RunContext,
    build_runnable_config,
    build_runtime_context,
)
from app.runtime.runs.input import extract_user_text, normalize_graph_input
from app.store.models import RunRecord
from app.store.thread_store import ThreadStore


class RuntimeRunInputTests(unittest.TestCase):
    def test_normalize_graph_input_preserves_supported_roles_and_text_blocks(self):
        graph_input = normalize_graph_input(
            {
                "messages": [
                    {
                        "type": "human",
                        "content": [
                            {"type": "text", "text": "first "},
                            {"type": "text", "text": "question"},
                        ],
                    },
                    {"type": "ai", "content": "prior answer"},
                    {"type": "system", "content": "system rule"},
                    {"type": "tool", "content": "do not replay"},
                    {"type": "human", "content": "   "},
                    "not a message",
                ]
            }
        )

        self.assertEqual(
            graph_input,
            {
                "messages": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "prior answer"},
                    {"role": "system", "content": "system rule"},
                ]
            },
        )

    def test_extract_user_text_accepts_raw_and_normalized_messages(self):
        self.assertEqual(
            extract_user_text(
                {
                    "messages": [
                        {"type": "human", "content": "  current question  "},
                        {"type": "ai", "content": "ignored"},
                    ]
                }
            ),
            "current question",
        )
        self.assertEqual(
            extract_user_text(
                {"messages": [{"role": "user", "content": " normalized "}]}
            ),
            "normalized",
        )


class RuntimeRunContextTests(unittest.TestCase):
    def setUp(self):
        self.record = RunRecord(
            run_id="run-1",
            thread_id="thread-1",
            assistant_id="lead_agent",
        )

    def test_runtime_context_protects_identity_and_preserves_model_options(self):
        runtime_context = build_runtime_context(
            self.record,
            {
                "thread_id": "caller-thread",
                "run_id": "caller-run",
                "model_name": "test-model",
                "temperature": 0.2,
            },
        )

        self.assertEqual(runtime_context["thread_id"], "thread-1")
        self.assertEqual(runtime_context["run_id"], "run-1")
        self.assertEqual(runtime_context["model_name"], "test-model")
        self.assertEqual(runtime_context["temperature"], 0.2)

    def test_runnable_config_preserves_request_fields_and_forces_identity(self):
        runtime_context = build_runtime_context(
            self.record,
            {"model_name": "test-model", "recursion_limit": 77},
        )

        config = build_runnable_config(
            self.record,
            {
                "configurable": {
                    "thread_id": "wrong",
                    "run_id": "caller-run",
                    "custom": "value",
                },
                "metadata": {"source": "test"},
                "recursion_limit": 88,
                "context": {
                    "request_context": True,
                    "run_id": "caller-run",
                },
            },
            runtime_context,
        )

        self.assertEqual(config["configurable"]["thread_id"], "thread-1")
        self.assertEqual(config["configurable"]["run_id"], "run-1")
        self.assertEqual(config["configurable"]["custom"], "value")
        self.assertEqual(config["metadata"], {"source": "test"})
        self.assertEqual(config["recursion_limit"], 88)
        self.assertTrue(config["context"]["request_context"])
        self.assertEqual(config["context"]["model_name"], "test-model")
        self.assertEqual(config["context"]["run_id"], "run-1")

    def test_runnable_config_does_not_evaluate_default_when_request_sets_limit(self):
        config = build_runnable_config(
            self.record,
            {"recursion_limit": 88},
            build_runtime_context(
                self.record,
                {"recursion_limit": "invalid"},
            ),
        )

        self.assertEqual(config["recursion_limit"], 88)

    def test_runnable_config_defaults_recursion_limit_from_runtime_context_or_50(self):
        with_runtime_default = build_runnable_config(
            self.record,
            {},
            build_runtime_context(self.record, {"recursion_limit": 77}),
        )
        with_fallback_default = build_runnable_config(
            self.record,
            None,
            build_runtime_context(self.record, None),
        )

        self.assertEqual(with_runtime_default["recursion_limit"], 77)
        self.assertEqual(with_fallback_default["recursion_limit"], 50)

    def test_run_context_groups_thread_store_and_agent_context(self):
        thread_store = ThreadStore()
        ctx = RunContext(
            thread_store=thread_store,
            agent_context={"temperature": 0.2},
        )

        self.assertIs(ctx.thread_store, thread_store)
        self.assertEqual(dict(ctx.agent_context), {"temperature": 0.2})

    def test_context_builders_do_not_mutate_caller_mappings(self):
        agent_context = {
            "thread_id": "caller-thread",
            "run_id": "caller-run",
            "model_name": "test-model",
        }
        request_config = {
            "configurable": {"thread_id": "wrong", "custom": "value"},
            "metadata": {"source": "test"},
            "context": {"request_context": True},
        }
        original_agent_context = copy.deepcopy(agent_context)
        original_request_config = copy.deepcopy(request_config)

        runtime_context = build_runtime_context(self.record, agent_context)
        original_runtime_context = copy.deepcopy(runtime_context)
        build_runnable_config(self.record, request_config, runtime_context)

        self.assertEqual(agent_context, original_agent_context)
        self.assertEqual(request_config, original_request_config)
        self.assertEqual(runtime_context, original_runtime_context)


if __name__ == "__main__":
    unittest.main()
