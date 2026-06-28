import ast
from pathlib import Path
import unittest


WORKER_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "runtime" / "runs" / "worker.py"
)


class RuntimeWorkerBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = WORKER_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_worker_does_not_import_projection_or_message_parsing_dependencies(self):
        imported_modules: set[str] = set()
        imported_names: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module or "")
                imported_names.update(alias.name for alias in node.names)

        self.assertTrue(
            {
                "app.middlewares.guardrail_middleware",
                "app.middlewares.clarification_controller",
                "app.middlewares.trace_middleware",
                "langchain_core.messages",
                "app.runtime.public_messages",
            }.isdisjoint(imported_modules)
        )
        self.assertTrue(
            {
                "AIMessage",
                "apply_guardrails",
                "build_chat_response",
                "extract_latest_assistant_message",
                "extract_pending_clarification",
                "extract_latest_clarification_question",
                "extract_trace_events_from_messages",
            }.isdisjoint(imported_names)
        )

    def test_worker_contains_no_message_projection_or_checkpoint_rewrite_logic(self):
        defined_names = {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        loaded_names = {
            node.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }

        forbidden_names = {
            "AIMessage",
            "apply_guardrails",
            "append_visible_messages",
            "extract_final_ai_text",
            "extract_latest_assistant_message",
            "extract_latest_clarification_question",
            "extract_pending_clarification",
            "extract_text_from_content",
            "extract_trace_events_from_messages",
            "message_to_dict",
            "normalize_messages",
            "replace_final_ai_message_in_checkpoint",
            "_checkpoint_message_count",
        }
        self.assertTrue(forbidden_names.isdisjoint(defined_names | loaded_names))

        string_literals = {
            node.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue({"content", "role", "tool_calls"}.isdisjoint(string_literals))

    def test_worker_has_only_run_agent_as_a_top_level_function(self):
        top_level_functions = [
            node.name
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual(top_level_functions, ["run_agent"])

        run_agent = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_agent"
        )
        nested_functions = {
            node.name
            for node in ast.walk(run_agent)
            if node is not run_agent
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEqual(nested_functions, {"publish_update"})

    def test_worker_imports_and_uses_runtime_boundary_components(self):
        imported_names: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                imported_names.update(alias.asname or alias.name for alias in node.names)

        required_names = {
            "RunContext",
            "build_runtime_context",
            "build_runnable_config",
            "checkpoint_message_count",
            "RunCompletionProjection",
            "LangGraphStreamAdapter",
        }
        self.assertTrue(required_names.issubset(imported_names))

        loaded_names = {
            node.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertTrue(required_names.issubset(loaded_names))


if __name__ == "__main__":
    unittest.main()
