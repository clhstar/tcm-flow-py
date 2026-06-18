import unittest
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from app.config import AppSettings
from app.checkpoints.factory import get_checkpointer, reset_checkpointer_cache


def settings(backend: str) -> AppSettings:
    return AppSettings(
        database_url="postgresql://user:pass@localhost:5432/tcm",
        postgres_pool_size=10,
        checkpoint_backend=backend,
        rag_engine="file",
        rag_fallback_file_engine=True,
        elasticsearch_url=None,
        elasticsearch_rag_index_alias="tcm_rag_chunks_current",
        elasticsearch_analyzer="standard",
    )


class CheckpointerFactoryTests(unittest.TestCase):
    def tearDown(self):
        reset_checkpointer_cache()

    def test_memory_backend_returns_single_in_memory_saver(self):
        first = get_checkpointer(settings("memory"))
        second = get_checkpointer(settings("memory"))

        self.assertIsInstance(first, InMemorySaver)
        self.assertIs(first, second)

    def test_postgres_backend_requires_database_url(self):
        bad = AppSettings(
            database_url=None,
            postgres_pool_size=10,
            checkpoint_backend="postgres",
            rag_engine="file",
            rag_fallback_file_engine=True,
            elasticsearch_url=None,
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )

        with self.assertRaisesRegex(ValueError, "DATABASE_URL"):
            get_checkpointer(bad)

    def test_postgres_backend_uses_langgraph_postgres_saver(self):
        events = []

        class FakeContext:
            def __init__(self, value):
                self.value = value
                self.saver = FakeSaverInstance(value)

            def __enter__(self):
                events.append(("enter", self.value))
                return self.saver

            def __exit__(self, exc_type, exc, tb):
                events.append(("exit", self.value))
                return False

        class FakeSaverInstance:
            def __init__(self, value):
                self.value = value
                self.setup_called = False

            def setup(self):
                self.setup_called = True
                events.append(("setup", self.value))

        class FakeSaver:
            @classmethod
            def from_conn_string(cls, value):
                return FakeContext(value)

        with patch("app.checkpoints.factory._import_postgres_saver") as importer:
            importer.return_value = FakeSaver
            first = get_checkpointer(settings("postgres"))
            second = get_checkpointer(settings("postgres"))

        self.assertIs(first, second)
        self.assertTrue(first.setup_called)
        self.assertEqual(
            events,
            [
                ("enter", "postgresql://user:pass@localhost:5432/tcm"),
                ("setup", "postgresql://user:pass@localhost:5432/tcm"),
            ],
        )

        reset_checkpointer_cache()

        self.assertEqual(
            events[-1],
            ("exit", "postgresql://user:pass@localhost:5432/tcm"),
        )

    def test_postgres_backend_reports_missing_optional_dependency(self):
        with patch("app.checkpoints.factory.import_module") as importer:
            importer.side_effect = ModuleNotFoundError(
                "No module named 'langgraph.checkpoint.postgres'",
                name="langgraph.checkpoint.postgres",
            )

            with self.assertRaisesRegex(
                ModuleNotFoundError,
                "langgraph-checkpoint-postgres",
            ):
                get_checkpointer(settings("postgres"))


if __name__ == "__main__":
    unittest.main()
