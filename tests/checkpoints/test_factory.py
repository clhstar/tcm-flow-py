import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from app.config import AppSettings
from app.checkpoints.factory import (
    get_checkpointer,
    get_checkpointer_async,
    reset_checkpointer_cache,
    reset_checkpointer_cache_async,
)


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

    def test_sync_postgres_backend_requires_async_factory(self):
        with self.assertRaisesRegex(RuntimeError, "async"):
            get_checkpointer(settings("postgres"))


class AsyncCheckpointerFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        await reset_checkpointer_cache_async()

    async def test_postgres_backend_reports_missing_optional_dependency(self):
        with patch("app.checkpoints.factory.import_module") as importer:
            importer.side_effect = ModuleNotFoundError(
                "No module named 'langgraph.checkpoint.postgres'",
                name="langgraph.checkpoint.postgres",
            )

            with self.assertRaisesRegex(
                ModuleNotFoundError,
                "langgraph-checkpoint-postgres",
            ):
                await get_checkpointer_async(settings("postgres"))

    async def test_non_postgres_module_import_error_is_not_rewritten(self):
        with patch("app.checkpoints.factory.import_module") as importer:
            importer.side_effect = ModuleNotFoundError(
                "No module named 'psycopg'",
                name="psycopg",
            )

            with self.assertRaisesRegex(ModuleNotFoundError, "psycopg"):
                await get_checkpointer_async(settings("postgres"))

    async def test_async_postgres_backend_uses_langgraph_postgres_saver(self):
        events = []

        class FakeSaverInstance:
            def __init__(self, value):
                self.value = value
                self.setup_called = False

            async def setup(self):
                self.setup_called = True
                events.append(("setup", self.value))

        class FakeSaver:
            @classmethod
            def from_conn_string(cls, value):
                @asynccontextmanager
                async def context():
                    events.append(("enter", value))
                    try:
                        yield FakeSaverInstance(value)
                    finally:
                        events.append(("exit", value))

                return context()

        with patch("app.checkpoints.factory._import_async_postgres_saver") as importer:
            importer.return_value = FakeSaver
            first = await get_checkpointer_async(settings("postgres"))
            second = await get_checkpointer_async(settings("postgres"))

        self.assertIs(first, second)
        self.assertTrue(first.setup_called)
        self.assertEqual(
            events,
            [
                ("enter", "postgresql://user:pass@localhost:5432/tcm"),
                ("setup", "postgresql://user:pass@localhost:5432/tcm"),
            ],
        )

        await reset_checkpointer_cache_async()

        self.assertEqual(
            events[-1],
            ("exit", "postgresql://user:pass@localhost:5432/tcm"),
        )


if __name__ == "__main__":
    unittest.main()
