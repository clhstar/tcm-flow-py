import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from app.config import AppSettings
from app.checkpoints.factory import (
    make_checkpointer,
)


def settings(backend: str) -> AppSettings:
    return AppSettings(
        database_url="postgresql://user:pass@localhost:5432/tcm",
        postgres_pool_size=10,
        checkpoint_backend=backend,
        elasticsearch_url=None,
        elasticsearch_rag_index_alias="tcm_rag_chunks_current",
        elasticsearch_analyzer="standard",
    )


class CheckpointerFactoryTests(unittest.TestCase):
    def test_postgres_backend_requires_database_url(self):
        bad = AppSettings(
            database_url=None,
            postgres_pool_size=10,
            checkpoint_backend="postgres",
            elasticsearch_url=None,
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )

        with self.assertRaisesRegex(ValueError, "DATABASE_URL"):
            async def enter():
                async with make_checkpointer(bad):
                    pass

            import asyncio

            asyncio.run(enter())


class AsyncCheckpointerFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_backend_context_manager_yields_in_memory_saver(self):
        async with make_checkpointer(settings("memory")) as checkpointer:
            self.assertIsInstance(checkpointer, InMemorySaver)

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
                async with make_checkpointer(settings("postgres")):
                    pass

    async def test_non_postgres_module_import_error_is_not_rewritten(self):
        with patch("app.checkpoints.factory.import_module") as importer:
            importer.side_effect = ModuleNotFoundError(
                "No module named 'psycopg'",
                name="psycopg",
            )

            with self.assertRaisesRegex(ModuleNotFoundError, "psycopg"):
                async with make_checkpointer(settings("postgres")):
                    pass

    async def test_postgres_backend_enters_sets_up_and_exits_saver_context(self):
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
            async with make_checkpointer(settings("postgres")) as checkpointer:
                self.assertTrue(checkpointer.setup_called)
                self.assertEqual(
                    events,
                    [
                        ("enter", "postgresql://user:pass@localhost:5432/tcm"),
                        ("setup", "postgresql://user:pass@localhost:5432/tcm"),
                    ],
                )

        self.assertEqual(
            events,
            [
                ("enter", "postgresql://user:pass@localhost:5432/tcm"),
                ("setup", "postgresql://user:pass@localhost:5432/tcm"),
                ("exit", "postgresql://user:pass@localhost:5432/tcm"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
