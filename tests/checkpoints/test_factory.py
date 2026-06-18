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
        class FakeSaver:
            @classmethod
            def from_conn_string(cls, value):
                return {"conn": value}

        with patch("app.checkpoints.factory.AsyncPostgresSaver", FakeSaver):
            checkpointer = get_checkpointer(settings("postgres"))

        self.assertEqual(
            checkpointer,
            {"conn": "postgresql://user:pass@localhost:5432/tcm"},
        )


if __name__ == "__main__":
    unittest.main()
