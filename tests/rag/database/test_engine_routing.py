import unittest
from unittest.mock import patch

from app.config import AppSettings
from app.rag.retriever import resolve_retrieval_result
from app.rag.vector_store import get_configured_retrieval_engine


class EngineRoutingTests(unittest.TestCase):
    def settings(self, rag_engine: str) -> AppSettings:
        return AppSettings(
            database_url="postgresql://user:pass@localhost:5432/tcm",
            postgres_pool_size=10,
            checkpoint_backend="memory",
            rag_engine=rag_engine,
            rag_fallback_file_engine=True,
            elasticsearch_url="http://localhost:9200",
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )

    def test_file_engine_uses_existing_production_engine(self):
        with patch(
            "app.rag.vector_store.get_production_engine",
            return_value="file-engine",
        ):
            engine = get_configured_retrieval_engine(self.settings("file"))

        self.assertEqual(engine, "file-engine")

    def test_database_engine_uses_database_factory(self):
        with patch(
            "app.rag.vector_store.get_database_engine",
            return_value="database-engine",
        ):
            engine = get_configured_retrieval_engine(self.settings("database"))

        self.assertEqual(engine, "database-engine")

    async def async_payload(self):
        return {"status": "ok"}

    def test_resolve_retrieval_result_accepts_sync_payload(self):
        self.assertEqual(resolve_retrieval_result({"status": "ok"}), {"status": "ok"})

    def test_resolve_retrieval_result_rejects_async_payload_in_sync_path(self):
        coroutine = self.async_payload()
        try:
            with self.assertRaisesRegex(RuntimeError, "async"):
                resolve_retrieval_result(coroutine)
        finally:
            coroutine.close()


if __name__ == "__main__":
    unittest.main()
