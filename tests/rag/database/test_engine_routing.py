import unittest
from unittest.mock import patch

from app.config import AppSettings
from app.rag.retriever import resolve_retrieval_result
from app.rag.vector_store import get_configured_retrieval_engine


class EngineRoutingTests(unittest.TestCase):
    def settings(self) -> AppSettings:
        return AppSettings(
            database_url="postgresql://user:pass@localhost:5432/tcm",
            postgres_pool_size=10,
            checkpoint_backend="memory",
            elasticsearch_url="http://localhost:9200",
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )

    def test_configured_engine_always_uses_database_factory(self):
        with patch(
            "app.rag.vector_store.get_database_engine",
            return_value="database-engine",
        ):
            engine = get_configured_retrieval_engine(self.settings())

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
