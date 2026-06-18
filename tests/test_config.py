import os
import unittest
from unittest.mock import patch

from app.config import AppSettings, get_settings, reset_settings_cache


class SettingsTests(unittest.TestCase):
    def tearDown(self):
        reset_settings_cache()

    def test_defaults_keep_current_file_and_memory_behavior(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(settings.checkpoint_backend, "memory")
        self.assertEqual(settings.rag_engine, "file")
        self.assertTrue(settings.rag_fallback_file_engine)
        self.assertEqual(settings.elasticsearch_rag_index_alias, "tcm_rag_chunks_current")

    def test_database_configuration_is_loaded_from_environment(self):
        env = {
            "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost:5432/tcm",
            "POSTGRES_POOL_SIZE": "7",
            "CHECKPOINT_BACKEND": "postgres",
            "RAG_ENGINE": "database",
            "RAG_FALLBACK_FILE_ENGINE": "false",
            "ELASTICSEARCH_URL": "http://localhost:9200",
            "ELASTICSEARCH_RAG_INDEX_ALIAS": "tcm_rag_chunks_test",
            "ELASTICSEARCH_ANALYZER": "ik_max_word",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(settings.database_url, env["DATABASE_URL"])
        self.assertEqual(settings.postgres_pool_size, 7)
        self.assertEqual(settings.checkpoint_backend, "postgres")
        self.assertEqual(settings.rag_engine, "database")
        self.assertFalse(settings.rag_fallback_file_engine)
        self.assertEqual(settings.elasticsearch_url, "http://localhost:9200")
        self.assertEqual(settings.elasticsearch_analyzer, "ik_max_word")

    def test_cached_settings_can_be_reset_for_tests(self):
        with patch.dict(os.environ, {"RAG_ENGINE": "database"}, clear=True):
            first = get_settings()
        with patch.dict(os.environ, {"RAG_ENGINE": "file"}, clear=True):
            cached = get_settings()
            reset_settings_cache()
            refreshed = get_settings()

        self.assertIs(first, cached)
        self.assertEqual(refreshed.rag_engine, "file")


if __name__ == "__main__":
    unittest.main()
