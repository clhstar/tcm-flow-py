import os
import unittest
from unittest.mock import patch

from app.config import AppSettings, get_settings, reset_settings_cache


class SettingsTests(unittest.TestCase):
    def setUp(self):
        reset_settings_cache()

    def tearDown(self):
        reset_settings_cache()

    def test_defaults_use_database_rag_and_memory_checkpointing(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(settings.checkpoint_backend, "memory")
        self.assertEqual(settings.postgres_pool_size, 10)
        self.assertEqual(settings.elasticsearch_rag_index_alias, "tcm_rag_chunks_current")
        self.assertEqual(settings.elasticsearch_analyzer, "standard")

    def test_database_configuration_is_loaded_from_environment(self):
        env = {
            "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost:5432/tcm",
            "POSTGRES_POOL_SIZE": "7",
            "CHECKPOINT_BACKEND": "postgres",
            "ELASTICSEARCH_URL": "http://localhost:9200",
            "ELASTICSEARCH_RAG_INDEX_ALIAS": "tcm_rag_chunks_test",
            "ELASTICSEARCH_ANALYZER": "ik_max_word",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(settings.database_url, env["DATABASE_URL"])
        self.assertEqual(settings.postgres_pool_size, 7)
        self.assertEqual(settings.checkpoint_backend, "postgres")
        self.assertEqual(settings.elasticsearch_url, "http://localhost:9200")
        self.assertEqual(settings.elasticsearch_rag_index_alias, "tcm_rag_chunks_test")
        self.assertEqual(settings.elasticsearch_analyzer, "ik_max_word")

    def test_pool_size_must_be_positive(self):
        with patch.dict(os.environ, {"POSTGRES_POOL_SIZE": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "POSTGRES_POOL_SIZE"):
                AppSettings.from_env()

    def test_cached_settings_can_be_reset_for_tests(self):
        with patch.dict(os.environ, {"ELASTICSEARCH_URL": "http://first:9200"}, clear=True):
            with patch("app.config.load_dotenv", return_value=False):
                first = get_settings()
        with patch.dict(os.environ, {"ELASTICSEARCH_URL": "http://second:9200"}, clear=True):
            with patch("app.config.load_dotenv", return_value=False):
                cached = get_settings()
                reset_settings_cache()
                refreshed = get_settings()

        self.assertEqual(first.elasticsearch_url, "http://first:9200")
        self.assertIs(first, cached)
        self.assertEqual(refreshed.elasticsearch_url, "http://second:9200")

    def test_get_settings_loads_dotenv_before_reading_environment(self):
        def load_test_dotenv(override=False):
            self.assertFalse(override)
            os.environ["ELASTICSEARCH_URL"] = "http://localhost:9200"
            return True

        with patch.dict(os.environ, {}, clear=True):
            with patch("app.config.load_dotenv", side_effect=load_test_dotenv):
                settings = get_settings()

        self.assertEqual(settings.elasticsearch_url, "http://localhost:9200")


if __name__ == "__main__":
    unittest.main()
