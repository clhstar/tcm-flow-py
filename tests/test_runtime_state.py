import unittest
from unittest.mock import patch

from app.runtime.state import build_state
from app.store.postgres_run_manager import PostgresRunManager
from app.store.postgres_thread_store import PostgresThreadStore
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


class RuntimeStateTests(unittest.TestCase):
    def test_memory_settings_use_current_in_memory_stores(self):
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "memory"
            get_settings.return_value.rag_engine = "file"
            get_settings.return_value.database_url = None
            get_settings.return_value.postgres_pool_size = 10

            state = build_state(pool=None)

        self.assertIsInstance(state.thread_store, ThreadStore)
        self.assertIsInstance(state.run_manager, RunManager)

    def test_postgres_settings_require_pool_and_use_postgres_stores(self):
        pool = object()
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "postgres"
            get_settings.return_value.rag_engine = "database"
            get_settings.return_value.database_url = "postgresql://x"
            get_settings.return_value.postgres_pool_size = 10

            state = build_state(pool=pool)

        self.assertIsInstance(state.thread_store, PostgresThreadStore)
        self.assertIsInstance(state.run_manager, PostgresRunManager)

    def test_postgres_settings_without_pool_fail_fast(self):
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "postgres"
            get_settings.return_value.rag_engine = "file"
            get_settings.return_value.database_url = "postgresql://x"
            get_settings.return_value.postgres_pool_size = 10

            with self.assertRaisesRegex(ValueError, "pool"):
                build_state(pool=None)


if __name__ == "__main__":
    unittest.main()
