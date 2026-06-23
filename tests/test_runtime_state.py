import importlib
import os
import unittest
from unittest.mock import patch

from app.config import reset_settings_cache
from app.runtime import state as runtime_state
from app.runtime.state import build_state, configure_state, reset_state_to_memory
from app.store.postgres_run_manager import PostgresRunManager
from app.store.postgres_thread_store import PostgresThreadStore
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


class RuntimeStateTests(unittest.TestCase):
    def tearDown(self):
        reset_state_to_memory()
        reset_settings_cache()

    def test_memory_settings_use_current_in_memory_stores(self):
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "memory"
            get_settings.return_value.database_url = None
            get_settings.return_value.postgres_pool_size = 10

            state = build_state(pool=None)

        self.assertIsInstance(state.thread_store, ThreadStore)
        self.assertIsInstance(state.run_manager, RunManager)

    def test_postgres_settings_require_pool_and_use_postgres_stores(self):
        pool = object()
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "postgres"
            get_settings.return_value.database_url = "postgresql://x"
            get_settings.return_value.postgres_pool_size = 10

            state = build_state(pool=pool)

        self.assertIsInstance(state.thread_store, PostgresThreadStore)
        self.assertIsInstance(state.run_manager, PostgresRunManager)

    def test_database_rag_engine_does_not_force_runtime_store_postgres(self):
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "memory"
            get_settings.return_value.database_url = "postgresql://x"
            get_settings.return_value.postgres_pool_size = 10

            state = build_state(pool=None)

        self.assertIsInstance(state.thread_store, ThreadStore)
        self.assertIsInstance(state.run_manager, RunManager)

    def test_postgres_settings_without_pool_fail_fast(self):
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "postgres"
            get_settings.return_value.database_url = "postgresql://x"
            get_settings.return_value.postgres_pool_size = 10

            with self.assertRaisesRegex(ValueError, "pool"):
                build_state(pool=None)

    def test_import_keeps_memory_state_under_postgres_environment(self):
        env = {
            "CHECKPOINT_BACKEND": "postgres",
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/tcm",
        }
        with patch.dict(os.environ, env, clear=True):
            reset_settings_cache()

            reloaded = importlib.reload(runtime_state)

        self.assertIsInstance(reloaded.state.thread_store, ThreadStore)
        self.assertIsInstance(reloaded.state.run_manager, RunManager)

    def test_configure_state_mutates_shared_global_state(self):
        pool = object()
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "postgres"
            get_settings.return_value.database_url = "postgresql://x"
            get_settings.return_value.postgres_pool_size = 10

            configured = configure_state(pool=pool)

        self.assertIs(configured, runtime_state.state)
        self.assertIsInstance(runtime_state.state.thread_store, PostgresThreadStore)
        self.assertIsInstance(runtime_state.state.run_manager, PostgresRunManager)


if __name__ == "__main__":
    unittest.main()
