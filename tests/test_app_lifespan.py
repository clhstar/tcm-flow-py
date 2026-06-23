import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

from app.config import AppSettings
from app.gateway.app import create_app
from app.runtime import state as runtime_state


def settings(*, checkpoint_backend: str = "memory"):
    return AppSettings(
        database_url="postgresql://tcm:tcm@localhost:5432/tcm_flow",
        postgres_pool_size=10,
        checkpoint_backend=checkpoint_backend,
        elasticsearch_url="http://localhost:9200",
        elasticsearch_rag_index_alias="tcm_rag_chunks_current",
        elasticsearch_analyzer="standard",
    )


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self):
        self.connection = object()
        self.closed = False

    def acquire(self):
        return FakeAcquire(self.connection)

    async def close(self):
        self.closed = True


class AppLifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_database_rag_always_creates_pool_runs_schema_and_registers_engine(self):
        pool = FakePool()
        engine = object()
        with patch(
            "app.gateway.app.get_settings",
            return_value=settings(),
        ):
            with patch(
                "app.gateway.app.create_pool_from_settings",
                new=AsyncMock(return_value=pool),
            ) as create_pool:
                with patch(
                    "app.gateway.app.run_schema_migrations",
                    new=AsyncMock(),
                ) as migrate:
                    with patch(
                        "app.gateway.app.build_database_engine",
                        return_value=engine,
                    ) as build_engine:
                        with patch(
                            "app.gateway.app.configure_database_engine",
                            Mock(),
                        ) as configure_engine:
                            app = create_app()
                            async with app.router.lifespan_context(app):
                                self.assertIs(app.state.postgres_pool, pool)

        create_pool.assert_awaited_once()
        migrate.assert_awaited_once_with(pool.connection)
        build_engine.assert_called_once()
        configure_engine.assert_called_once_with(engine)
        self.assertTrue(pool.closed)

    async def test_postgres_checkpoint_configures_runtime_state(self):
        pool = FakePool()
        engine = object()
        checkpointer = object()

        @asynccontextmanager
        async def fake_make_checkpointer(_settings):
            yield checkpointer

        configured_settings = settings(checkpoint_backend="postgres")
        with patch(
            "app.gateway.app.get_settings",
            return_value=configured_settings,
        ):
            with patch(
                "app.gateway.app.create_pool_from_settings",
                new=AsyncMock(return_value=pool),
            ):
                with patch("app.gateway.app.run_schema_migrations", new=AsyncMock()):
                    with patch(
                        "app.gateway.app.build_database_engine",
                        return_value=engine,
                    ) as build_engine:
                        with patch(
                            "app.gateway.app.configure_database_engine",
                            Mock(),
                        ) as configure_engine:
                            with patch(
                                "app.gateway.app.make_checkpointer",
                                fake_make_checkpointer,
                            ):
                                with patch(
                                    "app.gateway.app.runtime_state.configure_state"
                                ) as configure_state:
                                    app = create_app()
                                    async with app.router.lifespan_context(app):
                                        pass

        build_engine.assert_called_once()
        configure_engine.assert_called_once_with(engine)
        configure_state.assert_called_once_with(
            pool=pool,
            checkpointer=checkpointer,
            settings=configured_settings,
        )
        self.assertTrue(pool.closed)

    async def test_lifespan_opens_shared_checkpointer_and_drains_runs_before_closing_it(self):
        pool = FakePool()
        engine = object()
        checkpointer = object()
        events = []

        @asynccontextmanager
        async def fake_make_checkpointer(settings):
            events.append(("checkpointer_enter", settings.checkpoint_backend))
            try:
                yield checkpointer
            finally:
                events.append(("checkpointer_exit", None))

        with patch(
            "app.gateway.app.get_settings",
            return_value=settings(checkpoint_backend="postgres"),
        ):
            with patch(
                "app.gateway.app.create_pool_from_settings",
                new=AsyncMock(return_value=pool),
            ):
                with patch("app.gateway.app.run_schema_migrations", new=AsyncMock()):
                    with patch(
                        "app.gateway.app.build_database_engine",
                        return_value=engine,
                    ):
                        with patch("app.gateway.app.configure_database_engine", Mock()):
                            with patch(
                                "app.gateway.app.make_checkpointer",
                                fake_make_checkpointer,
                            ):
                                app = create_app()
                                async with app.router.lifespan_context(app):
                                    self.assertIs(app.state.checkpointer, checkpointer)
                                    self.assertIs(runtime_state.state.checkpointer, checkpointer)

                                    async def fake_shutdown(timeout=5.0):
                                        events.append(("run_manager_shutdown", timeout))

                                    runtime_state.state.run_manager.shutdown = fake_shutdown

        self.assertEqual(
            events,
            [
                ("checkpointer_enter", "postgres"),
                ("run_manager_shutdown", 5.0),
                ("checkpointer_exit", None),
            ],
        )
        self.assertTrue(pool.closed)


if __name__ == "__main__":
    unittest.main()
