import unittest
from unittest.mock import AsyncMock, patch

from app.config import AppSettings
from app.db.pool import create_pool_from_settings, normalize_asyncpg_dsn


class PoolTests(unittest.IsolatedAsyncioTestCase):
    def test_normalize_asyncpg_dsn_converts_sqlalchemy_style_scheme(self):
        self.assertEqual(
            normalize_asyncpg_dsn("postgresql+asyncpg://user:pass@localhost:5432/tcm"),
            "postgresql://user:pass@localhost:5432/tcm",
        )
        self.assertEqual(
            normalize_asyncpg_dsn("postgresql://user:pass@localhost:5432/tcm"),
            "postgresql://user:pass@localhost:5432/tcm",
        )
        self.assertEqual(
            normalize_asyncpg_dsn("postgres://user:pass@localhost:5432/tcm"),
            "postgres://user:pass@localhost:5432/tcm",
        )

    async def test_create_pool_requires_database_url(self):
        settings = AppSettings(
            database_url=None,
            postgres_pool_size=10,
            checkpoint_backend="postgres",
            rag_engine="database",
            rag_fallback_file_engine=True,
            elasticsearch_url=None,
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )

        with self.assertRaisesRegex(ValueError, "DATABASE_URL"):
            await create_pool_from_settings(settings)

    async def test_create_pool_uses_configured_size(self):
        settings = AppSettings(
            database_url="postgresql://user:pass@localhost:5432/tcm",
            postgres_pool_size=3,
            checkpoint_backend="postgres",
            rag_engine="database",
            rag_fallback_file_engine=True,
            elasticsearch_url=None,
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )
        fake_pool = object()

        with patch("app.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)) as create_pool:
            pool = await create_pool_from_settings(settings)

        self.assertIs(pool, fake_pool)
        create_pool.assert_awaited_once_with(
            dsn=settings.database_url,
            min_size=1,
            max_size=3,
            command_timeout=60,
        )

    async def test_create_pool_normalizes_sqlalchemy_style_database_url(self):
        settings = AppSettings(
            database_url="postgresql+asyncpg://user:pass@localhost:5432/tcm",
            postgres_pool_size=3,
            checkpoint_backend="postgres",
            rag_engine="database",
            rag_fallback_file_engine=True,
            elasticsearch_url=None,
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )
        fake_pool = object()

        with patch("app.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)) as create_pool:
            pool = await create_pool_from_settings(settings)

        self.assertIs(pool, fake_pool)
        create_pool.assert_awaited_once_with(
            dsn="postgresql://user:pass@localhost:5432/tcm",
            min_size=1,
            max_size=3,
            command_timeout=60,
        )


if __name__ == "__main__":
    unittest.main()
