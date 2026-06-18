import asyncpg

from app.config import AppSettings


def normalize_asyncpg_dsn(dsn: str) -> str:
    prefix = "postgresql+asyncpg://"
    if dsn.startswith(prefix):
        return "postgresql://" + dsn[len(prefix) :]
    return dsn


async def create_pool_from_settings(settings: AppSettings):
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres persistence")
    return await asyncpg.create_pool(
        dsn=normalize_asyncpg_dsn(settings.database_url),
        min_size=1,
        max_size=settings.postgres_pool_size,
        command_timeout=60,
    )
