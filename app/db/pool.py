import asyncpg

from app.config import AppSettings


async def create_pool_from_settings(settings: AppSettings):
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres persistence")
    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=1,
        max_size=settings.postgres_pool_size,
        command_timeout=60,
    )
