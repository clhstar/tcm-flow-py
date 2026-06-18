import json
import uuid
from datetime import datetime
from typing import Any

from app.store.models import ThreadRecord


def _isoformat(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _thread_from_row(row) -> ThreadRecord | None:
    if row is None:
        return None
    return ThreadRecord(
        thread_id=str(row["thread_id"]),
        created_at=_isoformat(row["created_at"]),
        updated_at=_isoformat(row["updated_at"]),
        status=row["status"],
        values=dict(row["metadata"] or {}),
    )


class PostgresThreadStore:
    def __init__(self, pool):
        self.pool = pool

    async def create(self) -> ThreadRecord:
        now = datetime.utcnow()
        thread_id = uuid.uuid4()
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                insert into app_threads (thread_id, status, created_at, updated_at, metadata)
                values ($1, 'idle', $2, $2, '{}'::jsonb)
                returning thread_id, created_at, updated_at, status, metadata
                """,
                thread_id,
                now,
            )
        record = _thread_from_row(row)
        if record is None:
            raise RuntimeError("failed to create thread")
        return record

    async def get(self, thread_id: str) -> ThreadRecord | None:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                select thread_id, created_at, updated_at, status, metadata
                from app_threads
                where thread_id = $1
                """,
                uuid.UUID(thread_id),
            )
        return _thread_from_row(row)

    async def list(self) -> list[ThreadRecord]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                select thread_id, created_at, updated_at, status, metadata
                from app_threads
                order by updated_at desc
                """
            )
        return [record for row in rows if (record := _thread_from_row(row))]

    async def update_status(self, thread_id: str, status: str):
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                update app_threads
                set status = $2, updated_at = $3
                where thread_id = $1
                """,
                uuid.UUID(thread_id),
                status,
                datetime.utcnow(),
            )

    async def update_values(self, thread_id: str, values: dict[str, Any]):
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                update app_threads
                set metadata = metadata || $2::jsonb, updated_at = $3
                where thread_id = $1
                """,
                uuid.UUID(thread_id),
                json.dumps(values, ensure_ascii=False),
                datetime.utcnow(),
            )
