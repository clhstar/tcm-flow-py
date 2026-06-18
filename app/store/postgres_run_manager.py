import uuid
from datetime import datetime

from app.store.models import RunRecord


def _isoformat(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _run_from_row(row) -> RunRecord | None:
    if row is None:
        return None
    return RunRecord(
        run_id=str(row["run_id"]),
        thread_id=str(row["thread_id"]),
        assistant_id=row["assistant_id"],
        status=row["status"],
        created_at=_isoformat(row["created_at"]),
        updated_at=_isoformat(row["updated_at"]),
        error=row["error"],
    )


class PostgresRunManager:
    def __init__(self, pool):
        self.pool = pool

    async def create(self, thread_id: str, assistant_id: str) -> RunRecord:
        now = datetime.utcnow()
        run_id = uuid.uuid4()
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                insert into app_runs (
                  run_id, thread_id, assistant_id, status, input, context, created_at, updated_at
                )
                values ($1, $2, $3, 'pending', '{}'::jsonb, '{}'::jsonb, $4, $4)
                returning run_id, thread_id, assistant_id, status, created_at, updated_at, error
                """,
                run_id,
                uuid.UUID(thread_id),
                assistant_id,
                now,
            )
        record = _run_from_row(row)
        if record is None:
            raise RuntimeError("failed to create run")
        return record

    async def get(self, run_id: str) -> RunRecord | None:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                select run_id, thread_id, assistant_id, status, created_at, updated_at, error
                from app_runs
                where run_id = $1
                """,
                uuid.UUID(run_id),
            )
        return _run_from_row(row)

    async def set_status(
        self,
        run_id: str,
        status: str,
        error: str | None = None,
    ):
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                update app_runs
                set status = $2, error = $3, updated_at = $4
                where run_id = $1
                """,
                uuid.UUID(run_id),
                status,
                error,
                datetime.utcnow(),
            )
