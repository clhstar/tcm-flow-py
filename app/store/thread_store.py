import uuid
from datetime import datetime
from typing import Any

from app.store.models import ThreadRecord


class ThreadStore:
    def __init__(self):
        self.threads: dict[str, ThreadRecord] = {}

    async def create(self) -> ThreadRecord:
        now = datetime.utcnow().isoformat()
        thread_id = str(uuid.uuid4())

        record = ThreadRecord(
            thread_id=thread_id,
            created_at=now,
            updated_at=now,
        )

        self.threads[thread_id] = record
        return record

    async def get(self, thread_id: str) -> ThreadRecord | None:
        return self.threads.get(thread_id)

    async def list(self) -> list[ThreadRecord]:
        return list(self.threads.values())

    async def update_status(self, thread_id: str, status: str):
        record = self.threads.get(thread_id)

        if record:
            record.status = status
            record.updated_at = datetime.utcnow().isoformat()

    async def update_values(self, thread_id: str, values: dict[str, Any]):
        record = self.threads.get(thread_id)

        if record:
            record.values.update(values)
            record.updated_at = datetime.utcnow().isoformat()