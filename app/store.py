import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ThreadRecord:
    thread_id: str
    created_at: str
    updated_at: str
    status: str = "idle"
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    run_id: str
    thread_id: str
    assistant_id: str
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    task: asyncio.Task | None = None
    error: str | None = None


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


class RunManager:
    def __init__(self):
        self.runs: dict[str, RunRecord] = {}

    async def create(self, thread_id: str, assistant_id: str) -> RunRecord:
        run_id = str(uuid.uuid4())
        record = RunRecord(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        self.runs[run_id] = record
        return record

    async def set_status(self, run_id: str, status: str, error: str | None = None):
        record = self.runs.get(run_id)
        if record:
            record.status = status
            record.error = error
            record.updated_at = datetime.utcnow().isoformat()