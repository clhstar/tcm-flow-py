import uuid
import asyncio
from datetime import datetime

from app.store.models import RunRecord


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

    async def get(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    async def set_status(
        self,
        run_id: str,
        status: str,
        error: str | None = None,
    ):
        record = self.runs.get(run_id)

        if record:
            record.status = status
            record.error = error
            record.updated_at = datetime.utcnow().isoformat()

    async def shutdown(self, timeout: float = 5.0) -> None:
        tasks = [
            record.task
            for record in self.runs.values()
            if record.task is not None and not record.task.done()
        ]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
