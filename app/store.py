import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ThreadRecord:
    """线程记录，存储会话线程的基本信息"""
    thread_id: str
    created_at: str
    updated_at: str
    status: str = "idle"
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    """运行记录，存储单次Agent执行的信息"""
    run_id: str
    thread_id: str
    assistant_id: str
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    task: asyncio.Task | None = None
    error: str | None = None


class ThreadStore:
    """线程存储管理，负责创建和操作会话线程"""

    def __init__(self):
        self.threads: dict[str, ThreadRecord] = {}

    async def create(self) -> ThreadRecord:
        """创建一个新的线程记录"""
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
        """根据线程ID获取线程记录"""
        return self.threads.get(thread_id)

    async def update_status(self, thread_id: str, status: str):
        """更新线程状态"""
        record = self.threads.get(thread_id)
        if record:
            record.status = status
            record.updated_at = datetime.utcnow().isoformat()

    async def update_values(self, thread_id: str, values: dict[str, Any]):
        """更新线程的values数据（如对话历史）"""
        record = self.threads.get(thread_id)
        if record:
            record.values.update(values)
            record.updated_at = datetime.utcnow().isoformat()


class RunManager:
    """运行管理器，负责创建和管理Agent执行记录"""

    def __init__(self):
        self.runs: dict[str, RunRecord] = {}

    async def create(self, thread_id: str, assistant_id: str) -> RunRecord:
        """创建一个新的运行记录"""
        run_id = str(uuid.uuid4())
        record = RunRecord(
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        self.runs[run_id] = record
        return record

    async def set_status(self, run_id: str, status: str, error: str | None = None):
        """更新运行状态及错误信息"""
        record = self.runs.get(run_id)
        if record:
            record.status = status
            record.error = error
            record.updated_at = datetime.utcnow().isoformat()