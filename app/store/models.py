import asyncio
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