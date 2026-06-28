import asyncio
import json
from typing import Any


class StreamBridge:
    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}

    def create(self, run_id: str):
        self.queues[run_id] = asyncio.Queue()

    async def cleanup(self, run_id: str, delay: float = 60):
        if delay > 0:
            await asyncio.sleep(delay)
        self.queues.pop(run_id, None)

    async def publish(self, run_id: str, event: str, data: Any):
        queue = self.queues.get(run_id)

        if queue:
            await queue.put((event, data))

    async def publish_end(self, run_id: str):
        queue = self.queues.get(run_id)

        if queue:
            await queue.put(("end", {"status": "done"}))

    async def subscribe(self, run_id: str):
        queue = self.queues.get(run_id)

        if queue is None:
            return

        while True:
            event, data = await queue.get()

            yield self.format_sse(event, data)

            if event == "end":
                break

    @staticmethod
    def format_sse(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
