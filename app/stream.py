import asyncio
import json
from typing import Any


class StreamBridge:
    """流式事件桥接器，用于Agent执行过程中的实时事件推送"""

    def __init__(self):
        self.queues: dict[str, asyncio.Queue] = {}

    def create(self, run_id: str):
        """为特定运行创建事件队列"""
        self.queues[run_id] = asyncio.Queue()

    async def publish(self, run_id: str, event: str, data: Any):
        """发布事件到运行队列"""
        queue = self.queues.get(run_id)
        if queue:
            await queue.put((event, data))

    async def publish_end(self, run_id: str):
        """发布结束事件"""
        queue = self.queues.get(run_id)
        if queue:
            await queue.put(("end", {"status": "done"}))

    async def subscribe(self, run_id: str):
        """订阅运行事件流，以SSE格式yield事件"""
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
        """格式化Server-Sent Events数据"""
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"