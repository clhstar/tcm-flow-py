from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.schemas import RunCreateRequest
from app.services import state, start_run

app = FastAPI(title="Mini DeerFlow")


@app.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "ok"}


@app.post("/api/threads")
async def create_thread():
    """创建新的会话线程"""
    thread = await state.thread_store.create()
    return {
        "thread_id": thread.thread_id,
        "created_at": thread.created_at,
    }


@app.get("/api/threads")
async def list_threads():
    """列出所有线程"""
    return list(state.thread_store.threads.values())


@app.post("/api/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, body: RunCreateRequest):
    """
    启动Agent运行并返回流式响应
    通过SSE方式实时推送Agent执行过程中的事件
    """
    record = await start_run(body, thread_id)

    return StreamingResponse(
        state.bridge.subscribe(record.run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Run-Id": record.run_id,
        },
    )


@app.get("/api/threads/{thread_id}")
async def get_thread(thread_id: str):
    """获取指定线程的完整信息"""
    thread = await state.thread_store.get(thread_id)
    if thread is None:
        return {"error": "Thread not found"}

    return {
        "thread_id": thread.thread_id,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
        "status": thread.status,
        "values": thread.values,
    }


@app.get("/api/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    """获取线程的对话历史和状态"""
    thread = await state.thread_store.get(thread_id)

    if thread is None:
        return {
            "error": "Thread not found"
        }

    return {
        "thread_id": thread.thread_id,
        "status": thread.status,
        "conversation": thread.values.get("conversation", []),
        "pending_clarification": thread.values.get("pending_clarification"),
        "messages": thread.values.get("messages", []),
    }