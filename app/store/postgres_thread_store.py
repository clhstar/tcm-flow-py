from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from app.store.models import ThreadRecord


def _isoformat(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _metadata_from_value(value) -> dict[str, Any]:
    """
    把数据库中的 metadata 字段转换成 Python dict。

    metadata 是 PostgreSQL jsonb 字段。

    asyncpg 取出来时可能有两种情况：
        1. 已经是 dict
        2. 是 JSON 字符串

    所以这里做兼容处理。
    """
    if value is None:
        return {}
    if isinstance(value, str):
        return dict(json.loads(value))
    return dict(value)


def _thread_from_row(row) -> ThreadRecord | None:
    """
    把数据库查询出来的一行 row 转换成 ThreadRecord。

    row:
        asyncpg 查询返回的 Record。

    返回：
        ThreadRecord 或 None。
    """
    if row is None:
        return None
    return ThreadRecord(
        thread_id=str(row["thread_id"]),
        created_at=_isoformat(row["created_at"]),
        updated_at=_isoformat(row["updated_at"]),
        status=row["status"],
        values=_metadata_from_value(row["metadata"]),
    )


def _message_role(message: dict[str, Any]) -> str | None:
    """
    把 LangChain / LangGraph 消息类型转换成前端常用 role。

    LangChain 消息 type:
        human
        ai
        system
        tool

    前端 / OpenAI 风格 role:
        user
        assistant
        system
        tool
    """
    msg_type = message.get("type")
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"
    if msg_type == "system":
        return "system"
    if msg_type == "tool":
        return "tool"
    return None


def _message_visible(message: dict[str, Any]) -> bool:
    """
    判断这条消息是否应该作为普通对话内容展示给前端。

    当前逻辑：
        只展示 human 和 ai 消息；
        必须有 content；
        如果 ai 消息带 tool_calls，则不展示。

    为什么隐藏 tool_calls？
        因为带 tool_calls 的 ai 消息通常是“我要调用工具”的中间消息，
        不适合作为最终聊天内容展示。

    会展示：
        用户输入
        AI 最终回答

    不展示：
        system prompt
        tool 返回内容
        AI 工具调用中间消息
    """
    return (
        message.get("type") in {"human", "ai"}
        and bool(message.get("content"))
        and not message.get("tool_calls")
    )


class PostgresThreadStore:
    """
    PostgreSQL 版 Thread 存储。

    主要负责：
        1. 创建 thread
        2. 查询 thread
        3. 列出 thread
        4. 更新 thread 状态
        5. 更新 thread metadata
        6. 保存 thread 下的 messages

    对应数据库表大概是：
        app_threads
        app_messages
    """
    def __init__(self, pool):
        """
        pool 是 asyncpg 连接池。

        它通常来自：
            create_pool_from_settings(settings)
        """
        self.pool = pool

    async def create(self) -> ThreadRecord:
        """
        创建一条新的对话线程。

        新线程默认：
            status = idle
            metadata = {}
        """
        now = datetime.now(timezone.utc)
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
        """
        根据 thread_id 查询一条线程。

        如果不存在，返回 None。
        """
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
        """
        查询所有 thread，按更新时间倒序排列。

        最新更新的 thread 排在最前面。
        """
        async with self.pool.acquire() as connection:
            rows = await connection.fetch("""
                select thread_id, created_at, updated_at, status, metadata
                from app_threads
                order by updated_at desc
                """)
        return [record for row in rows if (record := _thread_from_row(row))]

    async def update_status(self, thread_id: str, status: str):
        """
        更新 thread 的状态。

        常见状态可能有：
            idle
            running
            waiting_for_clarification
            failed

        当前代码没有做状态白名单校验。
        """
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                update app_threads
                set status = $2, updated_at = $3
                where thread_id = $1
                """,
                uuid.UUID(thread_id),
                status,
                datetime.now(timezone.utc),
            )

    async def update_values(
        self,
        thread_id: str,
        values: dict[str, Any],
        run_id: str | None = None,
    ):
        """
        更新 thread 的 metadata。

        values:
            要合并进 metadata 的新值。

        run_id:
            如果这次 values 来自某次 run，可以传 run_id。
            保存 messages 时会把 message 和 run 关联起来。

        特别逻辑：
            如果 values 里包含 messages，
            会同步刷新 app_messages 表。
        """
        thread_uuid = uuid.UUID(thread_id)
        run_uuid = uuid.UUID(run_id) if run_id else None
        now = datetime.now(timezone.utc)

        async with self.pool.acquire() as connection:
            # metadata = metadata || 新 values
            #
            # PostgreSQL jsonb 的 || 表示合并 JSON。
            # 如果 key 已存在，会被右侧新值覆盖。
            await connection.execute(
                """
                update app_threads
                set metadata = metadata || $2::jsonb, updated_at = $3
                where thread_id = $1
                """,
                thread_uuid,
                json.dumps(values, ensure_ascii=False),
                now,
            )

            # 如果 values 里有 messages，就同步替换 app_messages 表
            if "messages" in values:
                await self._replace_messages(
                    connection,
                    thread_id=thread_uuid,
                    run_id=run_uuid,
                    messages=values["messages"] or [],
                    created_at=now,
                )

    async def _replace_messages(
        self,
        connection,
        *,
        thread_id: uuid.UUID,
        run_id: uuid.UUID | None,
        messages: list[dict[str, Any]],
        created_at: datetime,
    ):
        """
        替换某个 thread 下的所有 messages。

        当前策略：
            先删除这个 thread 旧的所有消息；
            再把新的 messages 全量插入。

        这种方式简单，适合 LangGraph 每次保存完整 messages 状态的场景。
        """
        await connection.execute(
            "delete from app_messages where thread_id = $1",
            thread_id,
        )
        for ordinal, message in enumerate(messages):
            await connection.execute(
                """
                insert into app_messages (
                  thread_id, run_id, message_id, ordinal, type, role, name,
                  tool_call_id, content, tool_calls, visible, created_at
                )
                values (
                  $1, $2, $3, $4, $5, $6, $7,
                  $8, $9::jsonb, $10::jsonb, $11, $12
                )
                """,
                thread_id,
                run_id,
                message.get("id"),
                ordinal,
                message.get("type", "unknown"),
                _message_role(message),
                message.get("name"),
                message.get("tool_call_id"),
                json.dumps(message.get("content", ""), ensure_ascii=False),
                (
                    json.dumps(message.get("tool_calls"), ensure_ascii=False)
                    if message.get("tool_calls") is not None
                    else None
                ),
                _message_visible(message),
                created_at,
            )
