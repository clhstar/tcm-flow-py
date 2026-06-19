import uuid
from datetime import datetime, timezone

from app.store.models import RunRecord


def _isoformat(value) -> str:
    """
    把时间值转换成字符串。

    如果 value 有 isoformat() 方法，比如 datetime 对象，
    就调用 value.isoformat()。

    如果没有 isoformat() 方法，就直接转成 str。

    作用：
        数据库里取出来的 created_at / updated_at
        通常是 datetime 类型。

        但是 API 返回给前端时，一般希望是字符串格式。

    例如：
        datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
        ↓
        "2026-06-19T12:00:00+00:00"
    """
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _run_from_row(row) -> RunRecord | None:
    """
    把数据库查询出来的一行 row 转换成 RunRecord 对象。

    参数：
        row:
            asyncpg 查询返回的一行数据。
            类型一般是 asyncpg.Record。

    返回：
        RunRecord 或 None。

    如果 row 是 None，说明数据库没有查到对应记录。
    """
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
    """
    PostgreSQL 版 Run 管理器。

    它负责对 app_runs 表进行操作。

    主要能力：
        1. create() 创建一次 run
        2. get() 查询一次 run
        3. set_status() 更新 run 状态

    在系统中的作用：
        当用户对某个 thread 发起一次 Agent 执行时，
        系统会创建一条 run 记录，用于追踪这次任务的状态。
    """

    def __init__(self, pool):
        """
        初始化 RunManager。

        参数：
            pool:
                asyncpg 连接池。

        这个 pool 通常来自前面讲过的：

            create_pool_from_settings(settings)

        然后在 FastAPI lifespan 中挂到 app.state 或传入 store。
        """
        self.pool = pool

    async def create(self, thread_id: str, assistant_id: str) -> RunRecord:
        """
        创建一条新的 run 记录。

        参数：
            thread_id:
                当前 run 所属的对话线程 ID。

            assistant_id:
                当前 run 使用的 assistant ID。

        返回：
            RunRecord。

        新建 run 时：
            status 默认是 pending。
            input 默认是空 JSON。
            context 默认是空 JSON。
            created_at 和 updated_at 都是当前 UTC 时间。
        """
        now = datetime.now(timezone.utc)
        run_id = uuid.uuid4()
        async with self.pool.acquire() as connection:
            # 插入一条 app_runs 记录，并返回核心字段
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
        """
        根据 run_id 查询 run 记录。

        参数：
            run_id:
                run 的 UUID 字符串。

        返回：
            如果找到，返回 RunRecord。
            如果没找到，返回 None。
        """
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
        """
        更新 run 的状态。

        参数：
            run_id:
                要更新的 run ID。

            status:
                新状态，例如：
                    pending
                    running
                    completed
                    failed
                    cancelled

            error:
                错误信息。
                如果 run 失败，可以把异常信息写到 error 字段。
                如果正常完成，一般是 None。

        作用：
            Agent 执行过程中，可以不断更新 run 状态。

        示例：
            刚创建：
                pending

            开始运行：
                running

            成功完成：
                completed

            出错：
                failed，并写入 error
        """
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
                datetime.now(timezone.utc),
            )
