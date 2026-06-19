import asyncpg

from app.config import AppSettings


def normalize_asyncpg_dsn(dsn: str) -> str:
    """
    规范化 PostgreSQL 连接字符串。

    为什么需要这个函数？

    有些框架，比如 SQLAlchemy async 写法，数据库 URL 常写成：

        postgresql+asyncpg://user:password@localhost:5432/dbname

    但是 asyncpg 原生库需要的是：

        postgresql://user:password@localhost:5432/dbname

    所以这里要把：

        postgresql+asyncpg://

    转换成：

        postgresql://

    如果传进来的 dsn 本来就是 postgresql://，
    那就原样返回。
    """
    prefix = "postgresql+asyncpg://"
    if dsn.startswith(prefix):
        return "postgresql://" + dsn[len(prefix) :]
    return dsn


async def create_pool_from_settings(settings: AppSettings):
    """
    根据应用配置创建 PostgreSQL 异步连接池。

    参数：
        settings:
            AppSettings 配置对象。
            里面包含：
                database_url
                postgres_pool_size

    返回：
        asyncpg.Pool 对象。

    作用：
        创建一个数据库连接池，后续服务可以复用这个 pool
        来执行数据库查询、保存 checkpoint、读取 RAG 数据等。
    为什么要创建连接池？

    如果每次访问数据库都重新建立连接：

    请求来了 → 新建数据库连接 → 查询 → 关闭连接

    性能会很差。

    连接池的方式是：

    服务启动时创建一批连接
      ↓
    请求来了，从池子里借一个连接
      ↓
    用完归还
      ↓
    下次请求继续复用

    优点是：

    减少连接创建开销
    提高并发能力
    统一管理数据库连接
    服务关闭时统一释放
    """
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres persistence")
    return await asyncpg.create_pool(
        dsn=normalize_asyncpg_dsn(settings.database_url),
        min_size=1,
        max_size=settings.postgres_pool_size,
        command_timeout=60,
    )
