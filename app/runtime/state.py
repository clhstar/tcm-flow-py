from app.config import get_settings
from app.runtime.stream import StreamBridge
from app.store.postgres_run_manager import PostgresRunManager
from app.store.postgres_thread_store import PostgresThreadStore
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


class AppState:
    """
    Runtime 全局状态。

    对齐 DeerFlow 的运行时上下文思想：
    - thread_store 管理会话
    - run_manager 管理运行任务
    - bridge 管理 SSE 流
    """

    def __init__(self, *, thread_store, run_manager, bridge: StreamBridge):
        self.thread_store = thread_store
        self.run_manager = run_manager
        self.bridge = bridge


def build_state(pool=None) -> AppState:
    settings = get_settings()
    if settings.checkpoint_backend == "postgres":
        if pool is None:
            raise ValueError("Postgres runtime state requires a database pool")
        return AppState(
            thread_store=PostgresThreadStore(pool),
            run_manager=PostgresRunManager(pool),
            bridge=StreamBridge(),
        )
    return AppState(
        thread_store=ThreadStore(),
        run_manager=RunManager(),
        bridge=StreamBridge(),
    )


def _memory_state() -> AppState:
    return AppState(
        thread_store=ThreadStore(),
        run_manager=RunManager(),
        bridge=StreamBridge(),
    )


state = _memory_state()


def configure_state(pool=None) -> AppState:
    configured = build_state(pool=pool)
    state.thread_store = configured.thread_store
    state.run_manager = configured.run_manager
    state.bridge = configured.bridge
    return state
