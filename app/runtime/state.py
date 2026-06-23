from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Checkpointer

from app.config import AppSettings, get_settings
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

    def __init__(
        self,
        *,
        thread_store,
        run_manager,
        bridge: StreamBridge,
        checkpointer: Checkpointer,
    ):
        self.thread_store = thread_store
        self.run_manager = run_manager
        self.bridge = bridge
        self.checkpointer = checkpointer


def build_state(
    pool=None,
    checkpointer: Checkpointer | None = None,
    settings: AppSettings | None = None,
) -> AppState:
    settings = settings or get_settings()
    checkpointer = checkpointer or InMemorySaver()
    if settings.checkpoint_backend == "postgres":
        if pool is None:
            raise ValueError("Postgres runtime state requires a database pool")
        return AppState(
            thread_store=PostgresThreadStore(pool),
            run_manager=PostgresRunManager(pool),
            bridge=StreamBridge(),
            checkpointer=checkpointer,
        )
    return AppState(
        thread_store=ThreadStore(),
        run_manager=RunManager(),
        bridge=StreamBridge(),
        checkpointer=checkpointer,
    )


def _memory_state() -> AppState:
    return AppState(
        thread_store=ThreadStore(),
        run_manager=RunManager(),
        bridge=StreamBridge(),
        checkpointer=InMemorySaver(),
    )


state = _memory_state()


def configure_state(
    pool=None,
    checkpointer: Checkpointer | None = None,
    settings: AppSettings | None = None,
) -> AppState:
    configured = build_state(
        pool=pool,
        checkpointer=checkpointer,
        settings=settings,
    )
    state.thread_store = configured.thread_store
    state.run_manager = configured.run_manager
    state.bridge = configured.bridge
    state.checkpointer = configured.checkpointer
    return state


def reset_state_to_memory() -> AppState:
    configured = _memory_state()
    state.thread_store = configured.thread_store
    state.run_manager = configured.run_manager
    state.bridge = configured.bridge
    state.checkpointer = configured.checkpointer
    return state
