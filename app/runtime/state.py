from app.runtime.stream import StreamBridge
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

    def __init__(self):
        self.thread_store = ThreadStore()
        self.run_manager = RunManager()
        self.bridge = StreamBridge()


state = AppState()