from typing import Any

from app.tools.builtins.clarification_tool import ask_clarification
from app.tools.builtins.present_file_tool import present_files
from app.tools.builtins.retrieval_tool import retrieve_tcm_knowledge
from app.tools.builtins.task_tool import task


def get_available_tools(context: dict[str, Any] | None = None):
    """
    工具注册中心。

    对齐 DeerFlow 的 tools/tools.py：
    根据 context 动态决定加载哪些工具。
    """

    context = context or {}

    tools = [
        ask_clarification,
        retrieve_tcm_knowledge,
        present_files,
    ]

    subagent_enabled = context.get("subagent_enabled", False)

    if subagent_enabled:
        tools.append(task)

    return tools