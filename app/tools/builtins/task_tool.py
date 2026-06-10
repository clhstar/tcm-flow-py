from langchain.tools import tool


@tool("task")
def task(description: str, agent_name: str = "tcm_subagent") -> str:
    """
    将复杂任务委派给子智能体处理。

    当前 V0.9 只做 DeerFlow-like 架构占位。
    V1.0 再实现真实 SubAgentExecutor。
    """
    return f"子任务已接收：agent={agent_name}, description={description}"