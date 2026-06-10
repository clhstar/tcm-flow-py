from langchain.tools import tool

from app.subagents.executor import SubAgentExecutor


@tool("task")
async def task(
    description: str,
    expected_output: str = "",
) -> str:
    """
    将复杂任务委派给动态子智能体处理。

    参数：
    - description：子任务描述。必须包含任务目标、必要上下文和限制条件。
    - expected_output：期望输出格式。例如“用要点输出”“输出表格形式”“输出证据摘要”。

    适用场景：
    - 任务较复杂，需要先整理信息
    - 需要对子问题进行独立分析
    - 需要基于检索证据提取要点
    - 需要生成中间分析结果供 Lead Agent 综合

    注意：
    子智能体结果只是辅助分析，最终回答仍由 Lead Agent 综合生成。
    """
    executor = SubAgentExecutor()

    return await executor.run(
        description=description,
        expected_output=expected_output,
    )