import json
from typing import Any

from app.subagents.dynamic_subagent import DynamicSubAgent


class SubAgentExecutor:
    """
    子智能体执行器。

    V1.0 DeerFlow-like 版本：
    - 不依赖固定业务子智能体
    - 根据 Lead Agent 传入的 description 创建动态子任务
    - 返回结构化 JSON 字符串，供 Lead Agent 阅读
    """

    async def run(
        self,
        description: str,
        expected_output: str | None = None,
    ) -> str:
        subagent = DynamicSubAgent()

        result = await subagent.run(
            description=description,
            expected_output=expected_output,
        )

        payload: dict[str, Any] = {
            "type": "subagent_result",
            "agent_name": "dynamic_subagent",
            "task_description": result.task_description,
            "expected_output": result.expected_output,
            "content": result.content,
            "metadata": result.metadata,
            "status": result.status,
            "needs_clarification": result.status == "needs_clarification",
            "clarification_questions": result.clarification_questions or [],
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
