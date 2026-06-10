import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.subagents.prompts import (
    DYNAMIC_SUBAGENT_SYSTEM_PROMPT,
    build_dynamic_subagent_user_prompt,
)

load_dotenv()


@dataclass
class DynamicSubAgentResult:
    """
    动态子智能体返回结果。
    """

    task_description: str
    expected_output: str
    content: str
    metadata: dict[str, Any]


class DynamicSubAgent:
    """
    动态子智能体。

    这个类不代表固定业务角色。
    它是一个临时任务执行者，任务由 Lead Agent 通过 task 工具动态描述。
    """

    name = "dynamic_subagent"

    def get_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.2,
        )

    async def run(
        self,
        description: str,
        expected_output: str | None = None,
    ) -> DynamicSubAgentResult:
        model = self.get_model()

        user_prompt = build_dynamic_subagent_user_prompt(
            description=description,
            expected_output=expected_output,
        )

        response = await model.ainvoke(
            [
                {
                    "role": "system",
                    "content": DYNAMIC_SUBAGENT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
        )

        content = str(getattr(response, "content", "")).strip()

        return DynamicSubAgentResult(
            task_description=description,
            expected_output=expected_output or "",
            content=content,
            metadata={
                "subagent_type": "dynamic",
            },
        )