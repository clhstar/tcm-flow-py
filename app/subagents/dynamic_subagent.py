import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic import model_validator

from app.middlewares.clarification_controller import normalize_question_items
from app.subagents.prompts import (
    DYNAMIC_SUBAGENT_SYSTEM_PROMPT,
    build_dynamic_subagent_user_prompt,
)

load_dotenv()


class SubAgentOutput(BaseModel):
    """
    Dynamic SubAgent 的模型输出协议。
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "needs_clarification"]
    clarification_questions: list[str] = Field(
        max_length=3,
        description=(
            "需要用户补充的问题正文，不要包含序号、项目符号或“问题1”等前缀"
        ),
    )
    content: str

    @model_validator(mode="before")
    @classmethod
    def validate_status_and_questions(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        status = value.get("status")
        if status not in {"completed", "needs_clarification"}:
            raise ValueError(
                "子 Agent status 只能是 completed 或 needs_clarification"
            )

        questions = value.get("clarification_questions")
        if not isinstance(questions, list) or any(
            not isinstance(question, str) for question in questions
        ):
            raise ValueError(
                "子 Agent clarification_questions 必须是字符串数组"
            )

        if status == "needs_clarification" and not 1 <= len(questions) <= 3:
            raise ValueError(
                "子 Agent needs_clarification 必须包含 1 到 3 个澄清问题"
            )

        if status == "completed" and questions:
            raise ValueError("子 Agent completed 状态不能包含澄清问题")

        return value

    @field_validator("clarification_questions")
    @classmethod
    def normalize_questions(cls, value: list[str]) -> list[str]:
        normalized = normalize_question_items(
            value,
            max_questions=max(len(value), 1),
        )

        if value and not normalized:
            raise ValueError("子 Agent 澄清问题不能为空")

        return normalized

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        return value.strip()


@dataclass
class DynamicSubAgentResult:
    """
    动态子智能体返回给执行器的结果。
    """

    task_description: str
    expected_output: str
    content: str
    metadata: dict[str, Any]
    status: str = "completed"
    clarification_questions: list[str] | None = None


def parse_subagent_output(
    output: str | dict[str, Any] | SubAgentOutput,
) -> DynamicSubAgentResult:
    """
    校验结构化协议，并转换为执行器使用的结果对象。
    """
    if isinstance(output, SubAgentOutput):
        parsed = output
    else:
        if isinstance(output, str):
            try:
                payload = json.loads(output)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "子 Agent 输出必须是合法的 JSON 对象"
                ) from exc
        elif isinstance(output, dict):
            payload = output
        else:
            raise ValueError("子 Agent 输出必须是合法的 JSON 对象")

        if not isinstance(payload, dict):
            raise ValueError("子 Agent 输出必须是合法的 JSON 对象")

        try:
            parsed = SubAgentOutput.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    return DynamicSubAgentResult(
        task_description="",
        expected_output="",
        content=parsed.content,
        metadata={},
        status=parsed.status,
        clarification_questions=parsed.clarification_questions,
    )


class DynamicSubAgent:
    """
    由 Lead Agent 通过 task 工具动态描述任务的临时子智能体。
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
        structured_model = model.with_structured_output(
            SubAgentOutput,
            method="json_mode",
        )

        user_prompt = build_dynamic_subagent_user_prompt(
            description=description,
            expected_output=expected_output,
        )

        response = await structured_model.ainvoke(
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

        parsed = parse_subagent_output(response)
        parsed.task_description = description
        parsed.expected_output = expected_output or ""
        parsed.metadata = {
            "subagent_type": "dynamic",
            "output_status": parsed.status,
            "structured_output_method": "json_mode",
        }

        return parsed
