import json
from collections.abc import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from app.middlewares.clarification_controller import (
    extract_latest_clarification_question,
    format_clarification_questions,
    normalize_question_items,
)


class ClarificationState(AgentState):
    pass


class ClarificationMiddleware(AgentMiddleware[ClarificationState]):
    state_schema = ClarificationState

    def normalize_questions(self, value) -> list[str]:
        # 兼容部分模型把数组输出成 JSON 字符串
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("澄清问题必须是JSON数组") from exc

        if not isinstance(value, list):
            raise ValueError("澄清问题必须是数组")

        non_empty_values = [
            question for question in value if str(question).strip()
        ]

        if len(non_empty_values) > 3:
            raise ValueError("澄清问题不能多于3个")

        questions = normalize_question_items(value, max_questions=3)

        if not questions:
            raise ValueError("澄清问题不能为空")

        return questions

    def format_questions(self, questions: list[str]) -> str:
        return format_clarification_questions(questions)

    def handle(self, request: ToolCallRequest) -> Command:
        tool_call = request.tool_call
        args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        questions = self.normalize_questions(args.get("questions"))
        content = self.format_questions(questions)

        message = ToolMessage(
            id=f"clarification:{tool_call_id}",
            name="ask_clarification",
            tool_call_id=tool_call_id,
            content=content,
        )

        return Command(
            update={"messages": [message]},
            goto=END,
        )

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: ClarificationState, runtime):
        messages = state.get("messages", [])

        if not messages:
            return None

        latest = messages[-1]
        if not isinstance(latest, ToolMessage) or latest.name != "task":
            return None

        question = extract_latest_clarification_question(
            [
                {
                    "type": "tool",
                    "name": latest.name,
                    "content": latest.content,
                }
            ]
        )

        if question:
            return {"jump_to": "end"}

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable,
    ):
        if request.tool_call.get("name") == "ask_clarification":
            return self.handle(request)

        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable,
    ):
        if request.tool_call.get("name") == "ask_clarification":
            return self.handle(request)

        return await handler(request)
