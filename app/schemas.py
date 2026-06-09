from pydantic import BaseModel, Field
from typing import Any, Literal


class MessageBlock(BaseModel):
    """消息块，用于结构化消息内容"""
    type: str = "text"
    text: str


class ChatMessage(BaseModel):
    """聊天消息结构，包含消息类型和内容"""
    type: Literal["human", "ai", "system", "tool"]
    content: str | list[MessageBlock]


class RunInput(BaseModel):
    """Agent运行的输入数据"""
    messages: list[ChatMessage]


class RunCreateRequest(BaseModel):
    """创建Agent运行的请求体"""
    assistant_id: str = "lead_agent"
    input: RunInput
    stream_mode: list[str] = Field(default_factory=lambda: ["values"])
    stream_subgraphs: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)