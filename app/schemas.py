from pydantic import BaseModel, Field
from typing import Any, Literal


class MessageBlock(BaseModel):
    type: str = "text"
    text: str


class ChatMessage(BaseModel):
    type: Literal["human", "ai", "system", "tool"]
    content: str | list[MessageBlock]


class RunInput(BaseModel):
    messages: list[ChatMessage]


class RunCreateRequest(BaseModel):
    assistant_id: str = "lead_agent"
    input: RunInput
    stream_mode: list[str] = Field(default_factory=lambda: ["values"])
    stream_subgraphs: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)