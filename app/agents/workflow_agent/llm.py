from __future__ import annotations

from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import get_settings


StructuredSchema = TypeVar("StructuredSchema", bound=BaseModel)


def _uses_deepseek_backend(model_name: str | None, base_url: str | None) -> bool:
    model = (model_name or "").lower()
    url = (base_url or "").lower()
    return "deepseek" in model or "deepseek" in url


def _text_attr(obj: Any, *names: str) -> str | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, str):
            return value
    return None


def build_workflow_model(context: dict[str, Any] | None = None) -> ChatOpenAI:
    context = context or {}
    settings = get_settings()

    return ChatOpenAI(
        model=context.get("model_name") or settings.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=context.get("temperature", 0.2),
        streaming=context.get("streaming", False),
    )


def structured_model(model: Any, schema: type[StructuredSchema]) -> Any:
    settings = get_settings()
    model_name = _text_attr(model, "model_name", "model") or settings.openai_model
    base_url = _text_attr(model, "base_url", "openai_api_base") or settings.openai_base_url
    if _uses_deepseek_backend(model_name, base_url):
        return model.with_structured_output(
            method="json_mode",
        )

    return model.with_structured_output(
        schema,
        method="json_schema",
        strict=True,
    )
