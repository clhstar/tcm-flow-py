from __future__ import annotations

import json
import re
from typing import Any, Generic, TypeVar
from venv import logger

from pydantic import BaseModel

from app.agents.workflow_agent.llm import structured_model
from app.agents.workflow_agent.models import InquiryState, KnownFacts
from app.agents.workflow_agent.prompts import compact_json


SchemaT = TypeVar("SchemaT", bound=BaseModel)
JSON_OUTPUT_INSTRUCTION = (
    "Return only a valid JSON object that conforms to the requested schema."
)
ITEM_SPLIT_PATTERN = re.compile(r"[、,，;；]\s*")


def _text_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [
            item.strip()
            for item in ITEM_SPLIT_PATTERN.split(value)
            if item.strip()
        ]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_inquiry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    top_level_risk_flags = result.pop("risk_flags", None)

    known_facts = result.get("known_facts")
    if isinstance(known_facts, dict):
        known_facts = dict(known_facts)
    elif known_facts:
        known_facts = {"associated_symptoms": _text_items(known_facts)}
    else:
        known_facts = {}

    risk_flags = [
        *_text_items(known_facts.get("risk_flags")),
        *_text_items(top_level_risk_flags),
    ]
    if risk_flags:
        known_facts["risk_flags"] = risk_flags

    known_allowed = set(KnownFacts.model_fields)
    result["known_facts"] = {
        key: value for key, value in known_facts.items() if key in known_allowed
    }

    for field_name in ("missing_info", "clarification_questions"):
        if isinstance(result.get(field_name), str):
            result[field_name] = _text_items(result[field_name])

    allowed = set(InquiryState.model_fields)
    return {key: value for key, value in result.items() if key in allowed}


def _normalize_structured_response(
    schema: type[SchemaT],
    response: Any,
) -> Any:
    if isinstance(response, str):
        response = json.loads(response)

    if schema is InquiryState and isinstance(response, dict):
        return _normalize_inquiry_payload(response)

    return response


class StructuredWorkflowComponent(Generic[SchemaT]):
    schema: type[SchemaT]
    system_prompt: str

    def __init__(self, model: Any) -> None:
        self._structured = structured_model(model, self.schema)

    async def invoke_structured(self, payload: dict[str, Any]) -> SchemaT:
        schema_json = compact_json(self.schema.model_json_schema())
        logger.info(f"Invoking structured component with schema: {schema_json}")
        response = await self._structured.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{self.system_prompt}\n\n"
                        f"{JSON_OUTPUT_INSTRUCTION}\n"
                        f"JSON schema: {schema_json}"
                    ),
                },
                {"role": "user", "content": compact_json(payload)},
            ]
        )
        return self.schema.model_validate(
            _normalize_structured_response(self.schema, response)
        )
