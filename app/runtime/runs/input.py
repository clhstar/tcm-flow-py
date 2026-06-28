from typing import Any


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    return str(content or "")


def normalize_graph_input(input_data: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    role_by_type = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }

    for message in input_data.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = role_by_type.get(str(message.get("type", "human")))
        if role is None:
            continue
        content = extract_text_from_content(message.get("content", ""))
        if content.strip():
            messages.append({"role": role, "content": content})

    return {"messages": messages}


def extract_user_text(input_data: dict[str, Any]) -> str:
    for message in input_data.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("type")
        if role in {"user", "human"}:
            return extract_text_from_content(message.get("content", "")).strip()
    return ""
