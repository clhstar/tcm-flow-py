import hashlib
import json
from typing import Any


def short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def get_tool_call_id(tool_call: dict[str, Any], index: int) -> str:
    return str(
        tool_call.get("id") or tool_call.get("tool_call_id") or f"tool_call_{index}"
    )


def build_tool_call_summary(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "retrieve_tcm_knowledge":
        query = args.get("query", "")
        mode = args.get("mode", "hybrid")
        return f"正在使用 {mode} 模式检索中医知识：{query}"

    if tool_name == "task":
        description = args.get("description", "")
        return f"正在委派动态子任务：{description[:120]}"

    if tool_name == "ask_clarification":
        questions = args.get("questions", [])
        if isinstance(questions, list):
            summary = "；".join(str(question) for question in questions)
        else:
            summary = str(questions)
        return f"正在生成澄清追问：{summary}"

    if tool_name == "present_files":
        return "正在展示生成文件。"

    return f"正在调用工具：{tool_name}"


def parse_retrieval_summary(content: str) -> dict[str, Any]:
    """
    从 retrieve_tcm_knowledge 工具返回文本中提取简要信息。
    """
    result = {
        "retrieval_mode": None,
        "original_query": None,
        "rewritten_query": None,
        "allowed_terms": [],
    }

    lines = content.splitlines()
    collecting_terms = False

    for line in lines:
        text = line.strip()

        if text.startswith("检索模式："):
            result["retrieval_mode"] = text.replace("检索模式：", "", 1).strip()

        elif text.startswith("原始检索问题："):
            result["original_query"] = text.replace("原始检索问题：", "", 1).strip()

        elif text.startswith("改写后检索问题："):
            result["rewritten_query"] = text.replace("改写后检索问题：", "", 1).strip()

        elif text.startswith("允许使用的专业术语"):
            collecting_terms = True
            continue

        elif collecting_terms and text.startswith("回答约束"):
            break

        elif collecting_terms and text.startswith("-"):
            term = text.replace("-", "", 1).strip()
            if term:
                result["allowed_terms"].append(term)

    result["allowed_terms"] = list(dict.fromkeys(result["allowed_terms"]))

    return result


def parse_subagent_result(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except Exception:
        return None

    if payload.get("type") != "subagent_result":
        return None

    return payload


def extract_trace_events_from_messages(
    messages: list[dict[str, Any]],
    emitted_keys: set[str],
) -> list[dict[str, Any]]:
    """
    从 messages 中提取“增量 agent_step 事件”。

    emitted_keys 用来防止每次 values 流重复推送同一个事件。
    """
    events: list[dict[str, Any]] = []

    for index, msg in enumerate(messages):
        msg_type = msg.get("type")

        # 1. AI 决定调用工具
        if msg_type == "ai" and msg.get("tool_calls"):
            for tool_index, tool_call in enumerate(msg.get("tool_calls", [])):
                tool_name = tool_call.get("name", "")
                args = tool_call.get("args") or {}
                call_id = get_tool_call_id(tool_call, tool_index)

                event_key = f"tool_call:{call_id}:{tool_name}"

                if event_key in emitted_keys:
                    continue

                emitted_keys.add(event_key)

                events.append(
                    {
                        "event_key": event_key,
                        "type": "tool_call",
                        "status": "started",
                        "agent": "lead_agent",
                        "tool": tool_name,
                        "tool_call_id": call_id,
                        "args": args,
                        "summary": build_tool_call_summary(tool_name, args),
                    }
                )

        # 2. 工具执行完成
        if msg_type == "tool":
            tool_name = msg.get("name", "")
            content = msg.get("content", "")

            if not isinstance(content, str):
                continue

            content_hash = short_hash(content[:500])
            event_key = f"tool_result:{index}:{tool_name}:{content_hash}"

            if event_key in emitted_keys:
                continue

            emitted_keys.add(event_key)

            if tool_name == "retrieve_tcm_knowledge":
                retrieval_summary = parse_retrieval_summary(content)

                events.append(
                    {
                        "event_key": event_key,
                        "type": "tool_result",
                        "status": "completed",
                        "agent": "lead_agent",
                        "tool": tool_name,
                        "summary": "中医知识检索完成。",
                        "retrieval": retrieval_summary,
                    }
                )

            elif tool_name == "task":
                payload = parse_subagent_result(content)

                if payload:
                    events.append(
                        {
                            "event_key": event_key,
                            "type": "subagent_result",
                            "status": "completed",
                            "agent": payload.get("agent_name", "dynamic_subagent"),
                            "tool": "task",
                            "task_description": payload.get("task_description", ""),
                            "expected_output": payload.get("expected_output", ""),
                            "needs_clarification": payload.get(
                                "needs_clarification",
                                False,
                            ),
                            "clarification_questions": payload.get(
                                "clarification_questions",
                                [],
                            ),
                            "summary": str(payload.get("content", ""))[:300],
                        }
                    )
                else:
                    events.append(
                        {
                            "event_key": event_key,
                            "type": "tool_result",
                            "status": "completed",
                            "agent": "lead_agent",
                            "tool": "task",
                            "summary": "动态子任务执行完成。",
                        }
                    )

            elif tool_name == "ask_clarification":
                events.append(
                    {
                        "event_key": event_key,
                        "type": "clarification",
                        "status": "completed",
                        "agent": "lead_agent",
                        "tool": tool_name,
                        "summary": content,
                    }
                )

            else:
                events.append(
                    {
                        "event_key": event_key,
                        "type": "tool_result",
                        "status": "completed",
                        "agent": "lead_agent",
                        "tool": tool_name,
                        "summary": f"工具 {tool_name} 执行完成。",
                    }
                )

    return events


def extract_agent_trace_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    从 messages 中提取最终 agent_trace。

    当前主要解析：
    - task 工具返回的 subagent_result
    - retrieve_tcm_knowledge 的检索记录
    """
    trace: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("type") != "tool":
            continue

        tool_name = msg.get("name")
        content = msg.get("content", "")

        if not isinstance(content, str):
            continue

        if tool_name == "retrieve_tcm_knowledge":
            retrieval = parse_retrieval_summary(content)

            trace.append(
                {
                    "agent": "lead_agent",
                    "action": "retrieve",
                    "tool": "retrieve_tcm_knowledge",
                    "summary": "完成中医知识检索。",
                    "retrieval": retrieval,
                }
            )

        elif tool_name == "task":
            payload = parse_subagent_result(content)

            if not payload:
                continue

            trace.append(
                {
                    "agent": payload.get("agent_name", "dynamic_subagent"),
                    "action": "complete",
                    "tool": "task",
                    "task_description": payload.get("task_description", ""),
                    "expected_output": payload.get("expected_output", ""),
                    "needs_clarification": payload.get(
                        "needs_clarification",
                        False,
                    ),
                    "clarification_questions": payload.get(
                        "clarification_questions",
                        [],
                    ),
                    "summary": str(payload.get("content", ""))[:300],
                }
            )

    return trace
