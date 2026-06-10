import traceback
from typing import Any

from app.agents.lead_agent import make_lead_agent
from app.store import RunManager, ThreadStore, RunRecord
from app.stream import StreamBridge

from app.guardrails.answer_validator import validate_answer
from app.guardrails.answer_rewriter import rewrite_answer


def extract_text_from_content(content: Any) -> str:
    """
    把 LangGraph / 前端传入的 content 统一转成字符串。
    content 可能是：
    1. 普通字符串
    2. [{"type": "text", "text": "..."}]
    """
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )

    return str(content or "")


def normalize_messages(input_data: dict[str, Any]) -> list[dict[str, str]]:
    """
    把前端传入的 messages 转成 create_agent 可以消费的 role/content 形式。
    这里只处理 human/ai/system，避免把 tool message 直接塞回模型导致格式问题。
    """
    result = []

    for msg in input_data.get("messages", []):
        msg_type = msg.get("type", "human")
        content = extract_text_from_content(msg.get("content", ""))

        if not content.strip():
            continue

        if msg_type == "human":
            result.append({"role": "user", "content": content})
        elif msg_type == "ai":
            result.append({"role": "assistant", "content": content})
        elif msg_type == "system":
            result.append({"role": "system", "content": content})

    return result


def extract_user_text(input_data: dict[str, Any]) -> str:
    """
    提取本轮用户输入文本。
    目前默认每次请求只有一条 human 消息。
    """
    for msg in input_data.get("messages", []):
        if msg.get("type") == "human":
            return extract_text_from_content(msg.get("content", "")).strip()

    return ""


# def build_agent_messages(
#     thread_values: dict[str, Any],
#     input_data: dict[str, Any],
# ) -> list[dict[str, str]]:
#     """
#     v0.4 核心：
#     从 thread.values.conversation 里读取历史对话，
#     再拼接本轮用户输入，形成完整上下文。
#     """
#     history = thread_values.get("conversation") or []

#     safe_history = []
#     for msg in history:
#         role = msg.get("role")
#         content = msg.get("content", "")

#         if role in {"user", "assistant", "system"} and content:
#             safe_history.append(
#                 {
#                     "role": role,
#                     "content": content,
#                 }
#             )

#     current_messages = normalize_messages(input_data)

#     return safe_history + current_messages


def message_to_dict(message: Any) -> dict[str, Any]:
    """
    把 LangChain / LangGraph message 对象转成可 JSON 序列化的 dict。
    这个用于调试完整 Agent 执行轨迹。
    """
    msg_type = getattr(message, "type", "ai")
    content = getattr(message, "content", "")

    if msg_type == "human":
        msg_type = "human"
    elif msg_type == "ai":
        msg_type = "ai"
    elif msg_type == "tool":
        msg_type = "tool"

    data = {
        "type": msg_type,
        "content": content,
    }

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = tool_calls

    name = getattr(message, "name", None)
    if name:
        data["name"] = name

    return data


def extract_final_ai_text(messages: list[dict]) -> str:
    """
    提取最后一条真正给用户看的 AI 回复。
    跳过带 tool_calls 的中间 AI 消息。
    """
    for msg in reversed(messages):
        if msg.get("type") == "ai" and msg.get("content"):
            if not msg.get("tool_calls"):
                return msg["content"]

    return ""


def extract_latest_clarification_question(messages: list[dict]) -> str:
    """
    只在 ask_clarification 工具已经执行完成后才中断。

    注意：
    不要在 AI message 的 tool_calls 阶段中断。
    因为 OpenAI 要求 assistant tool_calls 后面必须跟对应的 tool message。
    如果提前中断，下一轮会因为历史消息不完整而报 400。
    """
    if not messages:
        return ""

    latest = messages[-1]

    # 只检查最新消息是不是 ask_clarification 的 tool 结果
    if latest.get("type") == "tool" and latest.get("name") == "ask_clarification":
        content = latest.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def extract_allowed_terms_from_messages(messages: list[dict]) -> list[str]:
    """
    从最近一次 retrieve_tcm_knowledge 的 tool message 中解析 allowed_terms。

    当前 retrieve_tcm_knowledge 返回的是文本格式：
    允许使用的专业术语：
    - 胃胀
    - 嗳气
    ...
    回答约束：
    ...
    """
    for msg in reversed(messages):
        if msg.get("type") == "tool" and msg.get("name") == "retrieve_tcm_knowledge":
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            lines = content.splitlines()
            collecting = False
            terms = []

            for line in lines:
                text = line.strip()

                if text.startswith("允许使用的专业术语"):
                    collecting = True
                    continue

                if collecting and text.startswith("回答约束"):
                    break

                if collecting and text.startswith("-"):
                    term = text.replace("-", "", 1).strip()
                    if term:
                        terms.append(term)

            # 找最近一次有 allowed_terms 的检索结果
            if terms:
                return list(dict.fromkeys(terms))

    return []


def extract_latest_retrieval_evidence(messages: list[dict]) -> str:
    """
    提取最近一次 retrieve_tcm_knowledge 的完整工具返回内容。
    用于答案重写时提供证据。
    """
    for msg in reversed(messages):
        if msg.get("type") == "tool" and msg.get("name") == "retrieve_tcm_knowledge":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content

    return ""


def append_visible_messages(
    thread_values: dict[str, Any],
    user_text: str,
    assistant_text: str,
) -> list[dict[str, str]]:
    """
    把本轮用户输入和系统可见回复追加到 conversation。
    conversation 只保存用户可见内容，不保存 tool 调用细节。
    """
    conversation = list(thread_values.get("conversation") or [])

    if user_text:
        conversation.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

    if assistant_text:
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )

    return conversation


async def run_agent(
    bridge: StreamBridge,
    run_manager: RunManager,
    thread_store: ThreadStore,
    record: RunRecord,
    input_data: dict[str, Any],
    context: dict[str, Any],
):
    """
    异步执行Agent的核心函数
    1. 更新运行状态为running
    2. 调用make_lead_agent创建Agent实例
    3. 遍历Agent产生的消息流，实时推送事件
    4. 检测是否需要澄清（ask_clarification工具执行后中断）
    5. 执行完成后保存对话历史并更新状态
    """
    run_id = record.run_id
    thread_id = record.thread_id

    try:
        await run_manager.set_status(run_id, "running")
        await thread_store.update_status(thread_id, "running")

        await bridge.publish(
            run_id,
            "metadata",
            {
                "run_id": run_id,
                "thread_id": thread_id,
            },
        )

        thread = await thread_store.get(thread_id)
        thread_values = thread.values if thread else {}

        user_text = extract_user_text(input_data)

        agent = make_lead_agent(context)

        messages = normalize_messages(input_data)

        config = {
            "configurable": {
                "thread_id": thread_id,
            },
            "recursion_limit": context.get("recursion_limit", 50),
        }

        final_messages = []

        async for chunk in agent.astream(
            {"messages": messages},
            config=config,
            stream_mode="values",
        ):
            raw_messages = chunk.get("messages", [])
            final_messages = [message_to_dict(m) for m in raw_messages]

            await bridge.publish(
                run_id,
                "values",
                {
                    "messages": final_messages,
                },
            )

            clarification_question = extract_latest_clarification_question(
                final_messages
            )

            if clarification_question:
                conversation = append_visible_messages(
                    thread_values=thread_values,
                    user_text=user_text,
                    assistant_text=clarification_question,
                )

                await thread_store.update_values(
                    thread_id,
                    {
                        "messages": final_messages,
                        "conversation": conversation,
                        "pending_clarification": {
                            "run_id": run_id,
                            "question": clarification_question,
                        },
                    },
                )

                await bridge.publish(
                    run_id,
                    "clarification",
                    {
                        "question": clarification_question,
                        "thread_id": thread_id,
                        "run_id": run_id,
                    },
                )

                await run_manager.set_status(run_id, "interrupted")
                await thread_store.update_status(thread_id, "waiting")
                return
        final_text = extract_final_ai_text(final_messages)

        allowed_terms = extract_allowed_terms_from_messages(final_messages)
        evidence_text = extract_latest_retrieval_evidence(final_messages)

        validation_before = validate_answer(
            answer=final_text,
            allowed_terms=allowed_terms,
        )

        rewritten = False
        validation_after = validation_before

        if final_text and allowed_terms and not validation_before.get("passed"):
            unsupported_terms = validation_before.get("unsupported_terms", [])

            rewritten_text = await rewrite_answer(
                answer=final_text,
                allowed_terms=allowed_terms,
                unsupported_terms=unsupported_terms,
                evidence_text=evidence_text,
            )

            if rewritten_text:
                rewritten = True
                final_text = rewritten_text

                validation_after = validate_answer(
                    answer=final_text,
                    allowed_terms=allowed_terms,
                )

        conversation = append_visible_messages(
            thread_values=thread_values,
            user_text=user_text,
            assistant_text=final_text,
        )

        await thread_store.update_values(
            thread_id,
            {
                "messages": final_messages,
                "conversation": conversation,
                "pending_clarification": None,
                "last_validation": validation_after,
                "last_allowed_terms": allowed_terms,
                "last_rewritten": rewritten,
            },
        )

        await bridge.publish(
            run_id,
            "final",
            {
                "content": final_text,
                "messages": final_messages,
                "conversation": conversation,
                "validation": validation_after,
                "validation_before_rewrite": validation_before,
                "rewritten": rewritten,
            },
        )

        await run_manager.set_status(run_id, "success")
        await thread_store.update_status(thread_id, "idle")

    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        await run_manager.set_status(run_id, "error", error=error)
        await thread_store.update_status(thread_id, "error")
        await bridge.publish(
            run_id,
            "error",
            {
                "message": error,
            },
        )

    finally:
        await bridge.publish_end(run_id)
