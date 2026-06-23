import traceback
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
import logging

from app.middlewares.guardrail_middleware import apply_guardrails
from app.middlewares.trace_middleware import (
    extract_agent_trace_from_messages,
    extract_trace_events_from_messages,
)

from app.runtime.public_messages import (
    build_chat_response,
    extract_latest_assistant_message,
    extract_pending_clarification,
)
from app.runtime.serialization import serialize, serialize_message
from app.runtime.stream import StreamBridge
from app.store.models import RunRecord
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore

from app.middlewares.clarification_controller import (
    extract_latest_clarification_question,
)

logger = logging.getLogger(__name__)


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

    注意：
    这里只处理 human / ai / system。
    不要把 tool message 手动塞回模型，否则容易破坏 LangGraph checkpointer 中的工具调用链。
    """
    result = []

    for msg in input_data.get("messages", []):
        msg_type = msg.get("type", "human")
        content = extract_text_from_content(msg.get("content", ""))

        if not content.strip():
            continue

        if msg_type == "human":
            result.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        elif msg_type == "ai":
            result.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
        elif msg_type == "system":
            result.append(
                {
                    "role": "system",
                    "content": content,
                }
            )

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


def message_to_dict(message: Any) -> dict[str, Any]:
    """
    把 LangChain / LangGraph message 对象转成可 JSON 序列化的 dict。
    这个用于调试完整 Agent 执行轨迹。
    """
    return serialize_message(message)


def _normalize_stream_modes(modes: Any) -> list[str]:
    """Keep stream_mode as a list for LangGraph multi-mode streaming."""
    if modes is None:
        return ["messages"]
    if isinstance(modes, str):
        return [modes]
    if isinstance(modes, (list, tuple, set)):
        return [str(mode) for mode in modes]
    return ["messages"]


def _ensure_internal_stream_modes(modes: list[str]) -> list[str]:
    """Ensure internal LangGraph streaming always includes messages and values."""
    result: list[str] = []
    for mode in ("messages", "values", *modes):
        if mode not in result:
            result.append(mode)
    return result


def _split_stream_chunk(chunk: Any) -> tuple[str, Any]:
    """兼容 LangGraph 单模式 values 与多模式 (event, chunk) 两种标准输出。"""
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
        return chunk[0], chunk[1]

    return "values", chunk


def extract_final_ai_text(messages: list[dict[str, Any]]) -> str:
    """
    提取最后一条真正给用户看的 AI 回复。
    跳过带 tool_calls 的中间 AI 消息。
    """
    for msg in reversed(messages):
        if msg.get("type") == "ai" and msg.get("content"):
            if not msg.get("tool_calls"):
                return str(msg["content"])

    return ""


def append_visible_messages(
    thread_values: dict[str, Any],
    user_text: str,
    assistant_text: str,
) -> list[dict[str, str]]:
    """
    把本轮用户输入和系统可见回复追加到 conversation。

    conversation 只保存用户可见内容：
    - 用户输入
    - clarification 问题
    - final answer

    不保存 tool 调用细节；完整 Agent messages 由 LangGraph checkpointer 管理。
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


async def replace_final_ai_message_in_checkpoint(
    agent: Any,
    config: dict[str, Any],
    final_messages: list[dict[str, Any]],
    final_text: str,
) -> list[dict[str, Any]]:
    """
    使用相同 message id 覆盖最终 AIMessage，避免未通过 Guardrail 的原文
    留在 LangGraph checkpointer 中并污染下一轮对话。
    """
    target = next(
        (
            message
            for message in reversed(final_messages)
            if message.get("type") == "ai"
            and message.get("content")
            and not message.get("tool_calls")
        ),
        None,
    )

    if not target or target.get("content") == final_text:
        return final_messages

    message_id = target.get("id")
    if not message_id:
        raise RuntimeError("无法写回 Guardrail 答案：最终 AIMessage 缺少 id")

    await agent.aupdate_state(
        config,
        {
            "messages": [
                AIMessage(
                    id=message_id,
                    content=final_text,
                )
            ]
        },
    )

    snapshot = await agent.aget_state(config)
    return [message_to_dict(message) for message in snapshot.values.get("messages", [])]


async def _checkpoint_message_count(agent: Any, config: dict[str, Any]) -> int:
    aget_state = getattr(agent, "aget_state", None)
    if aget_state is None:
        return 0
    snapshot = await aget_state(config)
    return len(snapshot.values.get("messages", []))


async def run_agent(
    bridge: StreamBridge,
    run_manager: RunManager,
    thread_store: ThreadStore,
    record: RunRecord,
    agent_factory: Callable[[dict[str, Any] | None], Any],
    input_data: dict[str, Any],
    context: dict[str, Any],
):
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
                "assistant_id": record.assistant_id,
                "architecture": "tcm-flow",
                "stream_protocol": "messages-v1",
            },
        )

        thread = await thread_store.get(thread_id)
        thread_values = thread.values if thread else {}

        user_text = extract_user_text(input_data)

        agent = agent_factory(context)

        # 每次请求只传本轮用户消息，历史上下文交给 LangGraph checkpointer 管理
        messages = normalize_messages(input_data)

        config = {
            "configurable": {
                "thread_id": thread_id,
            },
            "recursion_limit": context.get("recursion_limit", 50),
        }
        message_start_index = await _checkpoint_message_count(agent, config)

        ##TODO
        emit_debug_events = bool(context.get("debug_events"))

        requested_stream_modes = _normalize_stream_modes(
            context.get("stream_mode", ["messages"])
        )
        # 内部固定订阅 messages + values；请求方额外需要的 updates 等模式继续保留。
        internal_stream_modes = _ensure_internal_stream_modes(requested_stream_modes)
        publish_values = "values" in requested_stream_modes or emit_debug_events

        final_messages: list[dict[str, Any]] = []
        emitted_trace_keys: set[str] = set()
        clarification_to_emit = ""

        logger.info(
            "Run %s: streaming with modes %s (requested: %s)",
            run_id,
            internal_stream_modes,
            context.get("stream_mode"),
        )

        async for stream_chunk in agent.astream(
            {"messages": messages},
            config=config,
            stream_mode=internal_stream_modes,
        ):

            stream_event, chunk = _split_stream_chunk(stream_chunk)

            if stream_event == "messages":
                await bridge.publish(
                    run_id,
                    "messages",
                    serialize(chunk, mode="messages"),
                )
                continue

            if stream_event != "values":
                update_payload = serialize(chunk, mode=stream_event)
                if stream_event != "updates":
                    update_payload = {
                        "stream_event": stream_event,
                        "data": update_payload,
                    }
                await bridge.publish(run_id, "updates", update_payload)
                continue

            serialized_values = serialize(chunk, mode="values")
            raw_messages = (
                serialized_values.get("messages", [])
                if isinstance(serialized_values, dict)
                else []
            )
            final_messages = [
                message if isinstance(message, dict) else message_to_dict(message)
                for message in raw_messages
            ]

            current_run_messages = final_messages[message_start_index:]

            if publish_values:
                await bridge.publish(run_id, "values", serialized_values)

            trace_events = []
            if emit_debug_events:
                trace_events = extract_trace_events_from_messages(
                    messages=current_run_messages,
                    emitted_keys=emitted_trace_keys,
                )

            for trace_event in trace_events:
                await bridge.publish(
                    run_id,
                    "updates",
                    trace_event,
                )

            # V0.9：澄清中断逻辑交给 middleware
            clarification_question = extract_latest_clarification_question(
                final_messages
            )

            if clarification_question:
                clarification_to_emit = clarification_question

        if clarification_to_emit:
            current_run_messages = final_messages[message_start_index:]
            pending_clarification = extract_pending_clarification(current_run_messages)
            assistant_message = (
                extract_latest_assistant_message(current_run_messages)
                or clarification_to_emit
            )
            conversation = append_visible_messages(
                thread_values=thread_values,
                user_text=user_text,
                assistant_text=assistant_message,
            )

            await thread_store.update_values(
                thread_id,
                {
                    "messages": final_messages,
                    "conversation": conversation,
                },
                run_id=run_id,
            )

            await bridge.publish(
                run_id,
                "final",
                build_chat_response(
                    thread_id=thread_id,
                    run_id=run_id,
                    status="need_clarification",
                    assistant_message=assistant_message,
                    pending_clarification=pending_clarification,
                ),
            )

            await run_manager.set_status(run_id, "waiting_clarification")
            await thread_store.update_status(thread_id, "waiting")
            return

        final_text = extract_final_ai_text(final_messages)
        original_final_text = final_text

        if emit_debug_events:
            await bridge.publish(
                run_id,
                "updates",
                {
                    "type": "guardrail",
                    "status": "started",
                    "agent": "guardrail_middleware",
                    "summary": "正在进行术语一致性校验与答案安全检查。",
                },
            )
        # V0.9：V0.8 术语校验与答案重写逻辑抽成 middleware
        guardrail_result = await apply_guardrails(
            final_text=final_text,
            messages=final_messages,
        )

        if emit_debug_events:
            await bridge.publish(
                run_id,
                "updates",
                {
                    "type": "guardrail",
                    "status": "completed",
                    "agent": "guardrail_middleware",
                    "summary": "术语一致性校验完成。",
                    "validation": guardrail_result.get("validation"),
                    "rewritten": guardrail_result.get("rewritten"),
                },
            )

        final_text = guardrail_result["final_text"]
        validation = guardrail_result["validation"]
        rewritten = guardrail_result["rewritten"]
        allowed_terms = guardrail_result["allowed_terms"]

        if final_text != original_final_text:
            final_messages = await replace_final_ai_message_in_checkpoint(
                agent=agent,
                config=config,
                final_messages=final_messages,
                final_text=final_text,
            )

        agent_trace = extract_agent_trace_from_messages(
            final_messages[message_start_index:]
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
                "last_validation": validation,
                "last_allowed_terms": allowed_terms,
                "last_rewritten": rewritten,
                "last_agent_trace": agent_trace,
            },
            run_id=run_id,
        )

        await bridge.publish(
            run_id,
            "final",
            build_chat_response(
                thread_id=thread_id,
                run_id=run_id,
                status="completed",
                assistant_message=final_text,
            ),
        )

        await run_manager.set_status(run_id, "success")
        await thread_store.update_status(thread_id, "idle")

    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()

        await run_manager.set_status(
            run_id,
            "error",
            error=error,
        )
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
