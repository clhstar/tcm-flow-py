# Runtime Worker Boundary Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `app/runtime/runs/worker.py` into a DeerFlow-aligned lifecycle coordinator while preserving the current FastAPI request contract, SSE events, clarification/final payloads, guardrail checkpoint rewrite, and visible conversation projection.

**Architecture:** Introduce explicit run context and input modules, a generic LangGraph stream adapter, and a TCM completion projection. `worker.py` coordinates those units and owns statuses, error/cancellation handling, terminal publication, and cleanup without parsing messages or constructing business responses.

**Tech Stack:** Python 3.10, FastAPI, LangChain messages, LangGraph async streaming/checkpoint APIs, `unittest`, `unittest.mock`, dataclasses, asyncio.

---

## Working Tree Safety

The worktree already contains staged changes in files that this refactor must preserve, including `app/runtime/runs/worker.py`, `tests/gateway/test_threads_router.py`, `tests/test_workflow_agent_flow.py`, and `tests/test_collaboration_history.py`.

Before each task:

```powershell
git status --short
git diff -- <task-files>
git diff --cached -- <task-files>
```

Use `git commit --only <task-files> ...` for task commits. Never use `git reset`, `git checkout --`, or a broad `git commit -a`. Do not include `app/agents/workflow_agent/agent.py` or `docs/superpowers/plans/2026-06-28-tcm-web-multi-agent-collaboration.md` unless the user separately requests it.

## Target File Map

- Create `app/runtime/runs/context.py`: `RunContext`, runtime context construction, runnable config construction.
- Create `app/runtime/runs/input.py`: HTTP-shaped message normalization and current user text extraction.
- Create `app/runtime/runs/stream_adapter.py`: stream-mode normalization, LangGraph item unpacking, forwarding, and latest snapshot collection.
- Create `app/runtime/runs/projection.py`: current-run message scoping, debug trace projection, clarification/guardrail completion, checkpoint rewrite, and `CompletionResult`.
- Modify `app/runtime/public_messages.py`: host visible conversation and final-assistant projection helpers.
- Modify `app/runtime/runs/worker.py`: retain lifecycle orchestration only.
- Modify `app/runtime/services.py`: prepare graph input, request config, `RunContext`, and stream modes for the worker.
- Modify `app/runtime/stream.py`: delayed queue cleanup.
- Create `tests/test_runtime_run_context.py`: input/context/config unit tests.
- Create `tests/test_runtime_stream_adapter.py`: generic stream adapter unit tests.
- Create `tests/test_runtime_completion_projection.py`: completion semantics unit tests.
- Create `tests/test_runtime_worker_boundary.py`: static worker responsibility enforcement.
- Create `tests/test_runtime_worker_lifecycle.py`: error, cancellation, terminal, and cleanup tests.
- Modify existing runtime/gateway tests to call the new internal `run_agent` signature and patch the new guardrail owner.

---

### Task 1: Extract Graph Input and Runtime Context Construction

**Files:**
- Create: `app/runtime/runs/input.py`
- Create: `app/runtime/runs/context.py`
- Create: `tests/test_runtime_run_context.py`

- [ ] **Step 1: Write failing input and context tests**

Create `tests/test_runtime_run_context.py`:

```python
import unittest

from app.runtime.runs.context import (
    RunContext,
    build_runnable_config,
    build_runtime_context,
)
from app.runtime.runs.input import extract_user_text, normalize_graph_input
from app.store.models import RunRecord
from app.store.thread_store import ThreadStore


class RuntimeRunInputTests(unittest.TestCase):
    def test_normalize_graph_input_preserves_supported_roles_and_text_blocks(self):
        graph_input = normalize_graph_input(
            {
                "messages": [
                    {
                        "type": "human",
                        "content": [
                            {"type": "text", "text": "first "},
                            {"type": "text", "text": "question"},
                        ],
                    },
                    {"type": "ai", "content": "prior answer"},
                    {"type": "system", "content": "system rule"},
                    {"type": "tool", "content": "do not replay"},
                    {"type": "human", "content": "   "},
                ]
            }
        )

        self.assertEqual(
            graph_input,
            {
                "messages": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "prior answer"},
                    {"role": "system", "content": "system rule"},
                ]
            },
        )

    def test_extract_user_text_keeps_current_compatibility(self):
        self.assertEqual(
            extract_user_text(
                {
                    "messages": [
                        {"type": "human", "content": "current question"},
                        {"type": "ai", "content": "ignored"},
                    ]
                }
            ),
            "current question",
        )
        self.assertEqual(
            extract_user_text(
                {"messages": [{"role": "user", "content": "normalized"}]}
            ),
            "normalized",
        )


class RuntimeRunContextTests(unittest.TestCase):
    def setUp(self):
        self.record = RunRecord(
            run_id="run-1",
            thread_id="thread-1",
            assistant_id="lead_agent",
        )

    def test_runtime_context_protects_run_identity(self):
        runtime_context = build_runtime_context(
            self.record,
            {
                "thread_id": "caller-thread",
                "run_id": "caller-run",
                "model_name": "test-model",
            },
        )

        self.assertEqual(runtime_context["thread_id"], "thread-1")
        self.assertEqual(runtime_context["run_id"], "run-1")
        self.assertEqual(runtime_context["model_name"], "test-model")

    def test_runnable_config_preserves_request_fields_and_forces_thread_id(self):
        runtime_context = build_runtime_context(
            self.record,
            {"model_name": "test-model", "recursion_limit": 77},
        )

        config = build_runnable_config(
            self.record,
            {
                "configurable": {"thread_id": "wrong", "custom": "value"},
                "metadata": {"source": "test"},
                "recursion_limit": 88,
                "context": {"request_context": True},
            },
            runtime_context,
        )

        self.assertEqual(config["configurable"]["thread_id"], "thread-1")
        self.assertEqual(config["configurable"]["custom"], "value")
        self.assertEqual(config["metadata"], {"source": "test"})
        self.assertEqual(config["recursion_limit"], 88)
        self.assertTrue(config["context"]["request_context"])
        self.assertEqual(config["context"]["model_name"], "test-model")
        self.assertEqual(config["context"]["run_id"], "run-1")

    def test_run_context_groups_thread_store_and_agent_context(self):
        thread_store = ThreadStore()
        ctx = RunContext(
            thread_store=thread_store,
            agent_context={"temperature": 0.2},
        )

        self.assertIs(ctx.thread_store, thread_store)
        self.assertEqual(dict(ctx.agent_context), {"temperature": 0.2})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_run_context -v
```

Expected: FAIL while importing `app.runtime.runs.context` or `app.runtime.runs.input` because the modules do not exist.

- [ ] **Step 3: Implement the input normalization module**

Create `app/runtime/runs/input.py`:

```python
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
```

- [ ] **Step 4: Implement `RunContext` and runnable config construction**

Create `app/runtime/runs/context.py`:

```python
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.store.models import RunRecord
from app.store.thread_store import ThreadStore


@dataclass(frozen=True)
class RunContext:
    thread_store: ThreadStore
    agent_context: Mapping[str, Any] = field(default_factory=dict)


def build_runtime_context(
    record: RunRecord,
    agent_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    runtime_context = dict(agent_context or {})
    runtime_context["thread_id"] = record.thread_id
    runtime_context["run_id"] = record.run_id
    return runtime_context


def build_runnable_config(
    record: RunRecord,
    request_config: Mapping[str, Any] | None,
    runtime_context: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(request_config or {})

    raw_configurable = config.get("configurable")
    configurable = (
        dict(raw_configurable)
        if isinstance(raw_configurable, Mapping)
        else {}
    )
    configurable["thread_id"] = record.thread_id
    config["configurable"] = configurable

    raw_context = config.get("context")
    installed_context = (
        dict(raw_context)
        if isinstance(raw_context, Mapping)
        else {}
    )
    installed_context.update(runtime_context)
    config["context"] = installed_context

    config.setdefault(
        "recursion_limit",
        int(runtime_context.get("recursion_limit", 50)),
    )
    return config
```

- [ ] **Step 5: Run the task tests and commit only the new files**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_run_context -v
git add -- app/runtime/runs/context.py app/runtime/runs/input.py tests/test_runtime_run_context.py
git diff --cached --check -- app/runtime/runs/context.py app/runtime/runs/input.py tests/test_runtime_run_context.py
git commit --only app/runtime/runs/context.py app/runtime/runs/input.py tests/test_runtime_run_context.py -m "refactor: extract run input and context"
```

Expected: all Task 1 tests pass; the commit contains only the three named files.

---

### Task 2: Introduce the Generic LangGraph Stream Adapter

**Files:**
- Create: `app/runtime/runs/stream_adapter.py`
- Modify: `app/runtime/stream.py`
- Create: `tests/test_runtime_stream_adapter.py`

- [ ] **Step 1: Write failing stream adapter tests**

Create `tests/test_runtime_stream_adapter.py` with a recording bridge and fake agents covering:

```python
import unittest

from langchain_core.messages import AIMessageChunk, HumanMessage

from app.runtime.runs.stream_adapter import (
    LangGraphStreamAdapter,
    ensure_internal_stream_modes,
    normalize_stream_modes,
)
from app.runtime.stream import StreamBridge


class RecordingBridge:
    def __init__(self):
        self.events = []

    async def publish(self, run_id, event, data):
        self.events.append((run_id, event, data))


class MultiModeAgent:
    def __init__(self):
        self.seen_stream_mode = None

    async def astream(self, graph_input, *, config, stream_mode):
        self.seen_stream_mode = stream_mode
        yield (
            "messages",
            (AIMessageChunk(content="partial"), {"langgraph_node": "model"}),
        )
        yield (
            "values",
            {
                "messages": [
                    HumanMessage(content="question", id="human-1"),
                ],
                "agent_trace": [{"agent": "IntentAgent"}],
            },
        )
        yield "tasks", {"id": "task-1"}


class RawValuesAgent:
    async def astream(self, graph_input, *, config, stream_mode):
        yield {"messages": [HumanMessage(content="question", id="human-1")]}


class RuntimeStreamAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_stream_mode_normalization_and_internal_modes_are_stable(self):
        self.assertEqual(normalize_stream_modes(None), ["messages"])
        self.assertEqual(normalize_stream_modes("values"), ["values"])
        self.assertEqual(
            ensure_internal_stream_modes(["values", "tasks", "messages"]),
            ["messages", "values", "tasks"],
        )

    async def test_forward_publishes_compatible_events_and_returns_snapshot(self):
        bridge = RecordingBridge()
        agent = MultiModeAgent()
        observed_values = []

        async def observe_values(values):
            observed_values.append(values)
            return [("updates", {"type": "trace"})]

        snapshot = await LangGraphStreamAdapter(
            bridge=bridge,
            run_id="run-1",
            emit_debug_events=False,
        ).forward(
            agent=agent,
            graph_input={"messages": []},
            config={"configurable": {"thread_id": "thread-1"}},
            requested_modes=["messages", "values", "tasks"],
            values_observer=observe_values,
        )

        self.assertEqual(agent.seen_stream_mode, ["messages", "values", "tasks"])
        self.assertEqual(snapshot.latest_values["agent_trace"], [{"agent": "IntentAgent"}])
        self.assertEqual(snapshot.latest_messages[0]["type"], "human")
        self.assertEqual(len(observed_values), 1)
        self.assertIn("messages", [event for _, event, _ in bridge.events])
        self.assertIn("values", [event for _, event, _ in bridge.events])
        self.assertIn(
            ("run-1", "updates", {"stream_event": "tasks", "data": {"id": "task-1"}}),
            bridge.events,
        )
        self.assertIn(("run-1", "updates", {"type": "trace"}), bridge.events)

    async def test_raw_chunk_falls_back_to_values_and_is_not_published_unrequested(self):
        bridge = RecordingBridge()

        snapshot = await LangGraphStreamAdapter(
            bridge=bridge,
            run_id="run-1",
            emit_debug_events=False,
        ).forward(
            agent=RawValuesAgent(),
            graph_input={"messages": []},
            config={"configurable": {"thread_id": "thread-1"}},
            requested_modes=["messages"],
        )

        self.assertEqual(snapshot.latest_messages[0]["content"], "question")
        self.assertNotIn("values", [event for _, event, _ in bridge.events])

    async def test_stream_bridge_cleanup_removes_queue_after_delay(self):
        bridge = StreamBridge()
        bridge.create("run-1")

        await bridge.cleanup("run-1", delay=0)

        self.assertNotIn("run-1", bridge.queues)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_stream_adapter -v
```

Expected: FAIL importing `app.runtime.runs.stream_adapter`. After that module is scaffolded, the cleanup assertion must still fail until `StreamBridge.cleanup(...)` is added.

- [ ] **Step 3: Implement stream mode helpers and `StreamSnapshot`**

Create `app/runtime/runs/stream_adapter.py` with:

```python
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.runtime.serialization import serialize, serialize_message
from app.runtime.stream import StreamBridge


ProjectedEvents = list[tuple[str, Any]]
ValuesObserver = Callable[
    [dict[str, Any]],
    Awaitable[ProjectedEvents] | ProjectedEvents,
]


@dataclass(frozen=True)
class StreamSnapshot:
    latest_values: dict[str, Any] = field(default_factory=dict)
    latest_messages: list[dict[str, Any]] = field(default_factory=list)


def normalize_stream_modes(modes: Any) -> list[str]:
    if modes is None:
        return ["messages"]
    if isinstance(modes, str):
        return [modes]
    if isinstance(modes, (list, tuple, set)):
        return [str(mode) for mode in modes]
    return ["messages"]


def ensure_internal_stream_modes(modes: list[str]) -> list[str]:
    result: list[str] = []
    for mode in ("messages", "values", *modes):
        if mode not in result:
            result.append(mode)
    return result


def split_stream_item(item: Any) -> tuple[str, Any]:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return item[0], item[1]
    return "values", item
```

- [ ] **Step 4: Implement adapter forwarding and observation**

Complete `LangGraphStreamAdapter` in the same module:

```python
class LangGraphStreamAdapter:
    def __init__(
        self,
        *,
        bridge: StreamBridge,
        run_id: str,
        emit_debug_events: bool,
    ) -> None:
        self.bridge = bridge
        self.run_id = run_id
        self.emit_debug_events = emit_debug_events

    async def forward(
        self,
        *,
        agent: Any,
        graph_input: dict[str, Any],
        config: dict[str, Any],
        requested_modes: list[str] | str | None,
        values_observer: ValuesObserver | None = None,
    ) -> StreamSnapshot:
        normalized_modes = normalize_stream_modes(requested_modes)
        internal_modes = ensure_internal_stream_modes(normalized_modes)
        publish_values = "values" in normalized_modes or self.emit_debug_events
        latest_values: dict[str, Any] = {}
        latest_messages: list[dict[str, Any]] = []

        async for item in agent.astream(
            graph_input,
            config=config,
            stream_mode=internal_modes,
        ):
            mode, chunk = split_stream_item(item)

            if mode == "messages":
                await self.bridge.publish(
                    self.run_id,
                    "messages",
                    serialize(chunk, mode="messages"),
                )
                continue

            if mode != "values":
                payload = serialize(chunk, mode=mode)
                if mode != "updates":
                    payload = {"stream_event": mode, "data": payload}
                await self.bridge.publish(self.run_id, "updates", payload)
                continue

            serialized_values = serialize(chunk, mode="values")
            if isinstance(serialized_values, dict):
                latest_values = serialized_values
                raw_messages = serialized_values.get("messages", [])
                if isinstance(raw_messages, list):
                    latest_messages = [
                        message
                        if isinstance(message, dict)
                        else serialize_message(message)
                        for message in raw_messages
                    ]

            if publish_values:
                await self.bridge.publish(
                    self.run_id,
                    "values",
                    serialized_values,
                )

            if isinstance(serialized_values, dict) and values_observer is not None:
                observed = values_observer(serialized_values)
                projected_events = (
                    await observed if inspect.isawaitable(observed) else observed
                )
                for event, data in projected_events:
                    await self.bridge.publish(self.run_id, event, data)

        return StreamSnapshot(
            latest_values=latest_values,
            latest_messages=latest_messages,
        )
```

- [ ] **Step 5: Add delayed bridge cleanup, run the adapter tests, and commit**

Add to `app/runtime/stream.py`:

```python
    async def cleanup(self, run_id: str, delay: float = 60) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        self.queues.pop(run_id, None)
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_stream_adapter -v
git add -- app/runtime/runs/stream_adapter.py app/runtime/stream.py tests/test_runtime_stream_adapter.py
git diff --cached --check -- app/runtime/runs/stream_adapter.py app/runtime/stream.py tests/test_runtime_stream_adapter.py
git commit --only app/runtime/runs/stream_adapter.py app/runtime/stream.py tests/test_runtime_stream_adapter.py -m "refactor: add LangGraph stream adapter"
```

Expected: adapter and cleanup tests pass; only the adapter, bridge, and adapter-test files are committed.

---

### Task 3: Extract Completion and Public Projection

**Files:**
- Create: `app/runtime/runs/projection.py`
- Modify: `app/runtime/public_messages.py`
- Create: `tests/test_runtime_completion_projection.py`

- [ ] **Step 1: Write failing completion projection tests**

Create `tests/test_runtime_completion_projection.py`. Use small fake agents with `aget_state`/`aupdate_state` and `unittest.mock.patch` on `app.runtime.runs.projection.apply_guardrails`. Cover these exact assertions:

```python
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.runtime.serialization import serialize_message
from app.runtime.runs.projection import RunCompletionProjection
from app.runtime.runs.stream_adapter import StreamSnapshot
from app.store.models import RunRecord


class CheckpointAgent:
    def __init__(self, messages):
        self.messages = list(messages)
        self.update_calls = []

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": list(self.messages)})

    async def aupdate_state(self, config, values):
        replacement = values["messages"][0]
        self.update_calls.append(replacement)
        for index, message in enumerate(self.messages):
            if getattr(message, "id", None) == replacement.id:
                self.messages[index] = replacement
                break


class RuntimeCompletionProjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_clarification_uses_only_current_run_messages_and_skips_guardrail(self):
        old_messages = [
            HumanMessage(content="old question", id="human-old"),
            AIMessage(content="old answer", id="ai-old"),
        ]
        current_messages = [
            HumanMessage(content="current question", id="human-new"),
            AIMessage(
                content="Please clarify:",
                id="ai-tool",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "ask_clarification",
                        "args": {"questions": ["How long?"]},
                    }
                ],
            ),
            {
                "type": "tool",
                "content": "Please clarify: How long?",
                "name": "ask_clarification",
                "tool_call_id": "call-1",
                "id": "clarification:call-1",
            },
        ]
        agent = CheckpointAgent(old_messages)
        serialized_messages = [
            message if isinstance(message, dict) else serialize_message(message)
            for message in old_messages + current_messages
        ]
        projection = RunCompletionProjection(
            record=RunRecord("run-1", "thread-1", "lead_agent"),
            thread_values={"conversation": []},
            user_text="current question",
            emit_debug_events=False,
            message_start_index=len(old_messages),
        )

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(side_effect=AssertionError("guardrail must be skipped")),
        ):
            result = await projection.complete(
                agent=agent,
                config={"configurable": {"thread_id": "thread-1"}},
                snapshot=StreamSnapshot(
                    latest_values={"messages": serialized_messages},
                    latest_messages=serialized_messages,
                ),
                publish_update=AsyncMock(),
            )

        self.assertEqual(result.run_status, "waiting_clarification")
        self.assertEqual(result.thread_status, "waiting")
        self.assertEqual(result.final_payload["status"], "need_clarification")
        self.assertEqual(result.final_payload["pending_clarification"], ["How long?"])
        self.assertNotIn("old answer", result.final_payload["assistant_message"])

    async def test_guardrail_rewrite_updates_checkpoint_and_public_projection(self):
        messages = [
            HumanMessage(content="question", id="human-1"),
            AIMessage(content="unsafe original", id="ai-final"),
        ]
        agent = CheckpointAgent(messages)
        serialized_messages = [serialize_message(message) for message in messages]
        projection = RunCompletionProjection(
            record=RunRecord("run-1", "thread-1", "lead_agent"),
            thread_values={"conversation": []},
            user_text="question",
            emit_debug_events=True,
            message_start_index=0,
        )
        publish_update = AsyncMock()
        await projection.observe_values(
            {
                "messages": serialized_messages,
                "agent_trace": [{"agent": "SafetyAgent"}],
            }
        )

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "safe rewrite",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": False},
                    "rewritten": True,
                    "allowed_terms": [],
                }
            ),
        ):
            result = await projection.complete(
                agent=agent,
                config={"configurable": {"thread_id": "thread-1"}},
                snapshot=StreamSnapshot(
                    latest_values={
                        "messages": serialized_messages,
                        "agent_trace": [{"agent": "SafetyAgent"}],
                    },
                    latest_messages=serialized_messages,
                ),
                publish_update=publish_update,
            )

        self.assertEqual(result.run_status, "success")
        self.assertEqual(result.thread_status, "idle")
        self.assertEqual(result.final_payload["assistant_message"], "safe rewrite")
        self.assertEqual(agent.messages[-1].content, "safe rewrite")
        self.assertEqual(
            result.thread_values["conversation"][-1]["agent_trace"],
            [{"agent": "SafetyAgent"}],
        )
        self.assertEqual(publish_update.await_count, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the missing projection failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_completion_projection -v
```

Expected: FAIL importing `app.runtime.runs.projection`.

- [ ] **Step 3: Move public-message helpers out of the worker**

Add to `app/runtime/public_messages.py`:

```python
def extract_final_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("type") != "ai":
            continue
        if message.get("tool_calls"):
            continue
        content = _message_content(message)
        if content.strip():
            return content
    return ""


def append_visible_messages(
    thread_values: dict[str, Any],
    user_text: str,
    assistant_text: str,
    *,
    run_id: str | None = None,
    agent_trace: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    conversation = list(thread_values.get("conversation") or [])
    if user_text:
        conversation.append({"role": "user", "content": user_text})
    if assistant_text:
        assistant_turn: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_text,
        }
        if run_id:
            assistant_turn["run_id"] = run_id
        if agent_trace:
            assistant_turn["agent_trace"] = [dict(item) for item in agent_trace]
        conversation.append(assistant_turn)
    return conversation
```

- [ ] **Step 4: Implement `RunCompletionProjection`**

Create `app/runtime/runs/projection.py` with these public contracts:

```python
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from app.middlewares.clarification_controller import (
    extract_latest_clarification_question,
)
from app.middlewares.guardrail_middleware import apply_guardrails
from app.middlewares.trace_middleware import extract_trace_events_from_messages
from app.runtime.public_messages import (
    append_visible_messages,
    build_chat_response,
    extract_final_assistant_text,
    extract_latest_assistant_message,
    extract_pending_clarification,
)
from app.runtime.serialization import serialize_message
from app.runtime.runs.stream_adapter import ProjectedEvents, StreamSnapshot
from app.store.models import RunRecord


PublishUpdate = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class CompletionResult:
    run_status: str
    thread_status: str
    thread_values: dict[str, Any]
    final_payload: dict[str, Any]


async def checkpoint_message_count(agent: Any, config: dict[str, Any]) -> int:
    aget_state = getattr(agent, "aget_state", None)
    if aget_state is None:
        return 0
    snapshot = await aget_state(config)
    return len(snapshot.values.get("messages", []))
```

Implement `RunCompletionProjection` so that:

```python
class RunCompletionProjection:
    def __init__(
        self,
        *,
        record: RunRecord,
        thread_values: dict[str, Any],
        user_text: str,
        emit_debug_events: bool,
        message_start_index: int,
    ) -> None:
        self.record = record
        self.thread_values = thread_values
        self.user_text = user_text
        self.emit_debug_events = emit_debug_events
        self.message_start_index = message_start_index
        self.current_agent_trace: list[dict[str, Any]] = []
        self.emitted_trace_keys: set[str] = set()

    def current_run_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return messages[self.message_start_index :]

    async def observe_values(
        self,
        values: dict[str, Any],
    ) -> ProjectedEvents:
        raw_trace = values.get("agent_trace")
        if isinstance(raw_trace, list):
            self.current_agent_trace = [
                dict(item) for item in raw_trace if isinstance(item, dict)
            ]

        raw_messages = values.get("messages", [])
        messages = [
            message if isinstance(message, dict) else serialize_message(message)
            for message in raw_messages
        ] if isinstance(raw_messages, list) else []

        if not self.emit_debug_events:
            return []
        return [
            ("updates", event)
            for event in extract_trace_events_from_messages(
                messages=self.current_run_messages(messages),
                emitted_keys=self.emitted_trace_keys,
            )
        ]
```

Add the checkpoint rewrite and completion methods to the same module:

```python
async def _replace_final_ai_message_in_checkpoint(
    *,
    agent: Any,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    final_text: str,
) -> list[dict[str, Any]]:
    target = next(
        (
            message
            for message in reversed(messages)
            if message.get("type") == "ai"
            and message.get("content")
            and not message.get("tool_calls")
        ),
        None,
    )
    if target is None or target.get("content") == final_text:
        return messages

    message_id = target.get("id")
    if not message_id:
        raise RuntimeError(
            "无法写回 Guardrail 答案：最终 AIMessage 缺少 id"
        )

    await agent.aupdate_state(
        config,
        {"messages": [AIMessage(id=message_id, content=final_text)]},
    )
    snapshot = await agent.aget_state(config)
    return [
        serialize_message(message)
        for message in snapshot.values.get("messages", [])
    ]


    async def complete(
        self,
        *,
        agent: Any,
        config: dict[str, Any],
        snapshot: StreamSnapshot,
        publish_update: PublishUpdate,
    ) -> CompletionResult:
        full_messages = list(snapshot.latest_messages)
        current_messages = self.current_run_messages(full_messages)
        clarification = extract_latest_clarification_question(current_messages)

        if clarification:
            assistant_message = (
                extract_latest_assistant_message(current_messages)
                or clarification
            )
            conversation = append_visible_messages(
                self.thread_values,
                self.user_text,
                assistant_message,
                run_id=self.record.run_id,
                agent_trace=self.current_agent_trace,
            )
            return CompletionResult(
                run_status="waiting_clarification",
                thread_status="waiting",
                thread_values={
                    "messages": full_messages,
                    "conversation": conversation,
                },
                final_payload=build_chat_response(
                    thread_id=self.record.thread_id,
                    run_id=self.record.run_id,
                    status="need_clarification",
                    assistant_message=assistant_message,
                    pending_clarification=extract_pending_clarification(
                        current_messages
                    ),
                ),
            )

        original_final_text = extract_final_assistant_text(current_messages)
        if self.emit_debug_events:
            await publish_update(
                {
                    "type": "guardrail",
                    "status": "started",
                    "agent": "guardrail_middleware",
                    "summary": "正在进行术语一致性校验与答案安全检查。",
                }
            )

        guardrail_result = await apply_guardrails(
            final_text=original_final_text,
            messages=full_messages,
        )

        if self.emit_debug_events:
            await publish_update(
                {
                    "type": "guardrail",
                    "status": "completed",
                    "agent": "guardrail_middleware",
                    "summary": "术语一致性校验完成。",
                    "validation": guardrail_result.get("validation"),
                    "rewritten": guardrail_result.get("rewritten"),
                }
            )

        final_text = guardrail_result["final_text"]
        if final_text != original_final_text:
            full_messages = await _replace_final_ai_message_in_checkpoint(
                agent=agent,
                config=config,
                messages=full_messages,
                final_text=final_text,
            )

        conversation = append_visible_messages(
            self.thread_values,
            self.user_text,
            final_text,
            run_id=self.record.run_id,
            agent_trace=self.current_agent_trace,
        )
        return CompletionResult(
            run_status="success",
            thread_status="idle",
            thread_values={
                "messages": full_messages,
                "conversation": conversation,
            },
            final_payload=build_chat_response(
                thread_id=self.record.thread_id,
                run_id=self.record.run_id,
                status="completed",
                assistant_message=final_text,
            ),
        )
```

Indent `complete(...)` as a method of `RunCompletionProjection`; keep `_replace_final_ai_message_in_checkpoint(...)` private at module scope. Do not re-export either through `worker.py`.

- [ ] **Step 5: Run projection/public-message tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_completion_projection tests.gateway.test_threads_router -v
git add -- app/runtime/runs/projection.py app/runtime/public_messages.py tests/test_runtime_completion_projection.py
git diff --cached --check -- app/runtime/runs/projection.py app/runtime/public_messages.py tests/test_runtime_completion_projection.py
git commit --only app/runtime/runs/projection.py app/runtime/public_messages.py tests/test_runtime_completion_projection.py -m "refactor: extract run completion projection"
```

Expected: projection tests pass and existing public-message contract assertions remain green.

---

### Task 4: Reduce the Worker to Lifecycle Orchestration

**Files:**
- Modify: `app/runtime/runs/worker.py`
- Modify: `app/runtime/services.py`
- Create: `tests/test_runtime_worker_boundary.py`
- Modify: `tests/test_async_agent_factory.py`
- Modify: `tests/test_clarification_flow.py`
- Modify: `tests/test_subagent_clarification.py`
- Modify: `tests/test_workflow_agent_flow.py`
- Modify: `tests/test_collaboration_history.py`
- Modify: `tests/gateway/test_threads_router.py`
- Modify: `tests/gateway/test_thread_run_services.py`

- [ ] **Step 1: Write the failing static worker-boundary test**

Create `tests/test_runtime_worker_boundary.py`:

```python
import ast
import unittest
from pathlib import Path


class RuntimeWorkerBoundaryTests(unittest.TestCase):
    def test_worker_does_not_import_business_message_or_guardrail_helpers(self):
        worker_path = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "runtime"
            / "runs"
            / "worker.py"
        )
        tree = ast.parse(worker_path.read_text(encoding="utf-8"))

        imported_names = set()
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module or "")
                imported_names.update(alias.name for alias in node.names)

        self.assertFalse(
            {
                "app.middlewares.guardrail_middleware",
                "app.middlewares.clarification_controller",
                "app.middlewares.trace_middleware",
                "langchain_core.messages",
            }
            & imported_modules
        )
        self.assertFalse(
            {
                "apply_guardrails",
                "extract_latest_clarification_question",
                "extract_trace_events_from_messages",
                "build_chat_response",
                "AIMessage",
            }
            & imported_names
        )

    def test_worker_contains_no_message_parsing_or_response_builder_functions(self):
        worker_path = (
            Path(__file__).resolve().parents[1]
            / "app"
            / "runtime"
            / "runs"
            / "worker.py"
        )
        tree = ast.parse(worker_path.read_text(encoding="utf-8"))
        function_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertEqual(function_names, {"run_agent"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the boundary test and verify it fails on current worker responsibilities**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_worker_boundary -v
```

Expected: FAIL because the current worker imports guardrail/clarification/trace/message helpers and defines parsing/projection functions.

- [ ] **Step 3: Replace `worker.py` with lifecycle orchestration**

Use this structure in `app/runtime/runs/worker.py`:

```python
import asyncio
import logging
import traceback
from collections.abc import Callable
from typing import Any

from app.runtime.runs.context import (
    RunContext,
    build_runnable_config,
    build_runtime_context,
)
from app.runtime.runs.input import extract_user_text
from app.runtime.runs.projection import (
    RunCompletionProjection,
    checkpoint_message_count,
)
from app.runtime.runs.stream_adapter import LangGraphStreamAdapter
from app.runtime.stream import StreamBridge
from app.store.models import RunRecord
from app.store.run_manager import RunManager


logger = logging.getLogger(__name__)


async def run_agent(
    bridge: StreamBridge,
    run_manager: RunManager,
    record: RunRecord,
    *,
    ctx: RunContext,
    agent_factory: Callable[[dict[str, Any] | None], Any],
    graph_input: dict[str, Any],
    config: dict[str, Any],
    stream_modes: list[str] | None = None,
) -> None:
    run_id = record.run_id
    thread_id = record.thread_id

    try:
        await run_manager.set_status(run_id, "running")
        await ctx.thread_store.update_status(thread_id, "running")
        await bridge.publish(
            run_id,
            "metadata",
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "assistant_id": record.assistant_id,
                "architecture": "tcm-flow",
            },
        )

        thread = await ctx.thread_store.get(thread_id)
        thread_values = dict(thread.values) if thread else {}
        runtime_context = build_runtime_context(record, ctx.agent_context)
        runnable_config = build_runnable_config(
            record,
            config,
            runtime_context,
        )
        agent = agent_factory(runtime_context)
        message_start_index = await checkpoint_message_count(
            agent,
            runnable_config,
        )
        emit_debug_events = bool(runtime_context.get("debug_events"))
        projection = RunCompletionProjection(
            record=record,
            thread_values=thread_values,
            user_text=extract_user_text(graph_input),
            emit_debug_events=emit_debug_events,
            message_start_index=message_start_index,
        )
        snapshot = await LangGraphStreamAdapter(
            bridge=bridge,
            run_id=run_id,
            emit_debug_events=emit_debug_events,
        ).forward(
            agent=agent,
            graph_input=graph_input,
            config=runnable_config,
            requested_modes=stream_modes,
            values_observer=projection.observe_values,
        )

        async def publish_update(payload: dict[str, Any]) -> None:
            await bridge.publish(run_id, "updates", payload)

        completion = await projection.complete(
            agent=agent,
            config=runnable_config,
            snapshot=snapshot,
            publish_update=publish_update,
        )
        await ctx.thread_store.update_values(
            thread_id,
            completion.thread_values,
            run_id=run_id,
        )
        await run_manager.set_status(run_id, completion.run_status)
        await ctx.thread_store.update_status(thread_id, completion.thread_status)
        await bridge.publish(run_id, "final", completion.final_payload)

    except Exception as exc:
        error = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
        logger.exception("Run %s failed: %s", run_id, error)
        await run_manager.set_status(run_id, "error", error=error)
        await ctx.thread_store.update_status(thread_id, "error")
        await bridge.publish(run_id, "error", {"message": error})
    finally:
        await bridge.publish_end(run_id)
        asyncio.create_task(bridge.cleanup(run_id, delay=60))
```

Keep this terminal order deliberate: persist completion thread values, commit the
run and thread terminal statuses, and only then publish `final`. If cancellation
lands during either status write, Task 5's explicit cancellation branch runs
before a `final` event escapes, preventing a client-visible final response for a
run that is subsequently recorded as `cancelled`.

Do not retain aliases or wrapper functions for the old worker-local business helpers.

- [ ] **Step 4: Update `services.py` and direct worker callers**

In `app/runtime/services.py`:

```python
from app.runtime.runs.context import RunContext
from app.runtime.runs.input import normalize_graph_input
```

Replace the old worker arguments with:

```python
        run_agent(
            bridge=state.bridge,
            run_manager=state.run_manager,
            record=record,
            ctx=RunContext(
                thread_store=state.thread_store,
                agent_context=dict(body.context or {}),
            ),
            agent_factory=agent_factory,
            graph_input=normalize_graph_input(body.input.model_dump()),
            config=dict(body.config or {}),
            stream_modes=list(body.stream_mode or []),
        )
```

Update direct test calls in the listed test files from:

```python
await run_agent(
    bridge=bridge,
    run_manager=run_manager,
    thread_store=thread_store,
    record=run,
    agent_factory=factory,
    input_data=input_data,
    context=context,
)
```

to:

```python
await run_agent(
    bridge=bridge,
    run_manager=run_manager,
    record=run,
    ctx=RunContext(
        thread_store=thread_store,
        agent_context=context,
    ),
    agent_factory=factory,
    graph_input=normalize_graph_input(input_data),
    config={},
    stream_modes=context.get("stream_mode"),
)
```

Update test imports accordingly. Replace patches of:

```python
"app.runtime.runs.worker.apply_guardrails"
```

with:

```python
"app.runtime.runs.projection.apply_guardrails"
```

Replace `message_to_dict(...)` test use with `serialize_message(...)` imported from `app.runtime.serialization`.

Update `tests/gateway/test_thread_run_services.py` so its fake worker asserts:

```python
self.assertEqual(captured["graph_input"]["messages"][0]["role"], "user")
self.assertEqual(captured["config"]["metadata"], {"source": "test"})
self.assertEqual(captured["stream_modes"], ["messages"])
self.assertIs(captured["ctx"].thread_store, state.thread_store)
```

- [ ] **Step 5: Run the worker boundary and compatibility suite, then commit named files only**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_worker_boundary tests.test_async_agent_factory tests.test_clarification_flow tests.test_subagent_clarification tests.gateway.test_threads_router tests.gateway.test_thread_run_services tests.test_workflow_agent_flow tests.test_collaboration_history -v
git add -- app/runtime/runs/worker.py app/runtime/services.py tests/test_runtime_worker_boundary.py tests/test_async_agent_factory.py tests/test_clarification_flow.py tests/test_subagent_clarification.py tests/test_workflow_agent_flow.py tests/test_collaboration_history.py tests/gateway/test_threads_router.py tests/gateway/test_thread_run_services.py
git diff --cached --check -- app/runtime/runs/worker.py app/runtime/services.py tests/test_runtime_worker_boundary.py tests/test_async_agent_factory.py tests/test_clarification_flow.py tests/test_subagent_clarification.py tests/test_workflow_agent_flow.py tests/test_collaboration_history.py tests/gateway/test_threads_router.py tests/gateway/test_thread_run_services.py
git commit --only app/runtime/runs/worker.py app/runtime/services.py tests/test_runtime_worker_boundary.py tests/test_async_agent_factory.py tests/test_clarification_flow.py tests/test_subagent_clarification.py tests/test_workflow_agent_flow.py tests/test_collaboration_history.py tests/gateway/test_threads_router.py tests/gateway/test_thread_run_services.py -m "refactor: make runtime worker lifecycle-only"
```

Expected: the static boundary test passes; success, clarification, guardrail rewrite, current-run isolation, debug updates, and agent-trace compatibility remain green.

---

### Task 5: Add Explicit Cancellation Lifecycle Handling

**Files:**
- Modify: `app/runtime/runs/worker.py`
- Create: `tests/test_runtime_worker_lifecycle.py`

- [ ] **Step 1: Write failing cancellation and cleanup tests**

Create `tests/test_runtime_worker_lifecycle.py`:

```python
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.runtime.runs.context import RunContext
from app.runtime.runs.worker import run_agent
from app.runtime.stream import StreamBridge
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


class CancelledAgent:
    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": []})

    async def astream(self, graph_input, *, config, stream_mode):
        if False:
            yield None
        raise asyncio.CancelledError


class SuccessfulAgent:
    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": []})

    async def astream(self, graph_input, *, config, stream_mode):
        yield (
            "values",
            {
                "messages": [
                    HumanMessage(content="question", id="human-1"),
                    AIMessage(content="answer", id="ai-1"),
                ]
            },
        )


class CancelOnSuccessRunManager(RunManager):
    async def set_status(
        self,
        run_id: str,
        status: str,
        error: str | None = None,
    ):
        if status == "success":
            task = asyncio.current_task()
            if task is not None:
                task.cancel()
            await asyncio.sleep(0)
        await super().set_status(run_id, status, error=error)


class FailingCancellationRunManager(RunManager):
    async def set_status(
        self,
        run_id: str,
        status: str,
        error: str | None = None,
    ):
        if status == "cancelled":
            raise RuntimeError("run cancellation status failed")
        await super().set_status(run_id, status, error=error)


class FailingCancellationThreadStore(ThreadStore):
    async def update_status(self, thread_id: str, status: str):
        if status == "idle":
            raise RuntimeError("thread cancellation status failed")
        await super().update_status(thread_id, status)


class RecordingBridge(StreamBridge):
    def __init__(self):
        super().__init__()
        self.cleanup_calls: list[tuple[str, float]] = []

    async def cleanup(self, run_id: str, delay: float = 60):
        self.cleanup_calls.append((run_id, delay))


class RuntimeWorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def drain_events(self, bridge: RecordingBridge, run_id: str):
        queued_events = []
        while not bridge.queues[run_id].empty():
            queued_events.append(await bridge.queues[run_id].get())
        return queued_events

    async def test_cancelled_run_is_finalized_without_error_or_final_event(self):
        bridge = RecordingBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with self.assertRaises(asyncio.CancelledError):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                record=run,
                ctx=RunContext(thread_store=thread_store),
                agent_factory=lambda context: CancelledAgent(),
                graph_input={"messages": []},
                config={},
                stream_modes=["messages"],
            )

        stored_run = await run_manager.get(run.run_id)
        stored_thread = await thread_store.get(thread.thread_id)
        self.assertIsNotNone(stored_run)
        self.assertIsNotNone(stored_thread)
        self.assertEqual(stored_run.status, "cancelled")
        self.assertEqual(stored_thread.status, "idle")

        await asyncio.sleep(0)
        self.assertEqual(bridge.cleanup_calls, [(run.run_id, 60)])

        queued_events = await self.drain_events(bridge, run.run_id)
        event_names = [event for event, _ in queued_events]
        self.assertNotIn("error", event_names)
        self.assertNotIn("final", event_names)
        self.assertLess(event_names.index("metadata"), event_names.index("end"))
        self.assertEqual(queued_events[-1], ("end", {"status": "done"}))

    async def test_cancellation_during_success_status_prevents_final_event(self):
        bridge = RecordingBridge()
        run_manager = CancelOnSuccessRunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with patch(
            "app.runtime.runs.projection.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "answer",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": [],
                }
            ),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await run_agent(
                    bridge=bridge,
                    run_manager=run_manager,
                    record=run,
                    ctx=RunContext(thread_store=thread_store),
                    agent_factory=lambda context: SuccessfulAgent(),
                    graph_input={
                        "messages": [{"type": "human", "content": "question"}]
                    },
                    config={},
                    stream_modes=["messages"],
                )

        stored_run = await run_manager.get(run.run_id)
        stored_thread = await thread_store.get(thread.thread_id)
        self.assertIsNotNone(stored_run)
        self.assertIsNotNone(stored_thread)
        self.assertEqual(stored_run.status, "cancelled")
        self.assertEqual(stored_thread.status, "idle")
        self.assertEqual(
            stored_thread.values["conversation"][-1]["content"],
            "answer",
        )

        await asyncio.sleep(0)
        self.assertEqual(bridge.cleanup_calls, [(run.run_id, 60)])
        queued_events = await self.drain_events(bridge, run.run_id)
        event_names = [event for event, _ in queued_events]
        self.assertNotIn("error", event_names)
        self.assertNotIn("final", event_names)
        self.assertEqual(queued_events[-1], ("end", {"status": "done"}))

    async def test_run_status_failure_does_not_mask_cancellation(self):
        bridge = RecordingBridge()
        run_manager = FailingCancellationRunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with self.assertLogs("app.runtime.runs.worker", level="ERROR") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await run_agent(
                    bridge=bridge,
                    run_manager=run_manager,
                    record=run,
                    ctx=RunContext(thread_store=thread_store),
                    agent_factory=lambda context: CancelledAgent(),
                    graph_input={"messages": []},
                    config={},
                )

        stored_thread = await thread_store.get(thread.thread_id)
        self.assertIsNotNone(stored_thread)
        self.assertEqual(stored_thread.status, "idle")
        self.assertIn("failed to mark run", "\n".join(logs.output).lower())

        await asyncio.sleep(0)
        self.assertEqual(bridge.cleanup_calls, [(run.run_id, 60)])
        queued_events = await self.drain_events(bridge, run.run_id)
        event_names = [event for event, _ in queued_events]
        self.assertNotIn("error", event_names)
        self.assertNotIn("final", event_names)
        self.assertEqual(queued_events[-1], ("end", {"status": "done"}))

    async def test_thread_status_failure_does_not_mask_cancellation(self):
        bridge = RecordingBridge()
        run_manager = RunManager()
        thread_store = FailingCancellationThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "lead_agent")
        bridge.create(run.run_id)

        with self.assertLogs("app.runtime.runs.worker", level="ERROR") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await run_agent(
                    bridge=bridge,
                    run_manager=run_manager,
                    record=run,
                    ctx=RunContext(thread_store=thread_store),
                    agent_factory=lambda context: CancelledAgent(),
                    graph_input={"messages": []},
                    config={},
                )

        stored_run = await run_manager.get(run.run_id)
        self.assertIsNotNone(stored_run)
        self.assertEqual(stored_run.status, "cancelled")
        self.assertIn("failed to reset thread", "\n".join(logs.output).lower())

        await asyncio.sleep(0)
        self.assertEqual(bridge.cleanup_calls, [(run.run_id, 60)])
        queued_events = await self.drain_events(bridge, run.run_id)
        event_names = [event for event, _ in queued_events]
        self.assertNotIn("error", event_names)
        self.assertNotIn("final", event_names)
        self.assertEqual(queued_events[-1], ("end", {"status": "done"}))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the lifecycle test and verify cancellation status is wrong**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_worker_lifecycle -v
```

Expected: FAIL because the worker has no explicit cancellation branch: ordinary
cancellation and cancellation during the terminal success-status write do not
finish as `cancelled`/`idle`. The persistence-failure tests also fail because the
worker does not yet log each failed cancellation write, continue with the other
best-effort write, and preserve `CancelledError` as the primary exception.

- [ ] **Step 3: Implement explicit cancellation handling**

Add this branch immediately before `except Exception` in `app/runtime/runs/worker.py`:

```python
    except asyncio.CancelledError:
        try:
            await run_manager.set_status(run_id, "cancelled")
        except Exception:
            logger.exception("Failed to mark run %s cancelled", run_id)
        try:
            await ctx.thread_store.update_status(thread_id, "idle")
        except Exception:
            logger.exception(
                "Failed to reset thread %s after run %s cancellation",
                thread_id,
                run_id,
            )
        logger.info("Run %s cancelled", run_id)
        raise
```

The two status writes are independent best-effort operations. A failure in either
write is logged without skipping the other write, and the bare `raise` always
re-raises the original `asyncio.CancelledError` rather than replacing it with a
persistence error. Keep the existing `finally` block unchanged so cancellation
still publishes `end` and schedules delayed cleanup.

- [ ] **Step 4: Run lifecycle and stream tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_worker_lifecycle tests.test_runtime_stream_adapter tests.test_runtime_state -v
```

Expected: all lifecycle, adapter, and state tests pass without pending-task
warnings. In particular, cancellation during the success-status commit persists
the already-produced thread values but emits neither `final` nor `error`, records
`cancelled`/`idle`, and still publishes `end` and schedules cleanup. Failure of
either cancellation status write is logged, does not skip the other status write,
and does not replace the original `CancelledError`.

- [ ] **Step 5: Commit the lifecycle files**

Run:

```powershell
git add -- app/runtime/runs/worker.py tests/test_runtime_worker_lifecycle.py
git diff --cached --check -- app/runtime/runs/worker.py tests/test_runtime_worker_lifecycle.py
git commit --only app/runtime/runs/worker.py tests/test_runtime_worker_lifecycle.py -m "fix: finalize cancelled runtime runs"
```

Expected: only the independently protected worker cancellation branch and the
terminal-ordering/persistence-failure lifecycle tests are committed.

---

### Task 6: Final Boundary and Compatibility Verification

**Files:**
- Verify: all files listed in the Target File Map
- Do not modify unrelated staged files

- [ ] **Step 1: Run new focused unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_run_context tests.test_runtime_stream_adapter tests.test_runtime_completion_projection tests.test_runtime_worker_boundary tests.test_runtime_worker_lifecycle -v
```

Expected: all new tests pass.

- [ ] **Step 2: Re-run the original 45-test compatibility baseline plus service tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_async_agent_factory tests.test_clarification_flow tests.test_subagent_clarification tests.gateway.test_threads_router tests.gateway.test_thread_run_services tests.test_workflow_agent_flow tests.test_collaboration_history -v
```

Expected: all tests pass; the original 45-test baseline remains green and service coverage also passes.

- [ ] **Step 3: Compile changed Python modules**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile app/runtime/runs/context.py app/runtime/runs/input.py app/runtime/runs/stream_adapter.py app/runtime/runs/projection.py app/runtime/runs/worker.py app/runtime/public_messages.py app/runtime/services.py app/runtime/stream.py
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Run the broad repository test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Expected: PASS. If unrelated optional dependencies prevent discovery, capture the exact missing dependency and still report the focused-suite result separately; do not weaken focused tests.

- [ ] **Step 5: Audit the final diff against the design acceptance criteria**

Run:

```powershell
git diff --check
rg -n "apply_guardrails|extract_latest_clarification_question|extract_trace_events_from_messages|AIMessage|build_chat_response|tool_calls|get\(\"content\"" app/runtime/runs/worker.py
git diff --stat HEAD~5..HEAD
git status --short
```

Expected:

- `git diff --check` reports no whitespace errors;
- the `rg` command reports no business parsing or response-construction references in `worker.py`;
- `worker.py` contains lifecycle coordination only;
- completion values and terminal statuses are persisted before `final`, so
  cancellation during a terminal status write cannot produce `final` plus a
  `cancelled` run;
- cancellation status persistence is independent and best-effort: failure of the
  run write does not skip the thread write, failure of the thread write does not
  undo the run write, and both paths re-raise the original `CancelledError`;
- every cancellation path emits neither `final` nor `error`, then publishes
  `end` and schedules delayed cleanup;
- unrelated staged files remain visible and unmodified unless explicitly included in a named task commit.

---

## Plan Self-Review

- Spec coverage: context/config construction, graph input, stream forwarding, current-run slicing, clarification, guardrails, checkpoint rewrite, public payload compatibility, agent trace, errors, terminal status-before-final ordering, cancellation during success commit, independent best-effort cancellation persistence, `CancelledError` preservation, `end`, and cleanup each have an implementation task and test coverage.
- Unresolved-marker scan: the plan contains no incomplete markers, deferred implementation, or undefined generic handling steps.
- Type consistency: `RunContext`, `StreamSnapshot`, `CompletionResult`, `LangGraphStreamAdapter.forward(...)`, `RunCompletionProjection.complete(...)`, and the new `run_agent(...)` signature are used consistently across tasks.
- Scope: no graph redesign, new cancellation endpoint, synthetic chunking, or thread/checkpointer ownership change is introduced.
- Dirty-worktree safety: every commit names exact files and explicitly excludes unrelated staged work.
