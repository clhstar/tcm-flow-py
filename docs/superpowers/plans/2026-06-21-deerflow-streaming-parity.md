# DeerFlow Streaming Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `POST /api/threads/{thread_id}/runs/stream` produce DeerFlow-style token-level `messages` SSE events and typewriter behavior by using DeerFlow's own `create_agent + agent.astream(stream_mode=["messages", ...])` pattern whenever that pattern works in this repo.

**Architecture:** First prove whether our current dependency set can reproduce DeerFlow's `create_agent` token stream. If it cannot, align the LangChain/LangGraph dependency set with DeerFlow and rerun the same parity probe. Only if the DeerFlow-style probe passes do we remove the local `create_streaming_react_agent` adapter and restore Lead Agent construction to `langchain.agents.create_agent`; if the probe still fails, stop and document the exact remaining provider or middleware delta instead of inventing synthetic chunks.

**Tech Stack:** FastAPI SSE, LangGraph `astream`, LangChain `create_agent`, `ChatOpenAI`, `AIMessageChunk`, Python `unittest`, PowerShell, existing `requirements.txt`.

---

## Non-Negotiable Acceptance Criteria

- `messages` SSE events must come from LangGraph `stream_mode="messages"` payloads, not from manually splitting strings.
- A deterministic fake streaming model must prove `create_agent(...).astream(..., stream_mode=["messages", "values"])` emits multiple `AIMessageChunk` payloads before any production code relies on it.
- If DeerFlow's dependency versions make `create_agent` stream correctly, tcm-flow must use `langchain.agents.create_agent` in `app/agents/lead_agent/agent.py` and remove `app/agents/lead_agent/streaming_graph.py`.
- If DeerFlow's dependency versions do not make `create_agent` stream correctly, do not keep guessing. Produce a short failure note with the exact `langchain`, `langgraph`, `langchain-core`, `langchain-openai`, model class, and observed event sequence.
- Existing thin frontend contract remains: provisional `messages`, final `final.assistant_message`, no raw full state to frontend unless explicitly requested.

## File Structure

- Create: `G:\work\tcm-flow\tests\test_deerflow_create_agent_streaming.py`
  - Holds the minimal DeerFlow-style reproduction: `create_agent`, fake streaming chat model, `agent.astream(stream_mode=["messages", "values"])`, assert multiple `AIMessageChunk` payloads.
- Create: `G:\work\tcm-flow\constraints-deerflow-streaming.txt`
  - Temporary dependency constraints copied from the inspected DeerFlow lock so the parity test can be run against DeerFlow-like versions without guessing.
- Modify: `G:\work\tcm-flow\requirements.txt`
  - Only after the parity probe passes under constraints, pin the minimal LangChain/LangGraph set needed for the same behavior in normal installs.
- Modify: `G:\work\tcm-flow\app\agents\lead_agent\agent.py`
  - Restore DeerFlow-style `create_agent(...)` construction with `ClarificationMiddleware`.
- Delete: `G:\work\tcm-flow\app\agents\lead_agent\streaming_graph.py`
  - Remove the local adapter once DeerFlow-style construction is proven.
- Modify: `G:\work\tcm-flow\tests\test_lead_agent_factory.py`
  - Update factory expectations from `create_streaming_react_agent` back to `create_agent`; remove local adapter tests after the parity test replaces them.
- Verify: `G:\work\tcm-flow\app\runtime\runs\worker.py`
  - Keep worker pass-through behavior: it forwards `message, metadata` from `stream_mode="messages"` unchanged.
- Verify: `G:\work\tcm-flow\tests\gateway\test_threads_router.py`
  - Keep tests proving worker forwards `AIMessageChunk` and does not synthesize chunks for full `AIMessage`.
- Optional after backend passes: `G:\work\tcm-consultation-system\tcm-web\src\api\tcmFlow.ts` and `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\PatientIntakeWorkspace.tsx`
  - Only needed if frontend typewriter behavior still does not render despite backend token chunks.

---

### Task 1: Add the DeerFlow-Style `create_agent` Streaming Parity Test

**Files:**
- Create: `G:\work\tcm-flow\tests\test_deerflow_create_agent_streaming.py`

- [ ] **Step 1: Write the failing parity test**

Create `G:\work\tcm-flow\tests\test_deerflow_create_agent_streaming.py`:

```python
import unittest
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool


class ToolBindableStreamingFakeModel(FakeListChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return self


class DeerFlowCreateAgentStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_agent_messages_mode_emits_token_chunks(self):
        agent = create_agent(
            model=ToolBindableStreamingFakeModel(responses=["abc"]),
            tools=[],
            system_prompt="You are a concise assistant.",
        )

        chunks: list[tuple[str, str, str | None]] = []
        async for event in agent.astream(
            {"messages": [{"role": "user", "content": "say abc"}]},
            stream_mode=["messages", "values"],
        ):
            if not (isinstance(event, tuple) and event[0] == "messages"):
                continue

            message, metadata = event[1]
            chunks.append(
                (
                    message.__class__.__name__,
                    str(message.content),
                    metadata.get("langgraph_node"),
                )
            )

        self.assertEqual(
            chunks,
            [
                ("AIMessageChunk", "a", "model"),
                ("AIMessageChunk", "b", "model"),
                ("AIMessageChunk", "c", "model"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the parity test against the current environment**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deerflow_create_agent_streaming
```

Expected in the current known-bad environment: FAIL. The observed `chunks` list is expected to contain one full `AIMessage` such as:

```text
[("AIMessage", "abc", "model")]
```

This failure is useful: it proves why tcm-flow currently cannot get typewriter chunks through plain `create_agent`.

- [ ] **Step 3: Record current installed versions**

Run:

```powershell
.\.venv\Scripts\python.exe - <<'PY'
from importlib.metadata import version

for package in [
    "langchain",
    "langgraph",
    "langchain-core",
    "langchain-openai",
    "langgraph-prebuilt",
    "langgraph-checkpoint",
    "langgraph-sdk",
]:
    print(f"{package}=={version(package)}")
PY
```

Expected current baseline from the last investigation:

```text
langchain==1.3.4
langgraph==1.2.4
langchain-core==1.4.2
langchain-openai==1.2.2
```

---

### Task 2: Run a DeerFlow Dependency Alignment Experiment

**Files:**
- Create: `G:\work\tcm-flow\constraints-deerflow-streaming.txt`
- Do not modify: `G:\work\tcm-flow\requirements.txt` in this task

- [ ] **Step 1: Add DeerFlow streaming constraints**

Create `G:\work\tcm-flow\constraints-deerflow-streaming.txt`:

```text
langchain==1.2.15
langchain-core==1.3.3
langchain-openai==1.2.1
langgraph==1.1.9
langgraph-prebuilt==1.0.11
langgraph-checkpoint==4.0.2
langgraph-sdk==0.3.13
```

These versions come from the inspected DeerFlow lock and are the minimum set directly related to `create_agent` and LangGraph `messages` streaming.

- [ ] **Step 2: Create an isolated parity virtualenv**

Run:

```powershell
py -3.10 -m venv .venv-deerflow-streaming
.\.venv-deerflow-streaming\Scripts\python.exe -m pip install --upgrade pip
.\.venv-deerflow-streaming\Scripts\pip.exe install -r requirements.txt -c constraints-deerflow-streaming.txt
```

Expected: install succeeds without dependency solver conflict. If install fails, capture the exact conflict output and skip to Task 7.

- [ ] **Step 3: Run only the parity test in the isolated virtualenv**

Run:

```powershell
.\.venv-deerflow-streaming\Scripts\python.exe -m unittest tests.test_deerflow_create_agent_streaming
```

Expected if DeerFlow's dependency behavior explains the difference: PASS with three `AIMessageChunk` events.

- [ ] **Step 4: Run a small gateway streaming subset in the isolated virtualenv**

Run:

```powershell
.\.venv-deerflow-streaming\Scripts\python.exe -m unittest tests.gateway.test_threads_router tests.gateway.test_thread_run_services
```

Expected: PASS. This proves the dependency alignment did not break the worker-side SSE contract.

- [ ] **Step 5: Decide the implementation branch**

If Step 3 and Step 4 pass, continue to Task 3.

If Step 3 fails, do not keep the local builder as a hidden permanent workaround. Continue to Task 7 and write the failure note first.

---

### Task 3: Restore Lead Agent Construction to DeerFlow-Style `create_agent`

**Files:**
- Modify: `G:\work\tcm-flow\app\agents\lead_agent\agent.py`
- Delete later in Task 4: `G:\work\tcm-flow\app\agents\lead_agent\streaming_graph.py`

- [ ] **Step 1: Restore imports in `agent.py`**

Change the imports at the top of `G:\work\tcm-flow\app\agents\lead_agent\agent.py` to:

```python
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.agents.lead_agent.prompt import SYSTEM_PROMPT
from app.checkpoints.factory import get_checkpointer, get_checkpointer_async
from app.config import get_settings
from app.middlewares.clarification_middleware import ClarificationMiddleware
from app.tools.tools import get_available_tools
```

- [ ] **Step 2: Restore the DeerFlow-style agent factory call**

In `_build_lead_agent`, replace the local builder call with:

```python
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[ClarificationMiddleware()],
    )
```

Keep the existing `ChatOpenAI(..., streaming=context.get("streaming", True))` model creation unchanged.

- [ ] **Step 3: Run factory and parity tests**

Run:

```powershell
.\.venv-deerflow-streaming\Scripts\python.exe -m unittest tests.test_deerflow_create_agent_streaming tests.test_lead_agent_factory
```

Expected: PASS.

---

### Task 4: Remove the Local Streaming Graph Adapter

**Files:**
- Delete: `G:\work\tcm-flow\app\agents\lead_agent\streaming_graph.py`
- Modify: `G:\work\tcm-flow\tests\test_lead_agent_factory.py`

- [ ] **Step 1: Delete `streaming_graph.py`**

Run:

```powershell
Remove-Item -LiteralPath .\app\agents\lead_agent\streaming_graph.py
```

Expected: file is removed. This is allowed only after Task 2 proves DeerFlow-style `create_agent` emits chunks.

- [ ] **Step 2: Update `test_lead_agent_factory.py` imports**

At the top of `G:\work\tcm-flow\tests\test_lead_agent_factory.py`, keep only:

```python
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.lead_agent.agent import make_lead_agent
```

Remove imports for:

```python
import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any
from app.agents.lead_agent.streaming_graph import create_streaming_react_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
```

- [ ] **Step 3: Restore the factory mock target**

In `test_memory_lead_agent_uses_configured_checkpointer`, patch:

```python
"app.agents.lead_agent.agent.create_agent"
```

and assert:

```python
self.assertIs(create_agent.call_args.kwargs["checkpointer"], checkpointer)
self.assertIs(chat_openai.call_args.kwargs["streaming"], True)
```

- [ ] **Step 4: Remove local adapter tests from `test_lead_agent_factory.py`**

Delete these test methods from `G:\work\tcm-flow\tests\test_lead_agent_factory.py`:

```python
def test_streaming_react_agent_emits_model_token_chunks(self):
    ...

def test_task_clarification_stops_before_next_model_call(self):
    ...

def test_completed_task_continues_to_model(self):
    ...
```

Their coverage is replaced by:

```text
tests.test_deerflow_create_agent_streaming
tests.test_clarification_flow
tests.test_subagent_clarification
```

- [ ] **Step 5: Run the relevant test set**

Run:

```powershell
.\.venv-deerflow-streaming\Scripts\python.exe -m unittest tests.test_deerflow_create_agent_streaming tests.test_lead_agent_factory tests.test_clarification_flow tests.test_subagent_clarification
```

Expected: PASS.

---

### Task 5: Promote the Working Dependency Set to Normal Project Install

**Files:**
- Modify: `G:\work\tcm-flow\requirements.txt`
- Keep or delete after decision: `G:\work\tcm-flow\constraints-deerflow-streaming.txt`

- [ ] **Step 1: Pin the streaming-critical packages in `requirements.txt`**

Replace the unpinned LangChain/LangGraph block:

```text
langchain
langchain-openai
langchain-community
langgraph
```

with:

```text
langchain==1.2.15
langchain-openai==1.2.1
langchain-community
langchain-core==1.3.3
langgraph==1.1.9
langgraph-prebuilt==1.0.11
langgraph-checkpoint==4.0.2
langgraph-sdk==0.3.13
```

Keep `langgraph-checkpoint-postgres` in the file. If pip resolves it to a version incompatible with `langgraph-checkpoint==4.0.2`, pin `langgraph-checkpoint-postgres==3.0.5`, matching DeerFlow's lock.

- [ ] **Step 2: Rebuild the main virtualenv with the pinned requirements**

Run:

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest tests.test_deerflow_create_agent_streaming
```

Expected: parity test PASS in the main `.venv`.

- [ ] **Step 3: Decide whether to keep the constraints file**

If `requirements.txt` now pins all streaming-critical versions, delete the temporary constraints file:

```powershell
Remove-Item -LiteralPath .\constraints-deerflow-streaming.txt
```

If the team wants constraints preserved for future diagnosis, keep it and add a comment at the top:

```text
# Temporary DeerFlow streaming parity constraints.
# Keep only while validating LangChain/LangGraph streaming behavior.
```

---

### Task 6: Verify the Real SSE Contract End to End

**Files:**
- Verify: `G:\work\tcm-flow\app\runtime\runs\worker.py`
- Verify: `G:\work\tcm-flow\scripts\chat.sh`
- Verify: `G:\work\tcm-flow\docs\RUNBOOK.md`

- [ ] **Step 1: Confirm worker still only forwards LangGraph messages**

Check `G:\work\tcm-flow\app\runtime\runs\worker.py` keeps this behavior:

```python
            if stream_event == "messages":
                message, metadata = chunk
                await bridge.publish(
                    run_id,
                    "messages",
                    [
                        _stream_message_to_dict(message),
                        metadata,
                    ],
                )
                continue
```

Do not add any function that splits `message.content` into characters or words.

- [ ] **Step 2: Run backend stream tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.gateway.test_threads_router tests.gateway.test_thread_run_services tests.test_deerflow_create_agent_streaming
```

Expected: PASS.

- [ ] **Step 3: Run all backend tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Expected: PASS.

- [ ] **Step 4: Run a live SSE smoke only when API credentials are available**

Use this only if `OPENAI_API_KEY` or the configured compatible provider key is present in the shell:

```powershell
$env:STREAM_MODE="messages"
bash .\scripts\chat.sh "请用一句话回答：什么是太阳？"
```

Expected SSE sequence contains multiple `event: messages` frames whose first payload object has:

```json
{"type":"AIMessageChunk"}
```

and ends with:

```text
event: final
```

If credentials are not available, skip this step and state that live provider verification was not run.

---

### Task 7: Failure Path if DeerFlow-Style `create_agent` Still Does Not Stream

**Files:**
- Create: `G:\work\tcm-flow\docs\deerflow-streaming-parity-failure.md`
- Do not modify production agent construction in this task

- [ ] **Step 1: Write the failure note**

Create `G:\work\tcm-flow\docs\deerflow-streaming-parity-failure.md`:

```markdown
# DeerFlow Streaming Parity Failure Note

## Goal

Verify whether tcm-flow can implement token-level typewriter output using DeerFlow's direct pattern:

```python
agent = create_agent(...)
agent.astream(..., stream_mode=["messages", "values"])
```

## Observed Result

The parity test `tests.test_deerflow_create_agent_streaming` did not emit multiple `AIMessageChunk` payloads.

## Installed Versions

Paste the exact output of:

```powershell
.\.venv-deerflow-streaming\Scripts\python.exe - <<'PY'
from importlib.metadata import version
for package in ["langchain", "langgraph", "langchain-core", "langchain-openai", "langgraph-prebuilt", "langgraph-checkpoint", "langgraph-sdk"]:
    print(f"{package}=={version(package)}")
PY
```

## Event Sequence

Paste the exact `chunks` list printed or asserted by `tests.test_deerflow_create_agent_streaming`.

## Decision

Do not implement synthetic chunking. The next investigation must compare model provider behavior and middleware behavior against DeerFlow before changing production streaming code.
```

- [ ] **Step 2: Add a diagnostic print command for the failing parity test**

Run:

```powershell
.\.venv-deerflow-streaming\Scripts\python.exe - <<'PY'
import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool

class ToolBindableStreamingFakeModel(FakeListChatModel):
    def bind_tools(self, tools: Sequence[dict[str, Any] | type | Callable | BaseTool], *, tool_choice: str | None = None, **kwargs: Any):
        return self

async def main():
    agent = create_agent(model=ToolBindableStreamingFakeModel(responses=["abc"]), tools=[], system_prompt="You are concise.")
    seen = []
    async for event in agent.astream({"messages": [{"role": "user", "content": "say abc"}]}, stream_mode=["messages", "values"]):
        if isinstance(event, tuple) and event[0] == "messages":
            message, metadata = event[1]
            seen.append((message.__class__.__name__, getattr(message, "type", None), str(message.content), metadata))
    print(seen)

asyncio.run(main())
PY
```

Expected: exact printed list is copied into the failure note.

- [ ] **Step 3: Stop implementation and report the blocker**

Do not merge `create_streaming_react_agent` as the final answer if this task is reached. Report that DeerFlow's direct pattern could not be reproduced in tcm-flow even after dependency alignment, with the failure note path and observed versions.

---

### Task 8: Optional Frontend Typewriter Verification

**Files:**
- Modify only if backend emits multiple `AIMessageChunk` but UI still does not type:
  - `G:\work\tcm-consultation-system\tcm-web\src\api\tcmFlow.ts`
  - `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\PatientIntakeWorkspace.tsx`
  - Existing frontend tests under `G:\work\tcm-consultation-system\tcm-web\src`

- [ ] **Step 1: Confirm backend wire payload before touching frontend**

Run the backend SSE smoke from Task 6 Step 4 and save three consecutive `event: messages` frames. The frontend task starts only if those frames exist and contain `AIMessageChunk`.

- [ ] **Step 2: Verify frontend parser treats `AIMessageChunk` as provisional text**

In `PatientIntakeWorkspace.tsx`, confirm event handling appends only `data[0].content` for:

```ts
event === 'messages' && data?.[0]?.type === 'AIMessageChunk'
```

Expected behavior: provisional assistant bubble grows as chunks arrive, then `final.assistant_message` replaces the final bubble content.

- [ ] **Step 3: Run frontend tests**

Run from `G:\work\tcm-consultation-system\tcm-web`:

```powershell
pnpm test -- src/api/tcmFlow.test.ts src/App.test.tsx
pnpm build
pnpm lint
```

Expected: PASS.

---

## Self-Review

- Spec coverage: The plan starts with a failing parity test, aligns dependencies to DeerFlow, restores `create_agent` only if the parity test passes, preserves worker pass-through, and verifies backend SSE. This covers the user requirement to imitate DeerFlow instead of keeping a custom local construction path.
- Placeholder scan: No `TBD`, `TODO`, or "implement later" steps are present. Failure handling is explicit in Task 7.
- Type consistency: The plan consistently uses `AIMessageChunk`, `create_agent`, `agent.astream`, `stream_mode=["messages", "values"]`, and the existing `unittest` style used by the repo.
