# TCM Web Multi-Agent Collaboration Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `tcm-web` show the real `workflow_agent` collaboration lifecycle for each assistant reply, including live Agent status, concise result summaries, automatic expand/collapse behavior, and trace restoration from consultation history.

**Architecture:** Keep LangGraph nodes as the source of truth. The Spring client requests `tasks` and `updates` in addition to `messages`; `tcm-web` parses those raw stream modes into a fixed six-Agent presentation model. At run completion, `tcm-flow` attaches the current run's `agent_trace` to the matching assistant entry in the public `conversation` projection, and Spring prefers that enriched projection for history while retaining raw `messages` as a backward-compatible fallback.

**Tech Stack:** Python 3.10, FastAPI, LangGraph 1.2.4, PostgreSQL JSONB thread metadata, Java 21, Spring Boot 4.0.6, Jackson, JUnit 5, React 19, TypeScript 6, Zod 4, Vitest, Testing Library, CSS.

---

## Confirmed Product Decisions

- Show each participating business Agent's name, lifecycle state, and a short user-safe output summary.
- Do not render prompts, chain-of-thought, raw model reasoning, full tool arguments, or complete internal state.
- Use the existing inline per-assistant process surface; rename it from `思考过程` to `多智能体协作`.
- Automatically expand the active collaboration card while a run is streaming.
- Automatically collapse the card after `final`, `clarification`, `end`, or `error`; users can reopen completed cards manually.
- Preserve collaboration history after refresh and when reopening an older consultation.
- Use the selected raw-stream approach: `tasks` supplies node start/finish/error lifecycle, while `updates` supplies incremental `agent_trace` results.
- Display the fixed workflow roles in this order:
  1. `IntentAgent` — 意图识别
  2. `InquiryAgent` — 问诊分析
  3. `EvidenceAgent` — 证据检索
  4. `SyndromeAgent` — 证候分析
  5. `AnswerAgent` — 回答生成
  6. `SafetyAgent` — 安全审查
- Mark roles that were not entered by a conditional graph branch as `skipped` when the run reaches a terminal event.
- Merge repeated rewrite nodes into the same `AnswerAgent` or `SafetyAgent` row rather than creating duplicate rows.

## Ground Rules

- Follow TDD for every behavior change: write the focused failing test, run it and confirm the expected failure, implement the minimum change, then rerun it.
- Do not change `.env`, credentials, unrelated RAG code, `lead_agent`, or the existing final response fields.
- Preserve `final.assistant_message`, `pending_clarification`, and `references`.
- Keep graph execution state in the LangGraph checkpointer. Do not forge `AIMessage.tool_calls` or `ToolMessage` objects to represent Agent nodes.
- Keep `thread_store` as the product-facing projection. Persist collaboration history on `conversation` assistant entries, not in LangGraph messages.
- Strip `run_id` and `agent_trace` before visible conversation is passed back into a future model call.
- Do not add a database migration. `app_threads.metadata` is JSONB and already stores `conversation`.
- Keep raw `messages` history as a fallback for pre-change threads.
- Do not commit unless the user explicitly requests it. `G:\work\tcm-consultation-system` is currently not a Git repository.

## Current-State Facts

- `CHECKPOINT_BACKEND=postgres`; LangGraph execution state survives process restarts.
- `run_agent()` always subscribes internally to `messages` and `values`, and forwards additionally requested modes such as `tasks` and `updates` as SSE `event: updates`.
- Installed LangGraph declares `StreamMode` values `values`, `updates`, `checkpoints`, `tasks`, `debug`, `messages`, and `custom`.
- `tasks` emits node start and finish events; `updates` emits node names plus node output after each graph step.
- `TCMWorkflow.astream()` slices `values.agent_trace` to the current run, so the runtime can persist one run's trace without mixing earlier turns.
- `tcm-flow /api/threads/{thread_id}/history` already returns both `conversation` and `messages`.
- Spring's `TcmFlowClient.HistoryResponse` currently deserializes only `messages`, and `ConsultationFlowServiceImpl.listMessages()` returns only that raw chain.
- The live Spring SSE path does not insert into the Spring `consultation_message` table.
- `tcm-web` already stores per-assistant process events and renders an inline `ThinkingProcess` card, but it currently understands tool events rather than LangGraph Agent nodes.

## Target File Structure

### `G:\work\tcm-flow`

Create:

- `tests/test_collaboration_history.py`
  - Verifies per-run trace persistence, no-trace compatibility, and model-input sanitization.

Modify:

- `app/runtime/runs/worker.py`
  - Tracks the latest current-run `agent_trace` from `values` and attaches it to the visible assistant turn.
- `app/agents/workflow_agent/agent.py`
  - Reduces stored conversation entries back to `{role, content}` before graph invocation.
- `tests/gateway/test_threads_router.py`
  - Verifies `/history` returns enriched `conversation` without changing raw `messages`.

Do not modify:

- `app/runtime/public_messages.py`; trace metadata belongs in the runtime projection write, not the thin final response builder.
- `app/agents/workflow_agent/graph.py`; existing `agent_trace` is already the authoritative structured result.
- `app/db/schema.sql`; JSONB metadata already supports the new fields.

### `G:\work\tcm-consultation-system\tcm-backend`

Modify:

- `src/main/java/com/tcm/consultation/integration/tcmflow/TcmFlowClient.java`
  - Requests `messages`, `tasks`, and `updates` and deserializes both history projections.
- `src/main/java/com/tcm/consultation/service/impl/ConsultationFlowServiceImpl.java`
  - Returns enriched `conversation` when present, otherwise raw `messages`.
- `src/test/java/com/tcm/consultation/integration/tcmflow/TcmFlowClientHistoryTest.java`
  - Covers stream modes, conversation preservation, and legacy fallback.
- `src/test/java/com/tcm/consultation/service/impl/ConsultationFlowServiceImplTest.java`
  - Covers service preference for conversation and fallback to messages.

### `G:\work\tcm-consultation-system\tcm-web`

Create:

- `src/features/consultation/collaboration.ts`
  - Owns the six-Agent presentation model, node mappings, raw event reduction, safe summaries, terminal reconciliation, and history restoration.
- `src/features/consultation/collaboration.test.ts`
  - Unit-tests lifecycle parsing, conditional skips, rewrites, summaries, and error handling.

Modify:

- `src/api/consultation.ts`
  - Accepts enriched `conversation` history entries with optional `run_id` and `agent_trace`.
- `src/features/consultation/tcmFlowHistory.ts`
  - Restores assistant messages and collaboration rows from enriched conversation; preserves raw-message fallback.
- `src/features/consultation/tcmFlowHistory.test.ts`
  - Covers enriched history and old raw history.
- `src/features/patient/PatientIntakeWorkspace.tsx`
  - Routes metadata, tasks, updates, and terminal events through the collaboration reducer for the correct assistant message.
- `src/features/consultation/ConsultationChatPanel.tsx`
  - Renders the collaboration card and controls automatic expansion/collapse.
- `src/App.css`
  - Styles pending/running/completed/skipped/failed rows without obscuring the chat.
- `src/App.test.tsx`
  - Proves the live and restored user journeys.

---

### Task 1: Persist Each Run's Agent Trace on Its Visible Assistant Turn

**Files:**
- Create: `tests/test_collaboration_history.py`
- Modify: `app/runtime/runs/worker.py`
- Test: `tests/test_collaboration_history.py`

- [ ] **Step 1: Write a failing runtime test for per-turn trace persistence**

Create `tests/test_collaboration_history.py` with a focused asynchronous test using the existing in-memory `ThreadStore`, `RunManager`, and `StreamBridge`. The fake Agent must yield one `values` snapshot containing messages and the current run trace:

```python
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from app.runtime.runs.worker import run_agent
from app.runtime.stream import StreamBridge
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


class TraceAgent:
    async def aget_state(self, config):
        class Snapshot:
            values = {"messages": []}

        return Snapshot()

    async def astream(self, input_data, *, config, stream_mode):
        yield (
            "values",
            {
                "messages": [
                    HumanMessage(content="最近头痛", id="human-1"),
                    AIMessage(content="请补充持续时间。", id="ai-1"),
                ],
                "agent_trace": [
                    {
                        "agent": "IntentAgent",
                        "primary_intent": "symptom_consultation",
                    },
                    {
                        "agent": "InquiryAgent",
                        "information_sufficiency": "insufficient",
                        "should_pause_for_clarification": True,
                    },
                ],
            },
        )


class CollaborationHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_persists_current_agent_trace_on_assistant_conversation(self):
        thread_store = ThreadStore()
        run_manager = RunManager()
        bridge = StreamBridge()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "workflow_agent")
        bridge.create(run.run_id)

        await run_agent(
            bridge=bridge,
            run_manager=run_manager,
            thread_store=thread_store,
            record=run,
            agent_factory=lambda context: TraceAgent(),
            input_data={"messages": [{"type": "human", "content": "最近头痛"}]},
            context={"stream_mode": ["messages", "updates", "tasks"]},
        )

        stored = await thread_store.get(thread.thread_id)
        assistant_turn = stored.values["conversation"][-1]

        self.assertEqual(assistant_turn["role"], "assistant")
        self.assertEqual(assistant_turn["run_id"], run.run_id)
        self.assertEqual(
            [step["agent"] for step in assistant_turn["agent_trace"]],
            ["IntentAgent", "InquiryAgent"],
        )
```

- [ ] **Step 2: Run the test and confirm the expected failure**

Run from `G:\work\tcm-flow`:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_collaboration_history.CollaborationHistoryTests.test_run_persists_current_agent_trace_on_assistant_conversation
```

Expected: FAIL because the stored assistant conversation entry has no `run_id` or `agent_trace`.

- [ ] **Step 3: Track the current run trace in `run_agent()`**

Add a runtime-local accumulator next to `final_messages`:

```python
final_messages: list[dict[str, Any]] = []
current_run_agent_trace: list[dict[str, Any]] = []
```

After `serialized_values` is created, copy only dictionary trace entries:

```python
if isinstance(serialized_values, dict):
    raw_agent_trace = serialized_values.get("agent_trace")
    if isinstance(raw_agent_trace, list):
        current_run_agent_trace = [
            dict(item) for item in raw_agent_trace if isinstance(item, dict)
        ]
```

- [ ] **Step 4: Attach run metadata only to the assistant projection**

Extend `append_visible_messages()` in `app/runtime/runs/worker.py`:

```python
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

Pass `run_id=run_id` and `agent_trace=current_run_agent_trace` in both the clarification and completed-answer branches.

- [ ] **Step 5: Run the focused test and existing runtime contract tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_collaboration_history tests.test_workflow_agent_flow tests.gateway.test_threads_router
```

Expected: PASS. If old assertions expect exact two-key conversation dictionaries, update only those assertions to include `run_id` and `agent_trace` for workflow-agent turns.

### Task 2: Keep Stored Trace Metadata Out of Future Model Context

**Files:**
- Modify: `app/agents/workflow_agent/agent.py`
- Modify: `tests/test_collaboration_history.py`
- Modify: `tests/gateway/test_threads_router.py`
- Test: `tests/test_collaboration_history.py`
- Test: `tests/gateway/test_threads_router.py`

- [ ] **Step 1: Write a failing test for conversation sanitization**

Add this test to `tests/test_collaboration_history.py`:

```python
from app.agents.workflow_agent.agent import WorkflowAgent


class RecordingWorkflow:
    def __init__(self):
        self.conversation = None

    async def astream(self, *, user_text, conversation, config, stream_mode):
        self.conversation = conversation
        yield "values", {"messages": [], "agent_trace": []}


class CollaborationModelInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_reads_only_role_and_content_from_stored_conversation(self):
        thread_store = ThreadStore()
        thread = await thread_store.create()
        await thread_store.update_values(
            thread.thread_id,
            {
                "conversation": [
                    {"role": "user", "content": "最近头痛"},
                    {
                        "role": "assistant",
                        "content": "请补充持续时间。",
                        "run_id": "run-1",
                        "agent_trace": [{"agent": "InquiryAgent"}],
                    },
                ]
            },
        )
        workflow = RecordingWorkflow()
        agent = WorkflowAgent(workflow=workflow, thread_store=thread_store)

        events = [
            event
            async for event in agent.astream(
                {"messages": [{"role": "user", "content": "已经两周"}]},
                config={"configurable": {"thread_id": thread.thread_id}},
                stream_mode=["values"],
            )
        ]

        self.assertEqual(len(events), 1)
        self.assertEqual(
            workflow.conversation,
            [
                {"role": "user", "content": "最近头痛"},
                {"role": "assistant", "content": "请补充持续时间。"},
            ],
        )
```

- [ ] **Step 2: Run the sanitization test and confirm it fails**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_collaboration_history.CollaborationModelInputTests
```

Expected: FAIL because `_read_conversation()` currently returns every stored field.

- [ ] **Step 3: Sanitize stored conversation in `WorkflowAgent`**

Replace `_read_conversation()` with:

```python
async def _read_conversation(self, thread_id: str) -> list[dict[str, str]]:
    values = await self._read_values(thread_id)
    conversation = values.get("conversation") or []
    result: list[dict[str, str]] = []
    for item in conversation:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant", "system"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        result.append({"role": role, "content": content})
    return result
```

- [ ] **Step 4: Prove `/history` exposes enriched conversation but unchanged messages**

Add a gateway test that stores:

```python
conversation = [
    {"role": "user", "content": "最近头痛"},
    {
        "role": "assistant",
        "content": "请补充持续时间。",
        "run_id": "run-1",
        "agent_trace": [{"agent": "InquiryAgent"}],
    },
]
```

Then assert:

```python
self.assertEqual(body["conversation"], conversation)
self.assertEqual(body["messages"], messages)
```

- [ ] **Step 5: Run the Python collaboration and gateway suites**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_collaboration_history tests.gateway.test_threads_router
```

Expected: PASS with no raw trace metadata included in `workflow.conversation`.

### Task 3: Request LangGraph Lifecycle Events and Prefer Enriched History in Spring

**Files:**
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\main\java\com\tcm\consultation\integration\tcmflow\TcmFlowClient.java`
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\main\java\com\tcm\consultation\service\impl\ConsultationFlowServiceImpl.java`
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\test\java\com\tcm\consultation\integration\tcmflow\TcmFlowClientHistoryTest.java`
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\test\java\com\tcm\consultation\service\impl\ConsultationFlowServiceImplTest.java`

- [ ] **Step 1: Write failing tests for stream modes and history selection**

Add to `TcmFlowClientHistoryTest`:

```java
@Test
void workflowRequestIncludesMessagesTasksAndUpdates() {
    Map<String, Object> body = TcmFlowClient.buildStreamRequestBody("最近头痛");

    assertEquals(List.of("messages", "tasks", "updates"), body.get("stream_mode"));
}

@Test
void visibleMessagesPreferEnrichedConversation() throws Exception {
    JsonNode conversation = objectMapper.readTree("""
        {
          "role": "assistant",
          "content": "请补充持续时间。",
          "run_id": "run-1",
          "agent_trace": [{"agent": "InquiryAgent"}]
        }
        """);
    JsonNode rawMessage = objectMapper.readTree("""
        {"type": "ai", "content": "raw"}
        """);

    TcmFlowClient.HistoryResponse history = new TcmFlowClient.HistoryResponse(
        List.of(conversation),
        List.of(rawMessage)
    );

    assertSame(conversation, history.visibleMessages().getFirst());
}
```

Update `ConsultationFlowServiceImplTest` to construct both lists and assert `listMessages()` returns `conversation`. Add a second test with an empty conversation and assert the raw message fallback is returned.

- [ ] **Step 2: Run the two Spring tests and confirm they fail**

From `G:\work\tcm-consultation-system`:

```powershell
mvn -f tcm-backend\pom.xml -Dtest=TcmFlowClientHistoryTest,ConsultationFlowServiceImplTest test
```

Expected: FAIL because `buildStreamRequestBody`, the two-list `HistoryResponse`, and `visibleMessages()` do not yet exist.

- [ ] **Step 3: Extract the exact workflow request body**

Add this package-visible helper to `TcmFlowClient` and use it from `streamRun()`:

```java
static Map<String, Object> buildStreamRequestBody(String content) {
    return Map.of(
        "assistant_id", "workflow_agent",
        "input", Map.of(
            "messages", List.of(
                Map.of(
                    "type", "human",
                    "content", List.of(Map.of("type", "text", "text", content))
                )
            )
        ),
        "stream_mode", List.of("messages", "tasks", "updates"),
        "stream_subgraphs", false,
        "config", Map.of("recursion_limit", 50),
        "context", Map.of("recursion_limit", 50, "subagent_enabled", true)
    );
}
```

Replace the existing inline request-body construction with:

```java
Map<String, Object> body = buildStreamRequestBody(content);
```

- [ ] **Step 4: Preserve both history projections and provide a compatibility selector**

Replace `HistoryResponse` with:

```java
public record HistoryResponse(
    List<JsonNode> conversation,
    List<JsonNode> messages
) {
    public HistoryResponse {
        conversation = conversation == null ? List.of() : List.copyOf(conversation);
        messages = messages == null ? List.of() : List.copyOf(messages);
    }

    public List<JsonNode> visibleMessages() {
        return conversation.isEmpty() ? messages : conversation;
    }
}
```

Change `ConsultationFlowServiceImpl.listMessages()` from `.messages()` to `.visibleMessages()`.

- [ ] **Step 5: Run the Spring tests and compile**

```powershell
mvn -f tcm-backend\pom.xml -Dtest=TcmFlowClientHistoryTest,ConsultationFlowServiceImplTest test
mvn -f tcm-backend\pom.xml -DskipTests compile
```

Expected: both commands exit `0`.

### Task 4: Build a Pure Frontend Collaboration Event Reducer

**Files:**
- Create: `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\collaboration.ts`
- Create: `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\collaboration.test.ts`

- [ ] **Step 1: Define failing lifecycle tests**

Create `collaboration.test.ts` with tests that prove:

```typescript
import { describe, expect, it } from 'vitest'
import {
  applyCollaborationSseEvent,
  createWorkflowSteps,
  finishCollaboration,
  restoreCollaborationFromTrace,
} from './collaboration'

describe('collaboration reducer', () => {
  it('moves a LangGraph task from pending to running and then completed', () => {
    const pending = createWorkflowSteps()
    const running = applyCollaborationSseEvent(pending, {
      event: 'updates',
      data: {
        stream_event: 'tasks',
        data: { id: 'task-1', name: 'evidence', input: {}, triggers: ['inquiry'] },
      },
    })
    const completed = applyCollaborationSseEvent(running, {
      event: 'updates',
      data: {
        evidence: {
          agent_trace: [
            {
              agent: 'EvidenceAgent',
              retrieval_status: 'success',
              evidence_count: 5,
            },
          ],
        },
      },
    })

    expect(running.find((step) => step.agent === 'EvidenceAgent')?.status).toBe('running')
    expect(completed.find((step) => step.agent === 'EvidenceAgent')).toMatchObject({
      status: 'completed',
      summary: '已完成中医证据检索，共获得 5 条相关依据',
    })
  })

  it('marks unvisited conditional roles as skipped at final', () => {
    const completed = finishCollaboration(createWorkflowSteps(), 'completed')
    expect(completed.every((step) => step.status === 'skipped')).toBe(true)
  })

  it('marks the active role failed on stream error', () => {
    const running = applyCollaborationSseEvent(createWorkflowSteps(), {
      event: 'updates',
      data: {
        stream_event: 'tasks',
        data: { id: 'task-1', name: 'syndrome', input: {}, triggers: ['evidence'] },
      },
    })
    const failed = finishCollaboration(running, 'failed')
    expect(failed.find((step) => step.agent === 'SyndromeAgent')?.status).toBe('failed')
  })

  it('merges rewrite traces into the existing AnswerAgent row', () => {
    const restored = restoreCollaborationFromTrace([
      { agent: 'AnswerAgent', stage: 'draft' },
      { agent: 'SafetyAgent', stage: 'initial', rewrite_required: true },
      { agent: 'AnswerAgent', stage: 'rewrite' },
      { agent: 'SafetyAgent', stage: 'rewrite', rewrite_required: false },
    ])
    expect(restored.filter((step) => step.agent === 'AnswerAgent')).toHaveLength(1)
    expect(restored.find((step) => step.agent === 'AnswerAgent')?.summary).toBe('已根据安全审查完成回答修订')
  })
})
```

- [ ] **Step 2: Run the reducer test and confirm module-not-found failure**

```powershell
pnpm --dir G:\work\tcm-consultation-system --filter tcm-web test -- src/features/consultation/collaboration.test.ts
```

Expected: FAIL because `collaboration.ts` does not exist.

- [ ] **Step 3: Implement the fixed presentation model and node mapping**

Create `collaboration.ts` with these public types and mappings:

```typescript
import type { TcmFlowSseEvent } from '../../api/consultation'

export type CollaborationStatus = 'pending' | 'running' | 'completed' | 'skipped' | 'failed'
export type CollaborationAgent =
  | 'IntentAgent'
  | 'InquiryAgent'
  | 'EvidenceAgent'
  | 'SyndromeAgent'
  | 'AnswerAgent'
  | 'SafetyAgent'

export type CollaborationStep = {
  id: string
  agent: CollaborationAgent
  label: string
  status: CollaborationStatus
  summary: string
}

const AGENT_DEFINITIONS: Array<Pick<CollaborationStep, 'agent' | 'label'>> = [
  { agent: 'IntentAgent', label: '意图识别 Agent' },
  { agent: 'InquiryAgent', label: '问诊分析 Agent' },
  { agent: 'EvidenceAgent', label: '证据检索 Agent' },
  { agent: 'SyndromeAgent', label: '证候分析 Agent' },
  { agent: 'AnswerAgent', label: '回答生成 Agent' },
  { agent: 'SafetyAgent', label: '安全审查 Agent' },
]

const NODE_TO_AGENT: Record<string, CollaborationAgent | undefined> = {
  intent: 'IntentAgent',
  direct_response: 'IntentAgent',
  inquiry: 'InquiryAgent',
  clarification: 'InquiryAgent',
  evidence: 'EvidenceAgent',
  syndrome: 'SyndromeAgent',
  answer_draft: 'AnswerAgent',
  answer_rewrite: 'AnswerAgent',
  safety_initial: 'SafetyAgent',
  safety_rewrite: 'SafetyAgent',
  safe_fallback: 'SafetyAgent',
}

export function createWorkflowSteps(): CollaborationStep[] {
  return AGENT_DEFINITIONS.map(({ agent, label }) => ({
    id: `agent:${agent}`,
    agent,
    label,
    status: 'pending',
    summary: '等待执行',
  }))
}
```

- [ ] **Step 4: Implement guarded parsing and safe summaries**

Add private guards for unknown SSE data. For `tasks`, use `data.name` only and never copy `input` or `result` into display state. For `updates`, iterate each node update's `agent_trace` and update by canonical Agent name.

Implement summaries with exact behavior:

```typescript
function summarizeTrace(agent: CollaborationAgent, trace: Record<string, unknown>): string {
  if (agent === 'IntentAgent') return '已识别咨询意图并确定处理路线'
  if (agent === 'InquiryAgent') {
    return trace.should_pause_for_clarification === true
      ? '已评估问诊信息，仍需补充关键情况'
      : '已完成问诊信息完整度评估'
  }
  if (agent === 'EvidenceAgent') {
    const count = typeof trace.evidence_count === 'number' ? trace.evidence_count : 0
    return count > 0 ? `已完成中医证据检索，共获得 ${count} 条相关依据` : '已完成中医证据检索，暂未获得充分依据'
  }
  if (agent === 'SyndromeAgent') return '已完成证候候选分析'
  if (agent === 'AnswerAgent') {
    return trace.stage === 'rewrite' ? '已根据安全审查完成回答修订' : '已生成本轮回答草稿'
  }
  if (trace.rewrite_required === true) return '已完成安全审查，回答需要调整'
  return '已完成回答安全审查'
}
```

Export:

```typescript
export function applyCollaborationSseEvent(
  current: CollaborationStep[],
  event: TcmFlowSseEvent,
): CollaborationStep[]

export function finishCollaboration(
  current: CollaborationStep[],
  outcome: 'completed' | 'failed',
): CollaborationStep[]

export function restoreCollaborationFromTrace(
  trace: Array<Record<string, unknown>>,
): CollaborationStep[]
```

`finishCollaboration(current, 'completed')` keeps completed rows and marks pending rows skipped. `finishCollaboration(current, 'failed')` marks running rows failed and pending rows skipped.

- [ ] **Step 5: Run the reducer tests**

```powershell
pnpm --dir G:\work\tcm-consultation-system --filter tcm-web test -- src/features/consultation/collaboration.test.ts
```

Expected: PASS.

### Task 5: Restore Enriched Collaboration History with Legacy Fallback

**Files:**
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\api\consultation.ts`
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\tcmFlowHistory.ts`
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\tcmFlowHistory.test.ts`

- [ ] **Step 1: Write a failing enriched-history test**

Add a test input containing `role`, `run_id`, and `agent_trace`:

```typescript
it('restores collaboration steps from enriched conversation history', () => {
  const restored = restoreTcmFlowHistory(101, [
    { role: 'user', content: '最近头痛。' },
    {
      role: 'assistant',
      content: '请补充持续时间。',
      run_id: 'run-1',
      agent_trace: [
        { agent: 'IntentAgent', primary_intent: 'symptom_consultation' },
        {
          agent: 'InquiryAgent',
          information_sufficiency: 'insufficient',
          should_pause_for_clarification: true,
        },
      ],
    },
  ])

  expect(restored.messages.map(({ role, content }) => ({ role, content }))).toEqual([
    { role: 'USER', content: '最近头痛。' },
    { role: 'ASSISTANT', content: '请补充持续时间。' },
  ])
  const assistantId = restored.messages[1].id
  expect(restored.collaborationByMessageId[assistantId].map((step) => step.agent)).toEqual([
    'IntentAgent',
    'InquiryAgent',
    'EvidenceAgent',
    'SyndromeAgent',
    'AnswerAgent',
    'SafetyAgent',
  ])
  expect(restored.collaborationByMessageId[assistantId][0].status).toBe('completed')
  expect(restored.collaborationByMessageId[assistantId][2].status).toBe('skipped')
})
```

Keep the existing raw tool-call restoration test unchanged to prove backward compatibility.

- [ ] **Step 2: Run the history tests and confirm failure**

```powershell
pnpm --dir G:\work\tcm-consultation-system --filter tcm-web test -- src/features/consultation/tcmFlowHistory.test.ts
```

Expected: FAIL because the API schema requires `type` and the restored structure has no collaboration map.

- [ ] **Step 3: Extend the history schema without weakening content validation**

Update `tcmFlowMessageSchema` so a history item accepts either raw `type` or visible `role`:

```typescript
const tcmFlowTraceItemSchema = z.record(z.string(), z.unknown())

const tcmFlowMessageSchema = z
  .object({
    id: z.string().nullable().optional(),
    type: z.string().optional(),
    role: z.enum(['user', 'assistant', 'system']).optional(),
    content: z.string(),
    run_id: z.string().nullable().optional(),
    agent_trace: z.array(tcmFlowTraceItemSchema).optional(),
    name: nullableString,
    tool_calls: z.array(z.record(z.string(), z.unknown())).nullable().optional(),
    tool_call_chunks: z.array(z.record(z.string(), z.unknown())).nullable().optional(),
    tool_call_id: nullableString,
    status: nullableString,
  })
  .refine((message) => Boolean(message.type || message.role), {
    message: 'tcm-flow history message requires type or role',
  })
  .passthrough()
```

- [ ] **Step 4: Restore visible conversation first and raw messages second**

Extend `RestoredTcmFlowHistory` with:

```typescript
collaborationByMessageId: Record<number, CollaborationStep[]>
```

When `message.role` is present, map `user` to `USER` and `assistant` to `ASSISTANT`. For assistant entries with `agent_trace`, call `restoreCollaborationFromTrace()` and store the result under the generated display message ID. Keep the existing `type`-based raw-message branch for legacy history.

- [ ] **Step 5: Run history, API, and reducer tests**

```powershell
pnpm --dir G:\work\tcm-consultation-system --filter tcm-web test -- src/features/consultation/tcmFlowHistory.test.ts src/features/consultation/collaboration.test.ts src/api/consultation.test.ts
```

Expected: PASS for enriched conversation and legacy raw history.

### Task 6: Integrate the Live Collaboration UI and Automatic Expansion

**Files:**
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\features\patient\PatientIntakeWorkspace.tsx`
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\ConsultationChatPanel.tsx`
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\App.css`
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\App.test.tsx`

- [ ] **Step 1: Replace the existing tool-only App test with a failing workflow lifecycle test**

Build a controllable SSE response containing:

```typescript
[
  {
    event: 'metadata',
    data: {
      run_id: 'run-101',
      thread_id: 'thread-101',
      assistant_id: 'workflow_agent',
      architecture: 'tcm-flow',
    },
  },
  {
    event: 'updates',
    data: {
      stream_event: 'tasks',
      data: { id: 'task-intent', name: 'intent', input: {}, triggers: ['__start__'] },
    },
  },
  {
    event: 'updates',
    data: {
      intent: {
        agent_trace: [{ agent: 'IntentAgent', primary_intent: 'symptom_consultation' }],
      },
    },
  },
  {
    event: 'updates',
    data: {
      stream_event: 'tasks',
      data: { id: 'task-evidence', name: 'evidence', input: {}, triggers: ['inquiry'] },
    },
  },
]
```

Before the terminal event, assert:

```typescript
const collaboration = await screen.findByRole('button', { name: '多智能体协作' })
expect(collaboration).toHaveAttribute('aria-expanded', 'true')
const steps = await screen.findByLabelText('多智能体协作步骤')
expect(within(steps).getByText('意图识别 Agent')).toBeInTheDocument()
expect(within(steps).getByText('证据检索 Agent')).toBeInTheDocument()
expect(within(steps).getByText('正在执行')).toBeInTheDocument()
```

After `final` and `end`, assert the trigger is collapsed and can be reopened manually.

- [ ] **Step 2: Run the App test and confirm failure**

```powershell
pnpm --dir G:\work\tcm-consultation-system --filter tcm-web test -- src/App.test.tsx
```

Expected: FAIL because `updates` does not yet update Agent rows and the UI still says `思考过程`.

- [ ] **Step 3: Route live events through the pure reducer**

In `PatientIntakeWorkspace.tsx`, add state keyed by assistant message ID:

```typescript
const [collaborationByMessageId, setCollaborationByMessageId] = useState<
  Record<number, CollaborationStep[]>
>({})
```

Initialize six pending rows when metadata identifies `workflow_agent`. Apply `updates` through `applyCollaborationSseEvent()`. On `final` or `clarification`, call `finishCollaboration(currentSteps, 'completed')`; on `error`, call `finishCollaboration(currentSteps, 'failed')`. Keep existing tool events only as a legacy path for old `lead_agent` streams.

When history loads, set both restored maps:

```typescript
setTcmFlowEventsByMessageId(restoredHistory.eventsByMessageId)
setCollaborationByMessageId(restoredHistory.collaborationByMessageId)
```

- [ ] **Step 4: Rename and specialize the card**

Change the trigger accessible name and headings:

```tsx
<strong>多智能体协作</strong>
<small>{isStreaming ? '协作进行中' : '协作已完成'}</small>
```

Render each `CollaborationStep` with a semantic status label:

```tsx
const STATUS_LABELS: Record<CollaborationStatus, string> = {
  pending: '等待执行',
  running: '正在执行',
  completed: '已完成',
  skipped: '本轮未执行',
  failed: '执行失败',
}
```

Do not render skipped roles as failures. Do not render raw trace fields.

- [ ] **Step 5: Implement automatic expand/collapse without blocking manual review**

Use an effect keyed by the latest assistant message and send state:

```tsx
useEffect(() => {
  if (isSending && latestAssistantMessageId != null) {
    setExpandedThinkingMessageId(latestAssistantMessageId)
    return
  }
  if (!isSending) {
    setExpandedThinkingMessageId(null)
  }
}, [isSending, latestAssistantMessageId])
```

After completion, the existing trigger remains clickable so users can reopen any completed turn.

- [ ] **Step 6: Add status-aware styling**

Add classes for `.collaboration-step.pending`, `.running`, `.completed`, `.skipped`, and `.failed`. Use the existing green accent for running/completed, muted gray for pending/skipped, and the existing error color for failed. Keep the card width responsive and the details list scrollable so it does not push the reply off-screen.

- [ ] **Step 7: Run focused frontend tests**

```powershell
pnpm --dir G:\work\tcm-consultation-system --filter tcm-web test -- src/features/consultation/collaboration.test.ts src/features/consultation/tcmFlowHistory.test.ts src/api/consultation.test.ts src/App.test.tsx
```

Expected: PASS with live auto-expansion, terminal auto-collapse, manual reopen, per-turn isolation, and history restoration.

### Task 7: Full Verification and Browser QA

**Files:**
- Verify only; do not create screenshots, traces, or temporary scripts inside either repository.

- [ ] **Step 1: Run focused Python verification**

```powershell
cd G:\work\tcm-flow
.\.venv\Scripts\python.exe -m unittest tests.test_collaboration_history tests.test_workflow_agent_flow tests.gateway.test_threads_router
```

Expected: all selected tests pass.

- [ ] **Step 2: Run Spring verification**

```powershell
cd G:\work\tcm-consultation-system
mvn -f tcm-backend\pom.xml -Dtest=TcmFlowClientHistoryTest,ConsultationFlowServiceImplTest test
mvn -f tcm-backend\pom.xml -DskipTests compile
```

Expected: both commands exit `0`.

- [ ] **Step 3: Run full frontend verification**

```powershell
cd G:\work\tcm-consultation-system
pnpm --filter tcm-web test
pnpm --filter tcm-web lint
pnpm --filter tcm-web build
```

Expected: tests, lint, TypeScript compilation, and Vite production build all exit `0`.

- [ ] **Step 4: Run a local end-to-end stream smoke test**

Start the configured `tcm-flow`, Spring backend, and Vite frontend. Create a consultation with enough symptom detail to traverse Evidence, Syndrome, Answer, and Safety. Verify in the browser:

1. The collaboration card opens automatically when streaming starts.
2. Intent, Inquiry, Evidence, Syndrome, Answer, and Safety change state from pending to running/completed in graph order.
3. The assistant answer still streams and the final payload replaces it correctly.
4. The card collapses after completion and reopens on click.
5. No prompt, model reasoning, raw state, tool arguments, or task input appears in the DOM.
6. Refreshing the page restores the same completed collaboration summaries under the correct assistant reply.
7. A clarification branch marks unvisited later roles as `本轮未执行`.
8. A pre-terminal stream failure remains visible and marks the active role `执行失败`.

- [ ] **Step 5: Inspect console and network behavior**

Confirm:

- No React key warnings, state-update warnings, Zod parse errors, or unhandled promise rejections.
- The Spring request body contains `"stream_mode":["messages","tasks","updates"]`.
- SSE `updates` events include `stream_event: "tasks"` for lifecycle and node-keyed `agent_trace` for completion.
- `/api/threads/{thread_id}/history` returns assistant conversation entries with the correct `run_id` and current-run `agent_trace`.
- `/api/consultations/{id}/messages` returns enriched conversation for new threads and still returns raw messages for legacy threads without conversation.

## Acceptance Criteria

- The user can see real LangGraph Agent collaboration rather than inferred or fabricated tool calls.
- The active Agent is visible while work is running, not only after it completes.
- Each Agent row exposes only a stable Chinese label, lifecycle state, and concise safe summary.
- Conditional branches and safety rewrites are represented accurately.
- Each assistant reply owns its own collaboration record; later turns never overwrite earlier ones.
- Collaboration is automatically expanded during streaming and collapsed after terminal completion.
- Refreshing or reopening a consultation restores the correct collaboration process.
- Old consultations with only raw `messages` continue to render.
- Existing SSE terminal-error behavior, Markdown answers, clarification handling, and final answer replacement remain unchanged.
- No schema migration, fake ToolMessage, chain-of-thought display, or duplicate graph-state store is introduced.
