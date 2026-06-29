# LangGraph-Native Stream Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repository-specific streamed business events with same-named LangGraph `messages`, `tasks`, `updates`, and `values` events, durable clarification resume, and `end`/run-status terminal handling across Python, Spring, and React.

**Architecture:** `tcm-flow` makes clarification, guardrails, and `public_response` graph-owned, while its worker forwards native modes and retains only lifecycle responsibilities. Spring transparently proxies the stream and exposes consultation-scoped resume/status APIs. React reduces the native modes, reconciles the assistant from `values.public_response`, and recovers streams that close without `end` through run status.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, LangChain 1.3.4, LangGraph 1.2.4, Java 21, Spring Boot 4.0.6, React 19, TypeScript 6, Vite 8, Vitest 4.

---

## Scope and Repository Safety

This is one coupled protocol migration, not three independent products. The
Python producer, Spring proxy, and React consumer must agree on one contract,
so they remain in a single dependency-ordered plan.

Repository state at plan time:

- `G:\work\tcm-flow` is a Git repository and is clean after the design commit.
- `G:\work\tcm-consultation-system\tcm-backend` is a separate Git repository
  with existing uncommitted collaboration-history changes in files this plan
  must modify.
- `G:\work\tcm-consultation-system\tcm-web` is not currently inside a Git
  repository.

Execution rules:

- Commit the Python tasks frequently in `tcm-flow`.
- Before every Spring edit, capture `git diff` and preserve all pre-existing
  hunks. Do not stage or commit the Spring files automatically while those
  changes overlap; leave a reviewed diff for the user unless they explicitly
  authorize committing the pre-existing work.
- Verify React after each task, but do not invent a repository or commit from
  `tcm-web`.

## File and Responsibility Map

### `tcm-flow`

- Modify `app/schemas.py`: mutually exclusive normal-input and resume-command
  request validation.
- Modify `app/runtime/runs/input.py`: convert request data into graph input or
  LangGraph `Command(resume=...)`.
- Modify `app/runtime/services.py`: pass graph invocation and
  `stream_subgraphs` to the worker.
- Modify `app/runtime/runs/stream_adapter.py`: same-named native mode forwarding
  and latest-values capture.
- Create `app/runtime/runs/state_projection.py`: copy graph-declared public
  state into `thread_store` without interpreting messages.
- Modify `app/runtime/runs/worker.py`: lifecycle-only completion, interruption,
  failure, and persistence.
- Delete `app/runtime/runs/projection.py`: remove post-stream business
  inference after replacement coverage is green.
- Modify `app/gateway/routers/thread_runs.py`: add thread-owned run-status API.
- Modify `app/agents/workflow_agent/state.py`: define `PublicResponse` and
  `run_outcome`.
- Modify `app/agents/workflow_agent/graph.py`: graph-owned guardrail completion,
  visible conversation, clarification preparation, and `interrupt`.
- Modify `app/agents/workflow_agent/workflow.py`: accept normal graph input or
  `Command` resume input.
- Modify `app/agents/workflow_agent/agent.py`: forward `Command` without trying
  to parse it as messages.
- Modify `app/agents/workflow_agent/components/base.py`: tag structured model
  calls `nostream`.
- Create `app/middlewares/guardrail_agent_middleware.py`: keep lead-agent
  guardrails inside its graph execution.
- Modify `app/middlewares/clarification_middleware.py`: interrupt lead-agent
  clarification rather than ending through runtime projection.
- Modify `app/agents/lead_agent/agent.py`: install the new guardrail middleware.

### Spring backend

- Modify `TcmFlowClient.java`: four-mode request, resume request, run-status
  lookup, and transparent SSE forwarding.
- Modify `ConsultationMessageRequest.java`: optional `resumeRunId`.
- Modify `ConsultationFlowService.java` and `ConsultationFlowServiceImpl.java`:
  validate resume ownership and proxy run status.
- Modify `ConsultationController.java`: consultation-scoped run-status endpoint.
- Modify the existing focused tests in the matching integration/service test
  packages.

### React frontend

- Modify `src/api/consultation.ts`: resume target, `end` tracking, bounded status
  recovery, and run-status schema.
- Modify `src/api/consultation.test.ts`: request, terminal, resume, and recovery
  tests.
- Create `src/features/consultation/nativeStream.ts`: pure parsing/reduction of
  `messages`, `updates`, and `values.public_response`.
- Create `src/features/consultation/nativeStream.test.ts`: reducer unit tests.
- Modify `src/features/consultation/collaboration.ts`: consume top-level
  `tasks` and native node `updates`.
- Modify `src/features/consultation/collaboration.test.ts`: native payload tests.
- Modify `PatientIntakeWorkspace.tsx`: integrate native reducers, retain resume
  target, and settle on `end`/`error`.
- Modify `src/App.test.tsx`: end-to-end UI stream fixtures.

---

### Task 1: Add normal-input and resume-command request forms

**Files:**
- Modify: `app/schemas.py`
- Modify: `app/runtime/runs/input.py`
- Modify: `app/runtime/services.py`
- Modify: `app/runtime/runs/worker.py`
- Test: `tests/test_runtime_run_context.py`
- Test: `tests/gateway/test_thread_run_services.py`

- [ ] **Step 1: Write failing request-validation and invocation tests**

Add tests that require exactly one input form and preserve the resume payload:

```python
from pydantic import ValidationError
from langgraph.types import Command

from app.schemas import RunCreateRequest
from app.runtime.runs.input import build_graph_invocation


def test_run_request_accepts_exactly_one_input_form(self):
    normal = RunCreateRequest(
        input={"messages": [{"type": "human", "content": "hello"}]}
    )
    resumed = RunCreateRequest(command={"resume": {"content": "two weeks"}})

    assert normal.command is None
    assert resumed.input is None

    for payload in ({}, {"input": {"messages": []}, "command": {"resume": {"content": "x"}}}):
        with self.assertRaises(ValidationError):
            RunCreateRequest(**payload)


def test_build_graph_invocation_injects_current_run_id_into_resume():
    command = build_graph_invocation(
        input_data=None,
        command_data={"resume": {"content": "two weeks"}},
        run_id="run-new",
    )

    assert isinstance(command, Command)
    assert command.resume == {"content": "two weeks", "run_id": "run-new"}
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```powershell
python -m unittest tests.test_runtime_run_context tests.gateway.test_thread_run_services
```

Expected: failure because `RunCreateRequest.command` and
`build_graph_invocation(...)` do not exist.

- [ ] **Step 3: Implement the request union and graph invocation helper**

Implement the schema contract in `app/schemas.py`:

```python
from pydantic import BaseModel, Field, model_validator


class ResumeValue(BaseModel):
    content: str


class RunCommand(BaseModel):
    resume: ResumeValue


class RunCreateRequest(BaseModel):
    assistant_id: str = "lead_agent"
    input: RunInput | None = None
    command: RunCommand | None = None
    stream_mode: list[str] = Field(default_factory=lambda: ["messages"])
    stream_subgraphs: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input_form(self) -> "RunCreateRequest":
        if (self.input is None) == (self.command is None):
            raise ValueError("exactly one of input or command is required")
        return self
```

Implement the input conversion in `app/runtime/runs/input.py`:

```python
from langgraph.types import Command


def build_graph_invocation(
    *,
    input_data: dict[str, Any] | None,
    command_data: dict[str, Any] | None,
    run_id: str,
) -> dict[str, Any] | Command:
    if command_data is not None:
        resume = dict(command_data["resume"])
        resume["run_id"] = run_id
        return Command(resume=resume)
    return normalize_graph_input(input_data or {"messages": []})
```

Update `start_run(...)` to create the record first, build the invocation using
that record's run ID, and pass `stream_subgraphs=body.stream_subgraphs` into
`run_agent(...)`. Widen the worker's `graph_input` type from `dict[str, Any]` to
`Any` without adding command interpretation to the worker.

- [ ] **Step 4: Run tests and confirm the request paths pass**

Run:

```powershell
python -m unittest tests.test_runtime_run_context tests.gateway.test_thread_run_services
```

Expected: all focused tests pass, including service forwarding of
`stream_subgraphs` and `Command`.

- [ ] **Step 5: Commit the Python request contract**

```powershell
git add app/schemas.py app/runtime/runs/input.py app/runtime/services.py app/runtime/runs/worker.py tests/test_runtime_run_context.py tests/gateway/test_thread_run_services.py
git commit -m "feat: add resumable run request contract"
```

---

### Task 2: Make workflow completion and visible output graph-owned

**Files:**
- Modify: `app/agents/workflow_agent/state.py`
- Modify: `app/agents/workflow_agent/graph.py`
- Modify: `app/agents/workflow_agent/workflow.py`
- Test: `tests/test_workflow_agent_models.py`
- Test: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Write failing public-response and guardrail tests**

Add a completed-workflow assertion that reads state instead of a `final` SSE
event:

```python
async def test_finalize_writes_guarded_public_response_and_conversation(self):
    result = await workflow.graph.ainvoke(
        workflow._initial_state(
            user_text="current question",
            conversation=[],
            run_id="run-1",
            human_message_id="human-1",
        ),
        config={"configurable": {"thread_id": "thread-1"}},
    )

    self.assertEqual(result["run_outcome"], "completed")
    self.assertEqual(
        result["public_response"],
        {
            "status": "completed",
            "assistant_message": "guarded answer",
            "pending_clarification": [],
            "references": [],
        },
    )
    self.assertEqual(result["conversation"][-1]["content"], "guarded answer")
```

Patch `app.agents.workflow_agent.graph.apply_guardrails` to return a rewritten
`final_text` and assert that the last `AIMessage` contains the same rewritten
text.

- [ ] **Step 2: Run workflow tests and confirm failure**

```powershell
python -m unittest tests.test_workflow_agent_models tests.test_workflow_agent_flow.WorkflowLLMBackedTests
```

Expected: failure because `public_response` and `run_outcome` are absent and
guardrails still run after graph execution.

- [ ] **Step 3: Add public state and a single completion node**

Define the stable state types in `state.py`:

```python
class PublicResponse(TypedDict):
    status: Literal["completed", "need_clarification"]
    assistant_message: str
    pending_clarification: list[str]
    references: list[dict[str, Any]]


class WorkflowState(TypedDict, total=False):
    # existing fields
    public_response: PublicResponse | None
    run_outcome: Literal["completed", "need_clarification"] | None
```

Refactor graph completion so direct responses and drafted answers both route to
one node. The node applies the existing guardrail helper before returning:

```python
async def finalize_node(state: WorkflowState) -> dict[str, Any]:
    candidate = str(state.get("final_text") or _answer(state).draft_answer)
    serialized_messages = [serialize_message(message) for message in state.get("messages", [])]
    guarded = await apply_guardrails(candidate, serialized_messages)
    final_text = guarded["final_text"]
    assistant_message = AIMessage(
        content=final_text,
        id=_message_id(state, "final-ai-1"),
    )
    conversation = append_visible_messages(
        {"conversation": state.get("conversation", [])},
        state["user_text"],
        final_text,
        run_id=str(state["run_id"]),
        agent_trace=list(state.get("agent_trace", [])),
    )
    return {
        "needs_clarification": False,
        "final_text": final_text,
        "messages": [assistant_message],
        "conversation": conversation,
        "run_outcome": "completed",
        "public_response": {
            "status": "completed",
            "assistant_message": final_text,
            "pending_clarification": [],
            "references": [],
        },
    }
```

Route `direct_response` to `finalize`, initialize `public_response` and
`run_outcome` to absent/empty transient state on new turns, and make graph state
the source returned by `TCMWorkflow.run(...)`.

- [ ] **Step 4: Run workflow completion tests**

```powershell
python -m unittest tests.test_workflow_agent_models tests.test_workflow_agent_flow.WorkflowLLMBackedTests
```

Expected: all completion, rewrite, and safety-order tests pass.

- [ ] **Step 5: Commit graph-owned completion**

```powershell
git add app/agents/workflow_agent/state.py app/agents/workflow_agent/graph.py app/agents/workflow_agent/workflow.py tests/test_workflow_agent_models.py tests/test_workflow_agent_flow.py
git commit -m "refactor: move workflow completion into graph state"
```

---

### Task 3: Add durable clarification interrupt and resume

**Files:**
- Modify: `app/agents/workflow_agent/graph.py`
- Modify: `app/agents/workflow_agent/workflow.py`
- Modify: `app/agents/workflow_agent/agent.py`
- Test: `tests/test_workflow_agent_flow.py`
- Test: `tests/test_collaboration_history.py`

- [ ] **Step 1: Write a failing pause/resume integration test**

Use the existing fake workflow agents and the real memory checkpointer:

```python
async def test_clarification_interrupt_resumes_same_checkpoint(self):
    config = {"configurable": {"thread_id": "thread-1", "run_id": "run-1"}}
    first_events = [
        event
        async for event in workflow.astream(
            graph_input=workflow._initial_state(
                user_text="headache",
                conversation=[],
                run_id="run-1",
                human_message_id="human-1",
            ),
            config=config,
            stream_mode=["updates", "values"],
        )
    ]
    paused = await workflow.graph.aget_state(config)

    self.assertTrue(paused.next)
    self.assertEqual(paused.values["public_response"]["status"], "need_clarification")

    resume = Command(resume={"content": "two weeks", "run_id": "run-2"})
    resumed_events = [
        event
        async for event in workflow.astream(
            graph_input=resume,
            config=config,
            stream_mode=["updates", "values"],
        )
    ]
    completed = await workflow.graph.aget_state(config)

    self.assertFalse(completed.next)
    self.assertEqual(completed.values["run_id"], "run-2")
    self.assertEqual(completed.values["public_response"]["status"], "completed")
```

- [ ] **Step 2: Run the clarification tests and confirm failure**

```powershell
python -m unittest tests.test_workflow_agent_flow tests.test_collaboration_history
```

Expected: failure because clarification routes to `END` and
`TCMWorkflow.astream(...)` always creates fresh initial state.

- [ ] **Step 3: Implement prepare, interrupt, and resume nodes**

Replace the old clarification node with two explicit nodes:

```python
async def prepare_clarification_node(state: WorkflowState) -> dict[str, Any]:
    inquiry = _inquiry(state)
    questions = list(inquiry.clarification_questions)
    text = format_clarification_questions(questions)
    conversation = append_visible_messages(
        {"conversation": state.get("conversation", [])},
        state["user_text"],
        text,
        run_id=str(state["run_id"]),
        agent_trace=list(state.get("agent_trace", [])),
    )
    return {
        "needs_clarification": True,
        "final_text": text,
        "conversation": conversation,
        "run_outcome": "need_clarification",
        "public_response": {
            "status": "need_clarification",
            "assistant_message": text,
            "pending_clarification": questions,
            "references": [],
        },
    }


def wait_for_clarification_node(state: WorkflowState) -> dict[str, Any]:
    resumed = interrupt(
        {
            "type": "clarification",
            "questions": state["public_response"]["pending_clarification"],
        }
    )
    content = str(resumed["content"]).strip()
    run_id = str(resumed["run_id"])
    return {
        "run_id": run_id,
        "user_text": content,
        "messages": [HumanMessage(content=content, id=f"workflow-{run_id}-human-1")],
        "intent": {},
        "inquiry": {},
        "evidence": {},
        "syndrome": {},
        "answer": {},
        "safety": {},
        "needs_clarification": False,
        "final_text": "",
        "public_response": {},
        "run_outcome": None,
    }
```

Wire `prepare_clarification -> wait_for_clarification -> intent` or `inquiry`
according to the existing routing policy. Update `TCMWorkflow.astream(...)` to
accept `graph_input: dict[str, Any] | Command`; create initial state only for a
normal dict request. Update `WorkflowAgent.astream(...)` to pass a `Command`
through without calling `_latest_user_text(...)`.

- [ ] **Step 4: Run clarification and history tests**

```powershell
python -m unittest tests.test_workflow_agent_flow tests.test_collaboration_history
```

Expected: pause, resume, stale-state, and visible-history tests all pass.

- [ ] **Step 5: Commit durable clarification**

```powershell
git add app/agents/workflow_agent/graph.py app/agents/workflow_agent/workflow.py app/agents/workflow_agent/agent.py tests/test_workflow_agent_flow.py tests/test_collaboration_history.py
git commit -m "feat: resume workflow clarification interrupts"
```

---

### Task 4: Suppress internal structured-model message streams

**Files:**
- Modify: `app/agents/workflow_agent/components/base.py`
- Test: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Write a failing structured-model tag test**

Extend the fake structured runnable so it records `with_config(...)`, then
assert every component installs `nostream`:

```python
def test_structured_components_tag_internal_model_calls_nostream(self):
    model = FakeWorkflowModel([])
    component = IntentAgent(model)

    self.assertEqual(component._structured.config["tags"], ["nostream"])
```

- [ ] **Step 2: Run the focused test and confirm failure**

```powershell
python -m unittest tests.test_workflow_agent_flow.WorkflowLLMBackedTests
```

Expected: failure because structured runnables are currently untagged.

- [ ] **Step 3: Apply the no-stream tag at the component boundary**

```python
class StructuredWorkflowComponent(Generic[SchemaT]):
    def __init__(self, model: Any) -> None:
        self._structured = structured_model(model, self.schema).with_config(
            {"tags": ["nostream"]}
        )
```

Update the fake runnable with a real `with_config(...)` implementation that
returns itself after storing a copied config.

- [ ] **Step 4: Run the workflow component tests**

```powershell
python -m unittest tests.test_workflow_agent_flow
```

Expected: all workflow tests pass and the tag assertion is green.

- [ ] **Step 5: Commit stream visibility filtering**

```powershell
git add app/agents/workflow_agent/components/base.py tests/test_workflow_agent_flow.py
git commit -m "fix: hide structured workflow model streams"
```

---

### Task 5: Forward native LangGraph modes under their own SSE names

**Files:**
- Modify: `app/runtime/runs/stream_adapter.py`
- Test: `tests/test_runtime_stream_adapter.py`

- [ ] **Step 1: Replace compatibility expectations with failing native-mode expectations**

The key test must expect top-level `tasks` and `custom` events:

```python
async def test_multi_mode_stream_forwards_each_mode_under_its_own_name(self):
    agent = FakeStreamingAgent(
        [
            ("messages", (AIMessageChunk(content="a"), {"langgraph_node": "answer"})),
            ("tasks", {"id": "task-1", "name": "answer", "input": {}}),
            ("updates", {"answer": {"public_response": {"status": "completed"}}}),
            ("values", {"public_response": {"status": "completed"}}),
            ("custom", {"progress": 50}),
        ]
    )
    snapshot = await adapter.forward(
        agent,
        {},
        {},
        ["messages", "tasks", "updates", "values", "custom"],
    )

    self.assertEqual(
        published_event_names,
        ["messages", "tasks", "updates", "values", "custom"],
    )
    self.assertEqual(snapshot.latest_values["public_response"]["status"], "completed")
```

Delete observer-specific assertions because manual projected events are no
longer part of the adapter contract.

- [ ] **Step 2: Run adapter tests and confirm failure**

```powershell
python -m unittest tests.test_runtime_stream_adapter.RuntimeStreamAdapterTests
```

Expected: `tasks` and `custom` are still received as `updates` wrappers.

- [ ] **Step 3: Simplify mode forwarding**

Remove `ValuesObserver`, `ProjectedEvents`, and observer invocation. Preserve an
internal `values` subscription for persistence when necessary, but publish only
requested modes:

```python
stream_kwargs = {"config": config, "stream_mode": internal_modes}
if stream_subgraphs:
    stream_kwargs["subgraphs"] = True

async for stream_item in agent.astream(graph_input, **stream_kwargs):
    stream_event, chunk = split_stream_item(stream_item)
    payload = serialize(chunk, mode=stream_event)

    if stream_event == "values" and isinstance(payload, dict):
        latest_values = payload
        raw_messages = payload.get("messages", [])
        latest_messages = raw_messages if isinstance(raw_messages, list) else []

    if stream_event in normalized_modes:
        await self.bridge.publish(self.run_id, stream_event, deepcopy(payload))
```

Add `stream_subgraphs: bool = False` to `forward(...)` and pass it unchanged to
LangGraph. Keep full `AIMessage` forwarding and existing serialization filters.

- [ ] **Step 4: Run adapter and serialization tests**

```powershell
python -m unittest tests.test_runtime_stream_adapter tests.test_runtime_serialization
```

Expected: same-named modes, snapshot capture, cleanup, and serialization tests
all pass.

- [ ] **Step 5: Commit the native adapter**

```powershell
git add app/runtime/runs/stream_adapter.py tests/test_runtime_stream_adapter.py
git commit -m "refactor: forward native LangGraph stream modes"
```

---

### Task 6: Replace post-stream business projection with declarative state persistence

**Files:**
- Create: `app/runtime/runs/state_projection.py`
- Modify: `app/runtime/runs/worker.py`
- Delete: `app/runtime/runs/projection.py`
- Create: `tests/test_runtime_state_projection.py`
- Modify: `tests/test_runtime_worker_lifecycle.py`
- Delete: `tests/test_runtime_completion_projection.py`

- [ ] **Step 1: Write failing projection and lifecycle tests**

Test that only graph-declared fields are persisted and lifecycle uses graph
snapshot metadata:

```python
def test_declared_state_projection_copies_only_public_history_fields():
    result = project_declared_state(
        {"existing": "keep"},
        {
            "messages": [{"type": "ai", "content": "answer"}],
            "conversation": [{"role": "assistant", "content": "answer"}],
            "agent_trace": [{"agent": "AnswerAgent"}],
            "public_response": {"status": "completed", "assistant_message": "answer"},
            "safety": {"private": True},
        },
    )

    self.assertEqual(result["existing"], "keep")
    self.assertIn("public_response", result)
    self.assertNotIn("safety", result)


def test_graph_lifecycle_status_distinguishes_interrupt_and_completion():
    paused = SimpleNamespace(tasks=[SimpleNamespace(interrupts=(object(),))], next=("wait",))
    completed = SimpleNamespace(tasks=(), next=())

    self.assertEqual(graph_lifecycle_status(paused), "waiting_clarification")
    self.assertEqual(graph_lifecycle_status(completed), "success")
```

Update worker tests to assert successful streams contain no `final`,
`clarification`, or `agent_step`.

- [ ] **Step 2: Run the new tests and confirm failure**

```powershell
python -m unittest tests.test_runtime_state_projection tests.test_runtime_worker_lifecycle
```

Expected: import failures for the new module and legacy `final` assertions in
the worker path.

- [ ] **Step 3: Implement the declarative boundary and simplify the worker**

Create `state_projection.py`:

```python
from copy import deepcopy
from typing import Any

DECLARED_THREAD_FIELDS = ("messages", "conversation", "agent_trace", "public_response")


def project_declared_state(
    previous: dict[str, Any],
    latest: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(previous)
    for key in DECLARED_THREAD_FIELDS:
        if key in latest:
            result[key] = deepcopy(latest[key])
    return result


def graph_lifecycle_status(snapshot: Any) -> str:
    tasks = tuple(getattr(snapshot, "tasks", ()) or ())
    if any(tuple(getattr(task, "interrupts", ()) or ()) for task in tasks):
        return "waiting_clarification"
    return "success"
```

In `worker.py`, remove `RunCompletionProjection`, its values observer, all
guardrail publishing, and `final` publication. After adapter completion:

```python
aget_state = getattr(agent, "aget_state", None)
graph_state = await aget_state(runnable_config) if aget_state is not None else None
next_status = graph_lifecycle_status(graph_state)
thread_values = project_declared_state(thread_values, snapshot.latest_values)

await thread_store.update_values(thread_id, thread_values, run_id=run_id)
await run_manager.set_status(run_id, next_status)
await thread_store.update_status(
    thread_id,
    "waiting" if next_status == "waiting_clarification" else "idle",
)
```

Delete `projection.py` and its old completion tests only after the replacement
tests pass in the same working tree.

- [ ] **Step 4: Run runtime, gateway, and workflow integration tests**

```powershell
python -m unittest tests.test_runtime_state_projection tests.test_runtime_worker_lifecycle tests.test_async_agent_factory tests.gateway.test_threads_router tests.test_workflow_agent_flow
```

Expected: all tests pass with no legacy business event publication.

- [ ] **Step 5: Commit lifecycle-only worker state persistence**

```powershell
git add app/runtime/runs/state_projection.py app/runtime/runs/worker.py tests/test_runtime_state_projection.py tests/test_runtime_worker_lifecycle.py tests/test_async_agent_factory.py tests/gateway/test_threads_router.py tests/test_workflow_agent_flow.py
git rm app/runtime/runs/projection.py tests/test_runtime_completion_projection.py
git commit -m "refactor: persist graph-declared run state"
```

---

### Task 7: Keep lead-agent guardrails and clarification inside agent execution

**Files:**
- Create: `app/middlewares/guardrail_agent_middleware.py`
- Modify: `app/middlewares/clarification_middleware.py`
- Modify: `app/agents/lead_agent/agent.py`
- Create: `tests/test_guardrail_agent_middleware.py`
- Modify: `tests/test_clarification_flow.py`
- Modify: `tests/test_lead_agent_factory.py`

- [ ] **Step 1: Write failing middleware tests**

Test final-message rewrite and tool-call interruption:

```python
async def test_guardrail_middleware_replaces_final_ai_message_by_id(self):
    middleware = GuardrailAgentMiddleware()
    state = {"messages": [AIMessage(content="unsafe", id="ai-final")]}

    with patch(
        "app.middlewares.guardrail_agent_middleware.apply_guardrails",
        new=AsyncMock(return_value={"final_text": "safe", "rewritten": True}),
    ):
        update = await middleware.aafter_agent(state, runtime=None)

    self.assertEqual(update["messages"][0].id, "ai-final")
    self.assertEqual(update["messages"][0].content, "safe")


def test_clarification_tool_uses_interrupt_payload(self):
    with patch(
        "app.middlewares.clarification_middleware.interrupt",
        return_value={"content": "two weeks"},
    ) as pause:
        command = ClarificationMiddleware().handle(request)

    pause.assert_called_once_with(
        {"type": "clarification", "questions": ["How long?"]}
    )
    self.assertEqual(command.update["messages"][0].content, "two weeks")
```

- [ ] **Step 2: Run middleware tests and confirm failure**

```powershell
python -m unittest tests.test_guardrail_agent_middleware tests.test_clarification_flow tests.test_lead_agent_factory
```

Expected: missing guardrail middleware and clarification still returns a fixed
tool result without interruption.

- [ ] **Step 3: Implement and install both middlewares**

Create an async after-agent middleware:

```python
class GuardrailAgentMiddleware(AgentMiddleware[AgentState]):
    async def aafter_agent(self, state: AgentState, runtime) -> dict[str, Any] | None:
        messages = list(state.get("messages", []))
        target = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, AIMessage)
                and not message.tool_calls
                and str(message.content).strip()
            ),
            None,
        )
        if target is None or not target.id:
            return None
        serialized = [serialize_message(message) for message in messages]
        result = await apply_guardrails(str(target.content), serialized)
        if result["final_text"] == str(target.content):
            return None
        return {"messages": [AIMessage(id=target.id, content=result["final_text"])]}
```

Change clarification handling so `interrupt(...)` returns the user's response
and the resumed tool call writes a `ToolMessage` containing that response. Add
`GuardrailAgentMiddleware()` after `ClarificationMiddleware()` in the lead-agent
middleware list, preserving middleware ordering tests.

- [ ] **Step 4: Run lead-agent and clarification tests**

```powershell
python -m unittest tests.test_guardrail_agent_middleware tests.test_clarification_flow tests.test_lead_agent_factory
```

Expected: all middleware tests pass; no worker projection is required.

- [ ] **Step 5: Commit lead-agent in-graph semantics**

```powershell
git add app/middlewares/guardrail_agent_middleware.py app/middlewares/clarification_middleware.py app/agents/lead_agent/agent.py tests/test_guardrail_agent_middleware.py tests/test_clarification_flow.py tests/test_lead_agent_factory.py
git commit -m "refactor: move lead agent completion into middleware"
```

---

### Task 8: Expose thread-owned run status and finish Python protocol cleanup

**Files:**
- Modify: `app/gateway/routers/thread_runs.py`
- Modify: `tests/gateway/test_threads_router.py`
- Modify: `tests/test_runtime_worker_boundary.py`
- Modify: `scripts/chat.sh`

- [ ] **Step 1: Write failing run-status ownership and event-contract tests**

```python
async def test_get_run_status_requires_matching_thread(self):
    thread = await state.thread_store.create()
    other = await state.thread_store.create()
    run = await state.run_manager.create(thread.thread_id, "workflow_agent")
    await state.run_manager.set_status(run.run_id, "success")

    response = await client.get(f"/api/threads/{thread.thread_id}/runs/{run.run_id}")
    mismatch = await client.get(f"/api/threads/{other.thread_id}/runs/{run.run_id}")

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()["status"], "success")
    self.assertEqual(mismatch.status_code, 404)
```

Update the SSE integration test to assert the successful event set is limited
to `metadata`, the requested native modes, and `end`.

- [ ] **Step 2: Run gateway tests and confirm failure**

```powershell
python -m unittest tests.gateway.test_threads_router tests.test_runtime_worker_boundary
```

Expected: 404 for the missing route and old boundary assertions referring to
the completion projection.

- [ ] **Step 3: Add the status route and remove shell legacy handling**

```python
@router.get("/{thread_id}/runs/{run_id}")
async def get_run(thread_id: str, run_id: str):
    record = await state.run_manager.get(run_id)
    if record is None or record.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": record.run_id,
        "thread_id": record.thread_id,
        "status": record.status,
        "error": record.error,
    }
```

Update `scripts/chat.sh` to print `values.public_response.assistant_message`,
stop on `end`, and fail on `error`. Remove branches for `final` and
`clarification`.

- [ ] **Step 4: Run the complete Python suite**

```powershell
python -m unittest
```

Expected: all Python tests pass. Then verify no active legacy publications:

```powershell
rg -n 'publish\([^\n]*"(final|clarification|agent_step)"|event == "(final|clarification|agent_step)"' app tests scripts
```

Expected: no active runtime or consumer matches; historical documentation may
still contain those words.

- [ ] **Step 5: Commit the Python gateway cutover**

```powershell
git add app/gateway/routers/thread_runs.py tests/gateway/test_threads_router.py tests/test_runtime_worker_boundary.py scripts/chat.sh
git commit -m "feat: expose native stream run status"
```

---

### Task 9: Update the Spring native stream, resume, and status proxy

**Files:**
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\main\java\com\tcm\consultation\integration\tcmflow\TcmFlowClient.java`
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\main\java\com\tcm\consultation\pojo\dto\ConsultationMessageRequest.java`
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\main\java\com\tcm\consultation\service\ConsultationFlowService.java`
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\main\java\com\tcm\consultation\service\impl\ConsultationFlowServiceImpl.java`
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\main\java\com\tcm\consultation\controller\ConsultationController.java`
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\test\java\com\tcm\consultation\integration\tcmflow\TcmFlowClientHistoryTest.java`
- Modify: `G:\work\tcm-consultation-system\tcm-backend\src\test\java\com\tcm\consultation\service\impl\ConsultationFlowServiceImplTest.java`

- [ ] **Step 1: Snapshot the existing Spring diff and write failing contract tests**

Before editing:

```powershell
git diff -- src/main/java/com/tcm/consultation/integration/tcmflow/TcmFlowClient.java src/main/java/com/tcm/consultation/service/impl/ConsultationFlowServiceImpl.java src/test/java/com/tcm/consultation/integration/tcmflow/TcmFlowClientHistoryTest.java src/test/java/com/tcm/consultation/service/impl/ConsultationFlowServiceImplTest.java
```

Extend `TcmFlowClientHistoryTest`:

```java
@Test
void workflowRequestIncludesAllNativeModes() {
    Map<String, Object> body = TcmFlowClient.buildStreamRequestBody("最近头痛");
    assertEquals(List.of("messages", "tasks", "updates", "values"), body.get("stream_mode"));
}

@Test
void resumeRequestUsesCommandInsteadOfInput() {
    Map<String, Object> body = TcmFlowClient.buildResumeRequestBody("已经两周");
    assertEquals(Map.of("resume", Map.of("content", "已经两周")), body.get("command"));
    assertFalse(body.containsKey("input"));
}
```

Add service tests that reject `resumeRunId` when it differs from
`record.getLastTcmFlowRunId()` and proxy a matching run-status lookup.

- [ ] **Step 2: Run focused Maven tests and confirm failure**

```powershell
mvn "-Dtest=TcmFlowClientHistoryTest,ConsultationFlowServiceImplTest" test
```

Expected: the existing three-mode expectation/implementation mismatch is
visible, and resume/status methods are missing.

- [ ] **Step 3: Implement four modes, resume ownership, and status proxy**

Use one helper for shared stream options:

```java
private static final List<String> NATIVE_STREAM_MODES =
    List.of("messages", "tasks", "updates", "values");

static Map<String, Object> buildResumeRequestBody(String content) {
    return Map.of(
        "assistant_id", "workflow_agent",
        "command", Map.of("resume", Map.of("content", content)),
        "stream_mode", NATIVE_STREAM_MODES,
        "stream_subgraphs", false,
        "config", Map.of("recursion_limit", 50),
        "context", Map.of("subagent_enabled", true)
    );
}

public record RunStatusResponse(
    @JsonProperty("run_id") String runId,
    @JsonProperty("thread_id") String threadId,
    String status,
    String error
) {}
```

Add `resumeRunId` to `ConsultationMessageRequest`. In the service, accept resume
only when it equals the stored last run ID; otherwise raise `BAD_REQUEST`.
Expose:

```java
@GetMapping("/{id}/runs/{runId}")
public ApiResponse<TcmFlowClient.RunStatusResponse> getRunStatus(
    @PathVariable Long id,
    @PathVariable String runId
) {
    return ApiResponse.success(
        consultationFlowService.getRunStatus(id, runId),
        "查询运行状态成功"
    );
}
```

Keep `onEvent(String event, String data)` transparent: event name and data pass
through unchanged.

- [ ] **Step 4: Run focused and full Spring tests**

```powershell
mvn "-Dtest=TcmFlowClientHistoryTest,ConsultationFlowServiceImplTest" test
mvn test
```

Expected: all Spring tests pass. Re-run `git diff` and confirm every pre-existing
history hunk remains.

- [ ] **Step 5: Leave a reviewed Spring diff without auto-committing**

```powershell
git diff --check
git status --short
```

Expected: only intended Spring files are modified. Do not run `git add` or
`git commit` while the pre-existing overlapping changes remain uncommitted.

---

### Task 10: Add React stream API resume and bounded status recovery

**Files:**
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\api\consultation.ts`
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\api\consultation.test.ts`

- [ ] **Step 1: Write failing API tests for native events, resume, and recovery**

Replace `final` fixtures with `values` and `end`, then add:

```typescript
it('sends the interrupted run id when resuming clarification', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse([
    { event: 'end', data: { status: 'done' } },
  ])))

  await streamConsultationRun({
    consultationId: 101,
    message: '已经两周',
    resumeRunId: 'run-1',
    onEvent: vi.fn(),
  })

  expect(fetch).toHaveBeenCalledWith(
    'http://localhost:4040/api/consultations/101/runs/stream',
    expect.objectContaining({
      body: JSON.stringify({ content: '已经两周', resumeRunId: 'run-1' }),
    }),
  )
})

it('recovers a stream missing end from terminal run status', async () => {
  const fetchMock = vi
    .fn()
    .mockResolvedValueOnce(interruptedSseResponse([
      { event: 'metadata', data: { run_id: 'run-1' } },
    ]))
    .mockResolvedValueOnce(jsonResponse({
      code: 0,
      message: 'success',
      data: { run_id: 'run-1', thread_id: 'thread-1', status: 'success', error: null },
    }))
  vi.stubGlobal('fetch', fetchMock)

  await expect(streamConsultationRun({
    consultationId: 101,
    message: '最近头痛',
    onEvent: vi.fn(),
  })).resolves.toBeUndefined()
})
```

- [ ] **Step 2: Run API tests and confirm failure**

```powershell
pnpm --filter tcm-web test -- src/api/consultation.test.ts
```

Expected: missing `resumeRunId`, `end`-only terminal handling, and run-status
recovery.

- [ ] **Step 3: Implement the API contract**

Add the status schema and bounded recovery:

```typescript
const runStatusSchema = z.object({
  run_id: z.string(),
  thread_id: z.string(),
  status: z.enum(['pending', 'running', 'waiting_clarification', 'success', 'error', 'cancelled']),
  error: z.string().nullable(),
})

export type StreamConsultationRunInput = {
  consultationId: number
  message: string
  resumeRunId?: string | null
  onEvent: (event: TcmFlowSseEvent) => void
}

function isTransportTerminalEvent(event: SseEvent) {
  return event.event === 'end' || event.event === 'error'
}
```

Capture `metadata.run_id`. If `readSseStream` throws before `end`, query
`/api/consultations/{id}/runs/{runId}` up to three times with short bounded
delays. Return for `success` or `waiting_clarification`; throw the safe upstream
error for `error` or `cancelled`; rethrow the network error if status remains
`pending` or `running` after the third query.

- [ ] **Step 4: Run API tests**

```powershell
pnpm --filter tcm-web test -- src/api/consultation.test.ts
```

Expected: all stream, resume, and recovery tests pass.

- [ ] **Step 5: Verify the non-Git frontend checkpoint**

```powershell
pnpm --filter tcm-web build
```

Expected: TypeScript and Vite build pass. Record the modified filenames; no Git
commit is available in this directory.

---

### Task 11: Add pure native stream reducers

**Files:**
- Create: `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\nativeStream.ts`
- Create: `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\nativeStream.test.ts`
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\collaboration.ts`
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\features\consultation\collaboration.test.ts`

- [ ] **Step 1: Write failing parser and native collaboration tests**

```typescript
it('reads public response from native values', () => {
  expect(readPublicResponse({
    event: 'values',
    data: {
      public_response: {
        status: 'need_clarification',
        assistant_message: '请补充持续时间。',
        pending_clarification: ['持续多久？'],
        references: [],
      },
    },
  })).toEqual({
    status: 'need_clarification',
    assistantMessage: '请补充持续时间。',
    pendingClarification: ['持续多久？'],
    references: [],
  })
})

it('ignores workflow structured model chunks', () => {
  expect(readMessageDelta({
    event: 'messages',
    data: [
      { type: 'AIMessageChunk', content: '{"primary_intent":' },
      { tags: ['nostream'], langgraph_node: 'intent' },
    ],
  }, 'workflow_agent')).toBe('')
})
```

Update collaboration fixtures so a task is top-level:

```typescript
const running = applyCollaborationSseEvent(createWorkflowSteps(), {
  event: 'tasks',
  data: { id: 'task-1', name: 'evidence', input: {} },
})
```

- [ ] **Step 2: Run reducer tests and confirm failure**

```powershell
pnpm --filter tcm-web test -- src/features/consultation/nativeStream.test.ts src/features/consultation/collaboration.test.ts
```

Expected: the new module is missing and collaboration still expects nested
`updates.stream_event`.

- [ ] **Step 3: Implement pure safe parsers and native collaboration handling**

Create explicit safe output types:

```typescript
export type PublicResponse = {
  status: 'completed' | 'need_clarification'
  assistantMessage: string
  pendingClarification: string[]
  references: unknown[]
}

export function readPublicResponse(event: TcmFlowSseEvent): PublicResponse | null {
  if ((event.event !== 'updates' && event.event !== 'values') || !isRecord(event.data)) return null
  const candidates = event.event === 'values'
    ? [event.data.public_response]
    : Object.values(event.data).map((value) => isRecord(value) ? value.public_response : null)
  const raw = candidates.find(isRecord)
  if (!raw || (raw.status !== 'completed' && raw.status !== 'need_clarification')) return null
  return {
    status: raw.status,
    assistantMessage: typeof raw.assistant_message === 'string' ? raw.assistant_message : '',
    pendingClarification: Array.isArray(raw.pending_clarification)
      ? raw.pending_clarification.filter((item): item is string => typeof item === 'string')
      : [],
    references: Array.isArray(raw.references) ? raw.references : [],
  }
}
```

`readMessageDelta(...)` permits workflow text only when metadata explicitly
marks it public; lead-agent chunks retain current tool and answer parsing.
Refactor `applyCollaborationSseEvent(...)` to branch directly on `tasks` and
`updates`, never `event.data.stream_event`.

- [ ] **Step 4: Run reducer tests**

```powershell
pnpm --filter tcm-web test -- src/features/consultation/nativeStream.test.ts src/features/consultation/collaboration.test.ts
```

Expected: all native parsing and safe-summary tests pass.

- [ ] **Step 5: Build the frontend checkpoint**

```powershell
pnpm --filter tcm-web build
```

Expected: TypeScript and Vite build pass.

---

### Task 12: Integrate native reducers and clarification resume in the workspace

**Files:**
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\features\patient\PatientIntakeWorkspace.tsx`
- Modify: `G:\work\tcm-consultation-system\tcm-web\src\App.test.tsx`

- [ ] **Step 1: Replace UI fixtures with failing native-flow tests**

Change the collaboration test stream to:

```typescript
[
  { event: 'metadata', data: { run_id: 'run-1', assistant_id: 'workflow_agent' } },
  { event: 'tasks', data: { id: 'task-intent', name: 'intent', input: {} } },
  { event: 'updates', data: { intent: { agent_trace: [{ agent: 'IntentAgent' }] } } },
  {
    event: 'values',
    data: {
      public_response: {
        status: 'need_clarification',
        assistant_message: '请补充持续时间。',
        pending_clarification: ['持续多久？'],
        references: [],
      },
    },
  },
  { event: 'end', data: { status: 'done' } },
]
```

Then send the next user message and assert its request contains
`resumeRunId: 'run-1'`.

- [ ] **Step 2: Run UI tests and confirm failure**

```powershell
pnpm --filter tcm-web test -- src/App.test.tsx
```

Expected: the workspace still waits for `final`, unwraps nested tasks, and does
not retain a resume target.

- [ ] **Step 3: Integrate native events and end semantics**

Add per-consultation resume state:

```typescript
const [resumeRunId, setResumeRunId] = useState<string | null>(null)
```

Pass it into `streamConsultationRun(...)`. In the event handler:

```typescript
const publicResponse = readPublicResponse(event)
if (publicResponse) {
  replaceAssistantMessage(assistantMessageId, publicResponse.assistantMessage)
  if (publicResponse.status === 'need_clarification') {
    setResumeRunId(streamContext.runId)
  } else {
    setResumeRunId(null)
  }
}

if (event.event === 'tasks' || event.event === 'updates') {
  updateCollaboration(assistantMessageId, event)
}

if (event.event === 'end') {
  finishMessageCollaboration(assistantMessageId, streamContext.failed ? 'failed' : 'completed')
}
```

Record `run_id` from `metadata`, mark `streamContext.failed` on `error`, remove
the `final`, `clarification`, and `agent_step` branches, and clear resume state
when switching consultations.

- [ ] **Step 4: Run frontend tests and build**

```powershell
pnpm --filter tcm-web test -- src/App.test.tsx src/api/consultation.test.ts src/features/consultation/nativeStream.test.ts src/features/consultation/collaboration.test.ts
pnpm --filter tcm-web build
```

Expected: focused tests and production build pass.

- [ ] **Step 5: Scan the React runtime for removed business events**

```powershell
rg -n "event === 'final'|event === 'clarification'|event === 'agent_step'|stream_event === 'tasks'" tcm-web/src
```

Expected: no active runtime matches. Test descriptions may mention migration
history only if they do not create fixtures using removed events.

---

### Task 13: Run cross-repository verification and browser QA

**Files:**
- Verify only; modify the smallest responsible file if a failing test exposes a
  protocol defect.

- [ ] **Step 1: Run the complete Python verification**

```powershell
python -m unittest
git diff --check
git status --short
```

Expected: all Python tests pass; the Python worktree contains only intentional
changes or is clean after task commits.

- [ ] **Step 2: Run the complete Spring verification**

```powershell
mvn test
git diff --check
git status --short
```

Run from `G:\work\tcm-consultation-system\tcm-backend`.

Expected: all Java tests pass and pre-existing uncommitted history changes are
still present alongside the intentional protocol changes.

- [ ] **Step 3: Run the complete React verification**

```powershell
pnpm --filter tcm-web test
pnpm --filter tcm-web lint
pnpm --filter tcm-web build
```

Run from `G:\work\tcm-consultation-system`.

Expected: tests, lint, and production build pass.

- [ ] **Step 4: Exercise completed and clarification/resume flows against running services**

Send a normal workflow request and a clarification-triggering request. Capture
the SSE event names and verify each trace follows this grammar:

```text
metadata
(messages | tasks | updates | values)*
end
```

For clarification, verify the first run status is `waiting_clarification`, the
resume request continues the same thread checkpoint, and the resumed run ends
with status `success`. Verify no trace contains `final`, `clarification`, or
`agent_step`.

- [ ] **Step 5: Run browser QA with the Browser plugin**

Open the consultation UI and verify:

1. Agent rows move from pending to running to completed from native events.
2. The assistant placeholder is replaced from `values.public_response`.
3. Clarification ends loading, retains the question, and resumes on the next
   user message.
4. A simulated stream close without `end` triggers run-status recovery.
5. `error` followed by `end` remains visibly failed.
6. Refresh restores assistant content and collaboration history.

Capture screenshots only for defects or final QA evidence; do not redesign the
UI.

- [ ] **Step 6: Final protocol scan and handoff**

```powershell
rg -n 'publish\([^\n]*"(final|clarification|agent_step)"|event === ["'"'](final|clarification|agent_step)["'"']|stream_event === ["'"']tasks["'"']' G:\work\tcm-flow\app G:\work\tcm-consultation-system\tcm-backend\src G:\work\tcm-consultation-system\tcm-web\src
```

Expected: no active producer or consumer dependency on the removed protocol.
Report Python commit IDs, the uncommitted Spring diff status, React verification
results, and browser QA evidence.
