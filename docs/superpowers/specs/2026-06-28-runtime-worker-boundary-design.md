# Runtime Worker Boundary Refactor Design

## Summary

Refactor `app/runtime/runs/worker.py` into a thin run-runtime coordinator while preserving the current HTTP API, SSE event names, final response payload, clarification behavior, guardrail behavior, checkpoint history, and visible conversation projection.

The target boundary follows the current ByteDance DeerFlow worker at the responsibility level: the worker owns run lifecycle, runtime/config construction, agent creation, LangGraph streaming, status transitions, errors, cancellation, terminal publication, and cleanup. TCM-specific message interpretation and response shaping live behind explicit helpers outside the worker.

This is a bounded runtime refactor. It does not redesign the workflow graph, move guardrails into graph nodes, change the public request schema, or replace the existing `StreamBridge`, stores, agent registry, or agent factories.

## Current Problem

`app/runtime/runs/worker.py` currently combines four distinct responsibilities:

1. Run runtime orchestration:
   - mark run and thread as running;
   - construct the agent and LangGraph config;
   - invoke `agent.astream(...)`;
   - publish metadata, errors, and `end`.
2. Stream protocol adaptation:
   - normalize requested stream modes;
   - force internal `messages` and `values` modes;
   - unpack single-mode and multi-mode LangGraph output;
   - serialize and forward chunks.
3. TCM completion semantics:
   - slice checkpoint messages to the current run;
   - identify clarification outcomes;
   - run guardrails;
   - rewrite the final checkpoint message.
4. Public projection:
   - extract assistant text and pending questions;
   - extract debug trace events from tool messages;
   - construct the legacy final response;
   - append visible conversation and agent trace data.

This coupling makes lifecycle changes risky because stream protocol, checkpoint mutation, business decisions, and public response compatibility all have to be understood together.

## Goals

- Make `worker.py` read as a run lifecycle, not a business controller.
- Introduce a `RunContext` boundary and pass graph input, runnable config, and stream modes explicitly.
- Move LangGraph stream normalization, unpacking, serialization, forwarding, and latest-state collection into a stream adapter.
- Move clarification, guardrail, checkpoint rewrite, trace projection, conversation projection, and final payload construction into a completion projection helper.
- Preserve current-run message scoping so historical clarification messages cannot be reused accidentally.
- Preserve the existing external API and SSE protocol as characterized by tests.
- Add explicit cancellation handling and terminal bridge cleanup without introducing a new public cancellation API.
- Preserve the staged and unstaged collaboration-history changes already present in the worktree.

## Non-Goals

- Moving guardrails or clarification policy into new LangGraph nodes.
- Changing `lead_agent` or `workflow_agent` orchestration.
- Changing the `RunCreateRequest` wire shape.
- Adding synthetic token chunking.
- Replacing `thread_store` with the LangGraph checkpointer or reconstructing workflow state from `thread_store`.
- Introducing a separate abort endpoint or rollback semantics.
- Removing legacy `final` response fields or debug-event compatibility.

## Boundary Principles

### Execution state and business projection remain separate

The LangGraph checkpointer remains the source of truth for workflow execution messages and state. `thread_store` remains the product-facing projection for visible conversation, run-associated trace data, and thread status.

The refactor must not feed `thread_store.values["messages"]` back into the graph as execution history. Each run sends only its graph input; the graph checkpointer supplies prior execution context.

### Completion decisions use current-run messages only

Before streaming starts, the projection captures the existing checkpoint message count. After each `values` snapshot, the current-run view is:

```python
current_run_messages = latest_messages[message_start_index:]
```

Clarification detection, trace debug projection, assistant response extraction, and guardrail completion use this view where appropriate. Full messages remain available for checkpoint persistence and guardrail context.

### The worker knows outcomes, not message semantics

The worker may branch on an explicit completion outcome such as `completed` or `need_clarification`. It must not inspect roles, contents, tool calls, tool-result strings, or guardrail fields itself.

### The stream adapter knows protocol shapes, not TCM semantics

The stream adapter understands LangGraph stream modes and serialized snapshots. It does not identify clarification, parse tools, execute guardrails, or build final responses.

## Proposed Module Structure

### `app/runtime/runs/context.py`

Defines the infrastructure and runtime-context boundary:

```python
@dataclass(frozen=True)
class RunContext:
    thread_store: ThreadStore
    agent_context: Mapping[str, Any] = field(default_factory=dict)
```

It also provides small helpers to:

- build a run-scoped runtime context containing `thread_id`, `run_id`, and caller-provided agent options;
- merge the required `thread_id` into `config["configurable"]`;
- preserve caller-provided config fields;
- set the default recursion limit only when the caller did not provide one;
- install runtime context into the runnable config.

`RunContext` groups run infrastructure and agent options. It does not own business completion state.

### `app/runtime/runs/input.py`

Owns request-to-graph normalization currently embedded in `worker.py`:

- `extract_text_from_content(...)`;
- `normalize_graph_input(...)`;
- `extract_user_text(...)`.

`app/runtime/services.py` invokes these helpers before starting the background worker. The worker therefore receives `graph_input` rather than an HTTP-shaped `input_data` payload.

The existing behavior remains unchanged: empty messages are dropped; human, AI, and system messages are mapped to supported roles; tool messages are not manually replayed into the graph.

### `app/runtime/runs/stream_adapter.py`

Defines:

```python
@dataclass(frozen=True)
class StreamSnapshot:
    latest_values: dict[str, Any]
    latest_messages: list[dict[str, Any]]


class LangGraphStreamAdapter:
    async def forward(...) -> StreamSnapshot: ...
```

The adapter owns:

- normalizing requested stream modes;
- ensuring internal `messages` and `values` subscriptions required by the compatibility projection;
- invoking `agent.astream(...)`;
- unpacking raw, multi-mode, and non-values stream items;
- mode-specific serialization;
- publishing `messages`, requested `values`, and compatible `updates` events;
- retaining the latest serialized values and message list;
- invoking an optional values observer for debug projections without interpreting the observer's semantics.

Non-`messages`/`values` LangGraph modes retain the current compatibility behavior: the client sees them through `updates`, with the existing wrapping rule for modes other than `updates`.

The adapter forwards model chunks exactly as LangGraph emits them and never invents token chunks.

### `app/runtime/runs/projection.py`

Defines a run-scoped projection object and an immutable completion result:

```python
@dataclass(frozen=True)
class CompletionResult:
    run_status: str
    thread_status: str
    thread_values: dict[str, Any]
    final_payload: dict[str, Any]
```

The projection owns:

- capturing the pre-run checkpoint message count;
- exposing a values observer used for opt-in trace debug `updates` events;
- extracting `agent_trace` from the latest values snapshot;
- computing the current-run message slice;
- detecting clarification through the existing clarification controller;
- extracting visible clarification content and pending questions;
- extracting the final assistant answer;
- executing guardrails for non-clarification completion;
- publishing existing guardrail debug updates through an injected callback;
- replacing a rewritten final `AIMessage` in the checkpoint and re-reading the checkpoint;
- appending user/assistant visible conversation with `run_id` and current `agent_trace`;
- constructing the existing `build_chat_response(...)` payload;
- returning the status and persistence instructions to the worker.

Business extraction functions remain in their existing middleware/public-message modules where practical. Checkpoint rewrite details become private to this projection module.

### `app/runtime/public_messages.py`

Remains the canonical public response shaping layer. The worker-local visible-conversation and final-assistant helpers move here when they are public-message concerns. Existing functions and payload fields remain compatible.

### `app/runtime/runs/worker.py`

The resulting worker has one orchestration path:

1. Set run/thread status to running.
2. Publish metadata.
3. Read the existing thread projection.
4. Build runtime context and runnable config.
5. Create the agent.
6. Initialize completion projection and capture the message boundary.
7. Call the stream adapter.
8. Ask the completion projection for a `CompletionResult`.
9. Persist returned thread values.
10. Publish the returned final payload.
11. Apply returned run/thread statuses.
12. Handle cancellation or errors.
13. Always publish `end` and schedule bridge cleanup.

The intended internal signature is:

```python
async def run_agent(
    bridge: StreamBridge,
    run_manager: RunManager,
    record: RunRecord,
    *,
    ctx: RunContext,
    agent_factory: AgentFactory,
    graph_input: dict[str, Any],
    config: dict[str, Any],
    stream_modes: list[str] | None = None,
) -> None:
    ...
```

This is an internal API change. The FastAPI request and SSE response contracts do not change.

### `app/runtime/services.py`

The service layer continues to validate the thread, create the run and bridge queue, and resolve the agent factory. It additionally:

- normalizes `body.input` into graph input;
- preserves `body.config` while enforcing the route's `thread_id`;
- constructs `RunContext(thread_store=..., agent_context=body.context)`;
- passes top-level `body.stream_mode` separately to the worker.

This removes HTTP input normalization and stream-mode extraction from the worker.

### `app/runtime/stream.py`

Adds delayed queue cleanup after terminal publication. Cleanup must not remove the queue before a just-created streaming response has subscribed, so it follows DeerFlow's delayed-cleanup pattern rather than deleting immediately after `publish_end(...)`.

## Runtime Data Flow

```text
FastAPI service
  -> normalize graph input
  -> build RunContext + request config + stream modes
  -> worker marks running and publishes metadata
  -> worker builds runtime context/config and creates agent
  -> stream adapter calls agent.astream
       -> forwards messages/values/updates
       -> projection observer emits optional debug updates
       -> retains latest values/messages
  -> completion projection
       -> current-run slice
       -> clarification OR guardrail completion
       -> optional checkpoint rewrite
       -> final payload + thread projection + statuses
  -> worker persists/publishes/statuses
  -> worker publishes end and schedules cleanup
```

## External Compatibility Contract

| Surface | Required behavior after refactor |
|---|---|
| Request body | Existing `assistant_id`, `input`, `stream_mode`, `config`, `context`, and metadata fields remain accepted. |
| `metadata` | Contains current run/thread/assistant/architecture metadata as represented by the current worktree. |
| `messages` | Forwards serialized LangGraph message output without synthetic chunking. |
| `values` | Published only when requested or when debug events require it, matching current behavior. |
| Extra stream modes | Continue to use the current `updates` compatibility envelope. |
| Clarification | No separate `clarification` SSE event; terminal business payload is `final` with `status="need_clarification"`. |
| Completed final | `status="completed"` with `thread_id`, `run_id`, `assistant_message`, `pending_clarification`, and `references`. |
| Error | Publishes `error` with `message`, marks run/thread error, then publishes `end`. |
| Cancellation/abort | A cancelled background task marks the run `cancelled`, returns the thread to `idle`, publishes `end`, and re-raises cancellation. No new public endpoint is added. |
| Conversation | Stores only visible user/assistant turns; assistant turn retains `run_id` and current `agent_trace`. |
| Checkpoint | Rewritten guardrail text replaces the final checkpoint AI message by ID, preserving follow-up context safety. |

## Error, Cancellation, and Cleanup Semantics

### Errors

Unexpected exceptions are logged with traceback, persisted as run status `error`, reflected in thread status `error`, and published as the existing `error` event. The `finally` block always publishes `end`.

### Cancellation and abort

The current runtime has no separate abort flag or abort endpoint. Its in-scope abort mechanism is cancellation of `RunRecord.task`. `worker.py` handles `asyncio.CancelledError` separately from ordinary exceptions, marks the run `cancelled`, restores the thread status to `idle`, publishes no misleading success/final payload, and re-raises the cancellation after lifecycle state is consistent.

### Cleanup

After `end`, the worker schedules delayed removal of the bridge queue. The delay preserves late subscription behavior while preventing completed run queues from accumulating indefinitely.

## Test Strategy

Implementation follows test-first development.

### Characterization and architecture tests

- Add a failing worker-boundary test that rejects direct imports of guardrail, clarification, trace parsing, `AIMessage`, and public response construction from `worker.py`.
- Preserve the current contract tests for final/clarification payloads, stream modes, values publication, debug updates, agent trace persistence, and follow-up clarification isolation.

### Stream adapter tests

- requested modes normalize deterministically;
- internal `messages` and `values` modes are present exactly once;
- single raw values chunks and multi-mode tuples are unpacked correctly;
- message chunks are forwarded unchanged after serialization;
- requested values and extra modes retain current event names/envelopes;
- returned `StreamSnapshot` contains the latest serialized values/messages.

### Projection tests

- historical messages are excluded from clarification decisions;
- clarification completion skips guardrails and returns waiting statuses;
- normal completion runs guardrails and returns success/idle statuses;
- rewritten guardrail output replaces the checkpoint message and stored messages;
- visible conversation includes the current `run_id` and current `agent_trace`;
- debug trace extraction remains opt-in and is emitted through `updates`;
- final payload contains no internal messages, validation objects, or trace fields.

### Worker lifecycle tests

- metadata precedes streamed output;
- worker delegates streaming and completion rather than parsing messages;
- success, clarification, error, and cancellation status transitions are correct;
- every path publishes `end`;
- cleanup is scheduled only after terminal publication.

### Verification

The current baseline is 45 passing focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_async_agent_factory tests.test_clarification_flow tests.test_subagent_clarification tests.gateway.test_threads_router tests.test_workflow_agent_flow tests.test_collaboration_history
```

After refactoring, run the new unit tests, the same compatibility suite, targeted service tests, compilation checks, and the broadest repository suite supported by the installed dependencies. Report any unrelated dependency-limited discovery failures separately rather than treating them as evidence about this refactor.

## Migration Sequence

1. Add failing boundary and adapter tests.
2. Extract input and context construction without changing events.
3. Introduce the stream adapter and move protocol handling behind it.
4. Add projection tests and move completion semantics behind the projection.
5. Reduce the worker to lifecycle orchestration and update `services.py` to the explicit internal signature.
6. Add cancellation and delayed cleanup coverage.
7. Run compatibility and broad verification.

Each step must keep the external API and event protocol green before proceeding.

## Acceptance Criteria

- `worker.py` contains no direct guardrail execution, clarification extraction, role/content parsing, trace/tool parsing, final-response assembly, or checkpoint-message replacement implementation.
- `worker.py` directly coordinates lifecycle, context/config, agent creation, adapter execution, projection result persistence, statuses, error/cancellation, `end`, and cleanup.
- The worker uses explicit `RunContext`, graph input, config, and stream-mode inputs.
- Existing success and clarification final payloads remain byte-shape compatible at the JSON field level.
- Existing message, values, updates, error, and end event semantics remain compatible.
- Follow-up runs do not reuse clarification messages from earlier runs.
- Guardrail rewrites still replace unsafe final checkpoint content.
- Visible conversation still includes the current run's agent trace metadata.
- Focused compatibility tests and new boundary tests pass.
