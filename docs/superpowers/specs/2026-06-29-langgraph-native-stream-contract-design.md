# LangGraph-Native Stream Contract Design

## Summary

Replace the repository-specific streamed business events with a DeerFlow-style
runtime contract in which LangGraph stream modes are the public source of
execution changes.

The client requests `messages`, `tasks`, `updates`, and `values`. The Python
gateway forwards each LangGraph mode as the same-named SSE event without
wrapping one mode inside another. The graph owns clarification, guardrails, and
the public response state. The Spring service transparently proxies the stream.
The React client reduces the four native modes and uses `end` or the run-status
endpoint to settle the UI.

The only manually published SSE events that remain are transport and run
lifecycle events:

- `metadata`
- `error`
- `end`

The legacy business events `final`, `clarification`, and `agent_step` are
removed without a dual-protocol compatibility window.

## Scope

This is a coordinated change across:

- `G:\work\tcm-flow`
- `G:\work\tcm-consultation-system\tcm-backend`
- `G:\work\tcm-consultation-system\tcm-web`

The design applies to the shared `/api/threads/{thread_id}/runs/stream` runtime
contract. The complete `public_response` state contract is required for
`workflow_agent`. Other registered agents continue to expose model output via
`messages`, but must no longer depend on runtime-generated `final` or
`clarification` events.

## Goals

- Forward LangGraph `messages`, `tasks`, `updates`, and `values` modes as
  same-named SSE events.
- Make graph state and graph lifecycle the source of business changes.
- Move post-stream guardrail and clarification semantics into the relevant
  graph or agent middleware.
- Use LangGraph `interrupt` and `Command(resume=...)` for clarification.
- Give `workflow_agent` a stable `public_response` state field.
- Let the frontend settle runs from `end` plus observed outcome, with run-status
  recovery when `end` is missing.
- Preserve visible conversation and collaboration history across refreshes.
- Remove all runtime and frontend dependencies on `final`, `clarification`, and
  `agent_step`.

## Non-Goals

- Introducing another product-specific streaming envelope.
- Having Spring aggregate native events into replacement business events.
- Sending synthetic model-token chunks.
- Migrating the runtime to LangGraph stream format v2 as part of this change.
- Displaying or persisting raw private graph state, task input, structured-model
  JSON, or chain-of-thought in the frontend.
- Redesigning the six-Agent collaboration card or the consultation layout.
- Replacing the LangGraph checkpointer with `thread_store`.

## Current Problems

### Mode identity is lost

`app/runtime/runs/stream_adapter.py` currently forwards `messages` and `values`
directly but maps non-`values` modes into an `updates` SSE event. A `tasks`
event therefore arrives as an `updates` event containing:

```json
{
  "stream_event": "tasks",
  "data": {}
}
```

This makes the frontend understand a repository-specific wrapper instead of
the LangGraph stream protocol.

### Business completion happens after graph execution

`RunCompletionProjection` currently inspects streamed messages after
`agent.astream(...)`, detects clarification, applies guardrails, rewrites the
checkpoint, builds a public payload, and asks the worker to publish `final`.
Those business changes are invisible to LangGraph `tasks`, `updates`, and
`values` because they happen outside the graph.

### The frontend depends on legacy terminal events

`tcm-web` currently:

- finishes collaboration on `final` or `clarification`;
- replaces assistant text from `final.assistant_message`;
- treats `final`, `clarification`, `end`, and `error` as terminal stream events;
- unwraps `tasks` from an `updates.stream_event` compatibility payload.

### Structured LLM output is unsafe to render as answer text

`workflow_agent` uses structured model calls for intent, inquiry, syndrome,
answer drafting, and safety review. Raw `messages` chunks from those calls can
contain JSON or internal classification output. The React client must not append
all non-tool `AIMessageChunk` content indiscriminately.

## Selected Architecture

### 1. LangGraph owns business outcomes

`workflow_agent` state gains the following public fields:

```python
class PublicResponse(TypedDict):
    status: Literal["completed", "need_clarification"]
    assistant_message: str
    pending_clarification: list[str]
    references: list[dict[str, Any]]


class WorkflowState(TypedDict, total=False):
    # Existing internal fields remain.
    public_response: PublicResponse
    run_outcome: Literal["completed", "need_clarification"]
```

`public_response` is business output stored in graph state, not an SSE event.
It becomes visible naturally through `updates` and `values`.

The graph also owns the visible `conversation` update for the current turn.
The runtime may persist declared state fields, but it must not infer the answer
or clarification by parsing tool calls or message text.

### 2. Guardrails execute inside the agent path

The existing post-stream `apply_guardrails(...)` behavior moves into the agent
execution path:

- `workflow_agent` runs it in a terminal graph node before writing
  `public_response` and the final visible `AIMessage`;
- `lead_agent` receives equivalent protection through agent middleware or a
  graph terminal step;
- checkpoint replacement after the stream is removed;
- any guardrail rewrite appears in native `tasks`, `updates`, and `values`.

The terminal node is the only writer of a completed `public_response`.

### 3. Clarification uses durable interruption

The clarification route is split into two graph nodes:

1. `prepare_clarification`
   - writes the visible clarification message;
   - writes `public_response.status = "need_clarification"`;
   - writes the normalized `pending_clarification` list;
   - appends the visible conversation turn.
2. `wait_for_clarification`
   - calls LangGraph `interrupt(...)` with a small public payload;
   - does not repeat the full internal state in the interrupt value.

The pre-interrupt state therefore appears in `updates` and `values` before the
run stream ends.

When the user replies, the new HTTP run uses `Command(resume=...)` against the
same `thread_id`. The graph consumes the resume value, resets transient fields
for the resumed turn, and routes back through inquiry or the next appropriate
node. Historical clarification state must not be mistaken for a new pause.

### 4. The worker remains lifecycle-only

`app/runtime/runs/worker.py` owns:

- run/thread status transitions;
- runtime context and runnable config;
- agent creation;
- `agent.astream(...)` invocation;
- native mode forwarding through the adapter;
- cancellation and errors;
- graph lifecycle inspection after streaming;
- generic declared-state persistence;
- `end` publication and bridge cleanup.

The worker does not:

- parse message roles, tool calls, or content;
- detect `ask_clarification`;
- call business guardrails;
- construct `assistant_message`;
- publish `final`, `clarification`, or `agent_step`.

After streaming, the worker reads the LangGraph snapshot only for lifecycle
facts. A pending interrupt produces `waiting_clarification`; an exhausted graph
produces `success`. This decision must use graph lifecycle metadata such as
pending tasks/interrupts, not business-message parsing.

### 5. State projection remains declarative

The product-facing thread store remains the visible history boundary. A small
projection component may copy graph-declared fields from the latest state:

- `messages`
- `conversation`
- `agent_trace`
- `public_response`

It must not derive those fields from raw tool chatter. The LangGraph
checkpointer remains the source of workflow execution state and resumability.

## Native SSE Contract

The Spring request uses exactly:

```json
{
  "assistant_id": "workflow_agent",
  "input": {
    "messages": [
      {
        "type": "human",
        "content": "最近头痛"
      }
    ]
  },
  "stream_mode": ["messages", "tasks", "updates", "values"],
  "stream_subgraphs": false,
  "config": {"recursion_limit": 50},
  "context": {"subagent_enabled": true}
}
```

`tcm-flow` validates requested modes against the supported LangGraph modes. For
each yielded `(mode, payload)` item, it publishes:

```text
event: <mode>
data: <mode-specific serialized payload>
```

No native mode is wrapped in another native mode.

| SSE event | Source | Consumer purpose |
| --- | --- | --- |
| `metadata` | Runtime | Capture `run_id`, `thread_id`, and `assistant_id` |
| `messages` | LangGraph | Public model-token or message stream when allowed |
| `tasks` | LangGraph | Node/task start, completion, and error lifecycle |
| `updates` | LangGraph | Per-node state deltas, including `agent_trace` and `public_response` |
| `values` | LangGraph | Authoritative full-state reconciliation |
| `error` | Runtime | Transport or run failure |
| `end` | Runtime | Stream closure only; not proof of success |

The implementation continues to forward full `AIMessage` values when that is
what LangGraph emits. It never manufactures token chunks.

## Resume Contract

`RunCreateRequest` accepts exactly one of `input` or `command`.

The resume request is:

```json
{
  "assistant_id": "workflow_agent",
  "command": {
    "resume": {
      "content": "已经持续两周，饭后会缓解"
    }
  },
  "stream_mode": ["messages", "tasks", "updates", "values"],
  "stream_subgraphs": false,
  "config": {"recursion_limit": 50},
  "context": {"subagent_enabled": true}
}
```

The service converts this payload to `Command(resume=...)` before calling the
graph. Supplying both or neither `input` and `command` is a 422 validation error.

The React client retains the interrupted `run_id` after observing
`public_response.status = "need_clarification"`. Its next consultation message
includes that run ID as an optional resume target. Spring verifies that it
matches the consultation's `lastTcmFlowRunId` before sending the resume request.
A stale or mismatched resume target is rejected rather than silently starting a
different semantic flow.

## Run Status Contract

`tcm-flow` adds a read endpoint:

```text
GET /api/threads/{thread_id}/runs/{run_id}
```

The response contains at least:

```json
{
  "run_id": "...",
  "thread_id": "...",
  "status": "pending | running | waiting_clarification | success | error | cancelled",
  "error": null
}
```

The endpoint rejects a `run_id` that does not belong to `thread_id`.

Spring exposes the corresponding consultation-scoped endpoint and verifies the
consultation owns the `thread_id` before proxying the query. The browser never
calls the Python service directly.

## End-to-End Data Flow

### New turn

1. React creates a placeholder assistant turn and opens the Spring SSE request.
2. Spring sends the four requested stream modes to `tcm-flow`.
3. `tcm-flow` publishes `metadata` and invokes the graph.
4. LangGraph emits `tasks`, `updates`, `values`, and any allowed `messages`.
5. Spring forwards event name and data without business interpretation.
6. React reduces each native mode.
7. The graph writes `public_response`, visible conversation, and final messages.
8. The worker persists declared projection fields and sets the run status.
9. The worker publishes `end`.
10. React settles the placeholder from the latest `public_response` and observed
    terminal state.

### Clarification

1. `prepare_clarification` emits state containing `public_response`.
2. `wait_for_clarification` interrupts the graph.
3. The worker records `waiting_clarification` and publishes `end`.
4. React displays the clarification and retains the interrupted run ID.
5. The next user reply is sent as a resume request.
6. LangGraph resumes from the checkpoint and continues the same workflow state.

### Missing `end`

1. The browser stream reader fails or closes without `end`.
2. React queries the consultation-scoped run-status endpoint using the captured
   run ID.
3. `success`, `waiting_clarification`, `error`, or `cancelled` settles the UI.
4. `pending` or `running` is retried with a bounded policy.
5. Exhausting the bounded policy produces a recoverable connection error. It
   must not fabricate success.

## Frontend Reduction Rules

### `messages`

- `lead_agent` may continue to display public assistant token chunks.
- `workflow_agent` structured calls are marked with `nostream` or equivalent
  metadata and are never appended to the visible answer.
- Tool calls and tool results may update the existing lead-agent process UI but
  are not shown as assistant text.
- Complete `AIMessage` values are not appended after token chunks; `values`
  performs final reconciliation.

### `tasks`

- A task start marks the mapped Agent as running.
- A task result marks it completed only after the corresponding safe state
  update is available.
- A task error marks it failed.
- Raw task input, output, and errors are not rendered or persisted.

### `updates`

- Node-keyed `agent_trace` updates drive safe collaboration summaries.
- `public_response` immediately updates the assistant placeholder.
- Unknown node keys and fields are ignored.

### `values`

- `public_response` is authoritative for assistant text, clarification status,
  pending questions, and references.
- `agent_trace` reconciles collaboration state.
- Other full-state fields are not logged, rendered, or persisted by the browser.

### `error` and `end`

- `error` marks the run failed and stores only the safe error message.
- `end` stops loading.
- `error` followed by `end` remains failed.
- `end` without `public_response` is valid for agents whose visible answer came
  entirely from `messages`; otherwise the frontend verifies run status/history.

## History

The history route continues to expose enriched `conversation` and raw
`messages`, with consumers preferring `conversation` when present.

Each visible assistant conversation turn may contain:

- `content`
- `run_id`
- safe `agent_trace`
- public clarification metadata when applicable

Before conversation history is supplied to a model, runtime-only fields such as
`run_id`, `agent_trace`, and clarification UI metadata are removed. Tool chatter
and internal structured-model output remain excluded from visible history.

## Failure and Cancellation Semantics

- `end` means the bridge is closed; it never means success by itself.
- `error` is published before `end` for an unhandled run failure.
- A cancelled run sets status `cancelled`, emits no business event, and ends the
  stream.
- A paused graph sets `waiting_clarification`, not `success`.
- A normal exhausted graph sets `success` only after declared projection fields
  have been persisted.
- Spring does not replace an upstream terminal outcome with a generic network
  error after it has received `end`.
- A stream that dies before `end` remains unresolved until run-status recovery
  succeeds or the bounded recovery policy fails.

## Module Changes

### `tcm-flow`

- `app/schemas.py`
  - add mutually exclusive `input` and `command` request forms.
- `app/runtime/services.py`
  - normalize new input or resume command;
  - forward `stream_subgraphs` instead of dropping it.
- `app/runtime/runs/input.py`
  - build normal graph input or `Command(resume=...)`.
- `app/runtime/runs/stream_adapter.py`
  - forward same-named modes;
  - retain the latest values only for declarative persistence;
  - remove projected event callbacks.
- `app/runtime/runs/projection.py`
  - replace completion inference with a declarative state persistence boundary,
    or remove the module if that boundary fits elsewhere.
- `app/runtime/runs/worker.py`
  - remove business completion and legacy event publication;
  - derive paused/completed lifecycle from graph state.
- `app/gateway/routers/thread_runs.py`
  - add run-status lookup.
- `app/agents/workflow_agent/state.py`
  - add `public_response` and `run_outcome`.
- `app/agents/workflow_agent/graph.py`
  - add graph-owned guardrail completion;
  - split clarification preparation and interruption;
  - consume resume values.
- `app/agents/workflow_agent/components/base.py`
  - suppress public streaming for internal structured calls.
- lead-agent construction/middleware
  - preserve equivalent guardrail and interrupt behavior without runtime
    business projection.

### Spring backend

- `TcmFlowClient`
  - request all four modes;
  - build normal and resume request bodies;
  - proxy run status.
- `ConsultationFlowServiceImpl`
  - retain the last run ID;
  - transparently forward native SSE events;
  - validate resume ownership.
- consultation controller/DTO
  - accept an optional resume target;
  - expose consultation-scoped run status.

Existing uncommitted collaboration-history changes in the Spring repository are
preserved. Their current test/implementation mismatch is resolved by making both
expect all four modes.

### React frontend

- `src/api/consultation.ts`
  - make only `end` and `error` transport-terminal;
  - add resume and run-status calls;
  - recover a missing `end` from run status.
- `src/features/consultation/collaboration.ts`
  - consume top-level `tasks` and native `updates`;
  - remove the nested `stream_event` wrapper.
- `PatientIntakeWorkspace.tsx`
  - reduce `messages`, `tasks`, `updates`, and `values`;
  - remove legacy terminal event branches;
  - retain and use the interrupted run ID.
- history restoration
  - continue rebuilding safe collaboration state from persisted `agent_trace`.

## Test Strategy

### Python

- Requested native modes are passed to `agent.astream(...)`.
- `messages`, `tasks`, `updates`, and `values` are emitted under the same SSE
  names and retain mode-specific serialization.
- No successful or clarification path publishes `final`, `clarification`, or
  `agent_step`.
- A completed workflow writes a valid `public_response` before graph completion.
- A clarification writes public state, interrupts, and reports
  `waiting_clarification`.
- `Command(resume=...)` continues from the checkpoint without reusing stale
  clarification state.
- Guardrail rewrites appear in graph state and checkpoint messages.
- Run-status lookup enforces thread ownership.
- Error, cancellation, and cleanup behavior remains correct.
- Full `AIMessage` output is forwarded without synthetic chunking.

### Spring

- Normal and resume request bodies contain all four stream modes.
- Native event names and data pass through unchanged.
- Run ID capture and consultation ownership are enforced.
- Status lookup maps upstream outcomes without inventing business events.
- Enriched conversation remains preferred over raw message history.
- Existing dirty working-tree changes are preserved and their tests remain
  green.

### React

- Each native mode has reducer coverage.
- Top-level `tasks` start/fail lifecycle updates the collaboration UI.
- Native node `updates` finish safe summaries.
- `values.public_response` reconciles final and clarification text.
- Internal structured-output `messages` are ignored.
- `error` followed by `end` remains failed.
- Missing `end` triggers bounded run-status recovery.
- Clarification retains a resume target and the next message resumes the graph.
- Refresh/reopen restores visible messages and collaboration history.

### Cross-repository verification

- Completed consultation response.
- Clarification followed by resume.
- Guardrail rewrite.
- Model/provider failure.
- Spring/upstream stream truncation.
- Browser refresh after completion and after clarification.
- Browser QA for assistant text, collaboration steps, loading termination, and
  error display.

## Acceptance Criteria

- The Spring request asks for exactly `messages`, `tasks`, `updates`, and
  `values`.
- The browser receives those modes as top-level same-named SSE events.
- Runtime success and clarification paths publish no `final`, `clarification`,
  or `agent_step` events.
- `workflow_agent` final and clarification output is available in
  `updates/values.public_response`.
- Clarification pauses through LangGraph `interrupt` and continues through
  `Command(resume=...)`.
- Guardrails run inside the graph or agent path, not after streaming.
- `end` and run status settle every success, clarification, failure,
  cancellation, and truncated-stream path.
- The frontend never renders internal structured JSON or raw private task/state
  payloads.
- Visible answer and collaboration history survive refresh.
- Focused Python, Java, and React suites pass, followed by cross-repository
  browser verification.
- Runtime code and frontend consumers contain no active dependency on the
  removed legacy event names.

## Implementation Sequencing

The implementation must be planned and delivered in dependency order:

1. Add graph-owned public output, guardrails, interrupt, and resume support.
2. Add native mode forwarding and run-status APIs in `tcm-flow`.
3. Update Spring request, resume, transparent proxy, and status endpoint.
4. Update React reducers, terminal handling, and resume behavior.
5. Remove legacy event code and tests only after the new end-to-end path passes.
6. Run focused suites, cross-repository integration tests, and browser QA.

The protocol cutover is direct, but implementation sequencing prevents an
intermediate commit from being mistaken for a deployable mixed-protocol state.
