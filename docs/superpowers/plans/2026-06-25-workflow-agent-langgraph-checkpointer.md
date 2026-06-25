# Workflow Agent LangGraph Checkpointer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `workflow_agent` from a custom sequential orchestrator into a production-grade LangGraph `StateGraph` workflow whose execution state is persisted by the existing runtime checkpointer while `thread_store` remains the product-facing projection layer.

**Architecture:** Build a real LangGraph graph for the fixed Inquiry -> Evidence -> Syndrome -> Answer -> Safety workflow. The compiled graph owns internal workflow state, node transitions, message state, and resume/update behavior through `runtime_state.state.checkpointer`; the existing `thread_store` keeps run status, visible conversation, final responses, validation metadata, and trace projection. Keep the current `WorkflowAgent.astream` adapter as a thin compatibility facade around the compiled graph so existing SSE/runtime code can keep using `messages` and `values` stream modes.

**Tech Stack:** Python 3.10, `unittest`, Pydantic v2, LangGraph 1.2.4, `langgraph-checkpoint` 4.1.1, LangChain Core messages, existing `ChatOpenAI.with_structured_output(..., method="json_schema", strict=True)`, existing `retrieve_tcm_knowledge`, existing `run_agent` runtime.

---

## Ground Rules

- Do not commit unless the user explicitly asks.
- Do not edit `.env` or unrelated `.gitignore` changes.
- Preserve the current user-facing SSE contract:
  - `event: messages` for visible answer chunks.
  - `event: final` from `run_agent`.
  - `assistant_message`, `pending_clarification`, and `references` shape remains runtime-owned.
- Preserve `thread_store` as product projection:
  - `conversation`
  - `last_validation`
  - `last_allowed_terms`
  - `last_rewritten`
  - `last_agent_trace`
  - run/thread status
- Move workflow execution state into LangGraph:
  - `messages`
  - `user_text`
  - `conversation`
  - `inquiry`
  - `evidence`
  - `syndrome`
  - `answer`
  - `safety`
  - `agent_trace`
  - `needs_clarification`
- Keep `EvidenceAgent` as the only component that calls `retrieve_tcm_knowledge`.
- Keep the existing guardrail middleware as final runtime defense after graph output.
- Use path-limited staging if staging is requested.

## Local API Facts Verified

Installed package versions:

```text
langgraph==1.2.4
langgraph-checkpoint==4.1.1
langgraph-checkpoint-postgres is not installed in this environment
```

Important local signatures:

```python
StateGraph(
    state_schema,
    context_schema=None,
    *,
    input_schema=None,
    output_schema=None,
)

StateGraph.add_node(node, action=None, *, ...)
StateGraph.add_edge(start_key, end_key)
StateGraph.add_conditional_edges(source, path, path_map=None)
StateGraph.compile(checkpointer=None, *, cache=None, store=None, ...)
```

Message reducer:

```python
from langgraph.graph.message import add_messages
```

## Target File Structure

Create:

- `app/agents/workflow_agent/state.py`
  - Defines `WorkflowState` and reducer annotations.
  - Defines serializable fields used by graph nodes and runtime projection.

- `app/agents/workflow_agent/graph.py`
  - Builds the LangGraph `StateGraph`.
  - Defines node functions.
  - Defines conditional routing functions.
  - Compiles with the runtime checkpointer.

- `tests/test_workflow_agent_graph.py`
  - Proves graph construction uses `StateGraph`.
  - Proves graph nodes and conditional routes are present.
  - Proves `TCMWorkflow.run()` executes through the compiled graph.

Modify:

- `app/agents/workflow_agent/workflow.py`
  - Remove manual sequential orchestration from `run()`.
  - Hold a compiled graph.
  - Convert graph final state into `WorkflowRunResult`.

- `app/agents/workflow_agent/agent.py`
  - Build `TCMWorkflow` with `runtime_state.state.checkpointer`.
  - Keep `astream`, `aget_state`, and `aupdate_state` compatible with `run_agent`.
  - Prefer delegating state operations to the compiled graph where safe.

- `tests/test_workflow_agent_flow.py`
  - Update expectations so workflow tests assert graph-backed execution rather than only sequential method calls.
  - Keep behavior assertions around clarification, retrieval isolation, safety rewrite, and runtime final event.

- `tests/test_workflow_agent_components.py`
  - Keep component boundary tests unchanged unless imports move.

Do not modify:

- `app/agents/lead_agent/*`
- `app/runtime/public_messages.py`
- `.env`
- unrelated `.gitignore`

---

### Task 1: Add Graph-State Tests First

**Files:**
- Create: `tests/test_workflow_agent_graph.py`

- [ ] **Step 1: Write failing tests for graph structure and workflow graph ownership**

Create `tests/test_workflow_agent_graph.py`:

```python
import unittest

from langgraph.graph.state import CompiledStateGraph

from app.agents.workflow_agent.models import (
    AnswerDraft,
    InquiryState,
    KnownFacts,
    PatternCandidate,
    SafetyReview,
    SyndromeAnalysis,
)
from app.agents.workflow_agent.workflow import TCMWorkflow


class FakeStructuredRunnable:
    def __init__(self, model, schema):
        self.model = model
        self.schema = schema

    async def ainvoke(self, messages):
        self.model.invocations.append({"schema": self.schema, "messages": messages})
        responses = self.model.responses_by_schema[self.schema]
        return responses.pop(0)


class FakeWorkflowModel:
    def __init__(self):
        self.responses_by_schema = {
            InquiryState: [
                InquiryState(
                    chief_complaint="胃胀",
                    known_facts=KnownFacts(
                        duration="两周",
                        triggers=["油腻后加重"],
                        associated_symptoms=["嗳气"],
                    ),
                    information_sufficiency="sufficient",
                )
            ],
            SyndromeAnalysis: [
                SyndromeAnalysis(
                    possible_patterns=[
                        PatternCandidate(
                            term="食滞",
                            supporting_evidence=["E1"],
                            confidence="medium",
                            reason="用户提到油腻后胃胀加重。",
                        )
                    ],
                    not_enough_for_diagnosis=True,
                    need_more_info=["舌象"],
                )
            ],
            AnswerDraft: [AnswerDraft(draft_answer="可能与食滞相关，但不能诊断。[E1]")],
            SafetyReview: [SafetyReview(final_safety_level="low", rewrite_required=False)],
        }
        self.invocations = []

    def with_structured_output(self, schema, *, method, strict):
        return FakeStructuredRunnable(self, schema)


async def fake_retriever(query: str, mode: str) -> str:
    return (
        "检索状态：ok\n"
        "检索模式：hybrid_parent\n\n"
        "[E1]\n"
        "证据角色：syndrome_pattern\n"
        "原文：饮食停滞可见胃脘胀满。\n"
        "来源：《景岳全书》胃脘\n\n"
        "允许使用的专业术语：\n"
        "- 食滞\n\n"
        "回答约束：\n"
        "- 不得推荐方药。"
    )


class WorkflowAgentGraphTests(unittest.IsolatedAsyncioTestCase):
    def test_workflow_owns_compiled_langgraph(self):
        from app.agents.workflow_agent.components.evidence import EvidenceAgent

        workflow = TCMWorkflow(
            model=FakeWorkflowModel(),
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
        )

        self.assertIsInstance(workflow.graph, CompiledStateGraph)

    async def test_workflow_run_executes_through_graph(self):
        from app.agents.workflow_agent.components.evidence import EvidenceAgent

        model = FakeWorkflowModel()
        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
        )

        result = await workflow.run(
            user_text="胃胀两周，油腻后加重，嗳气。",
            conversation=[],
        )

        self.assertFalse(result.needs_clarification)
        self.assertIn("[E1]", result.final_text)
        self.assertEqual(
            [event["agent"] for event in result.agent_trace],
            ["InquiryAgent", "EvidenceAgent", "SyndromeAgent", "AnswerAgent", "SafetyAgent"],
        )
```

- [ ] **Step 2: Run graph tests and verify red**

Run:

```powershell
python -m unittest tests.test_workflow_agent_graph
```

Expected:

```text
FAILED
AttributeError: 'TCMWorkflow' object has no attribute 'graph'
```

If the failure is an import error for `CompiledStateGraph`, inspect the local class path with:

```powershell
python - <<'PY'
import langgraph.graph.state as state
print([name for name in dir(state) if "Compiled" in name])
PY
```

Then update the test import to the installed class path before implementing production code.

---

### Task 2: Define WorkflowState

**Files:**
- Create: `app/agents/workflow_agent/state.py`
- Test: `tests/test_workflow_agent_graph.py`

- [ ] **Step 1: Add state schema**

Create `app/agents/workflow_agent/state.py`:

```python
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agents.workflow_agent.models import (
    AnswerDraft,
    EvidenceResult,
    InquiryState,
    SafetyReview,
    SyndromeAnalysis,
)


def append_trace(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [*(left or []), *(right or [])]


class WorkflowState(TypedDict, total=False):
    user_text: str
    conversation: list[dict[str, Any]]
    messages: Annotated[list[BaseMessage], add_messages]
    inquiry: InquiryState
    evidence: EvidenceResult
    syndrome: SyndromeAnalysis
    answer: AnswerDraft
    safety: SafetyReview
    needs_clarification: bool
    final_text: str
    agent_trace: Annotated[list[dict[str, Any]], append_trace]
```

- [ ] **Step 2: Add state reducer unit tests**

Append to `tests/test_workflow_agent_graph.py`:

```python
class WorkflowStateTests(unittest.TestCase):
    def test_append_trace_preserves_existing_events(self):
        from app.agents.workflow_agent.state import append_trace

        self.assertEqual(
            append_trace([{"agent": "InquiryAgent"}], [{"agent": "EvidenceAgent"}]),
            [{"agent": "InquiryAgent"}, {"agent": "EvidenceAgent"}],
        )

    def test_workflow_state_uses_message_reducer(self):
        from typing import get_type_hints
        from app.agents.workflow_agent.state import WorkflowState

        hints = get_type_hints(WorkflowState, include_extras=True)
        self.assertIn("messages", hints)
        self.assertIn("agent_trace", hints)
```

- [ ] **Step 3: Run graph tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_graph
```

Expected:

```text
WorkflowStateTests pass.
WorkflowAgentGraphTests still fail because TCMWorkflow.graph is not implemented.
```

---

### Task 3: Build LangGraph Nodes and Routes

**Files:**
- Create: `app/agents/workflow_agent/graph.py`
- Modify: `app/agents/workflow_agent/workflow.py`
- Test: `tests/test_workflow_agent_graph.py`

- [ ] **Step 1: Add graph builder**

Create `app/agents/workflow_agent/graph.py`:

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agents.workflow_agent.components.answer import AnswerAgent
from app.agents.workflow_agent.components.evidence import EvidenceAgent
from app.agents.workflow_agent.components.inquiry import InquiryAgent
from app.agents.workflow_agent.components.safety import SafetyAgent
from app.agents.workflow_agent.components.syndrome import SyndromeAgent
from app.agents.workflow_agent.state import WorkflowState
from app.middlewares.clarification_controller import format_clarification_questions


def route_after_inquiry(state: WorkflowState) -> Literal["clarification", "evidence"]:
    inquiry = state["inquiry"]
    return "clarification" if inquiry.should_pause_for_clarification else "evidence"


def route_after_initial_safety(
    state: WorkflowState,
) -> Literal["answer_rewrite", "finalize"]:
    safety = state["safety"]
    return "answer_rewrite" if safety.rewrite_required else "finalize"


def route_after_rewrite_safety(
    state: WorkflowState,
) -> Literal["safe_fallback", "finalize"]:
    safety = state["safety"]
    return "safe_fallback" if safety.rewrite_required else "finalize"


def build_workflow_graph(
    *,
    inquiry_agent: InquiryAgent,
    evidence_agent: EvidenceAgent,
    syndrome_agent: SyndromeAgent,
    answer_agent: AnswerAgent,
    safety_agent: SafetyAgent,
    checkpointer: Any | None = None,
):
    async def inquiry_node(state: WorkflowState) -> dict[str, Any]:
        inquiry = await inquiry_agent.assess(
            user_text=state["user_text"],
            conversation=state.get("conversation", []),
        )
        return {
            "inquiry": inquiry,
            "agent_trace": [
                {
                    "agent": "InquiryAgent",
                    "model_call": True,
                    "information_sufficiency": inquiry.information_sufficiency,
                    "should_pause_for_clarification": inquiry.should_pause_for_clarification,
                }
            ],
        }

    async def clarification_node(state: WorkflowState) -> dict[str, Any]:
        inquiry = state["inquiry"]
        tool_call_id = "workflow-clarification-1"
        clarification_text = format_clarification_questions(
            inquiry.clarification_questions
        )
        return {
            "needs_clarification": True,
            "final_text": clarification_text,
            "messages": [
                AIMessage(
                    content="",
                    id="workflow-clarification-ai-1",
                    tool_calls=[
                        {
                            "id": tool_call_id,
                            "name": "ask_clarification",
                            "args": {"questions": inquiry.clarification_questions},
                        }
                    ],
                ),
                ToolMessage(
                    id=f"clarification:{tool_call_id}",
                    name="ask_clarification",
                    tool_call_id=tool_call_id,
                    content=clarification_text,
                ),
            ],
        }

    async def evidence_node(state: WorkflowState) -> dict[str, Any]:
        evidence = await evidence_agent.retrieve(
            user_text=state["user_text"],
            inquiry=state["inquiry"],
        )
        return {
            "evidence": evidence,
            "messages": [
                AIMessage(
                    content="",
                    id="workflow-retrieval-ai-1",
                    tool_calls=[
                        {
                            "id": "workflow-retrieval-1",
                            "name": "retrieve_tcm_knowledge",
                            "args": {"query": state["user_text"], "mode": "hybrid"},
                        }
                    ],
                ),
                ToolMessage(
                    id="tool:workflow-retrieval-1",
                    name="retrieve_tcm_knowledge",
                    tool_call_id="workflow-retrieval-1",
                    content=evidence.raw_tool_content,
                ),
            ],
            "agent_trace": [
                {
                    "agent": "EvidenceAgent",
                    "retrieval_status": evidence.retrieval_status,
                    "retrieval_mode": evidence.retrieval_mode,
                    "evidence_count": len(evidence.evidence),
                }
            ],
        }

    async def syndrome_node(state: WorkflowState) -> dict[str, Any]:
        syndrome = await syndrome_agent.analyze(
            user_text=state["user_text"],
            inquiry=state["inquiry"],
            evidence=state["evidence"],
        )
        return {
            "syndrome": syndrome,
            "agent_trace": [
                {
                    "agent": "SyndromeAgent",
                    "model_call": True,
                    "possible_patterns": [
                        pattern.term for pattern in syndrome.possible_patterns
                    ],
                }
            ],
        }

    async def answer_draft_node(state: WorkflowState) -> dict[str, Any]:
        answer = await answer_agent.compose(
            user_text=state["user_text"],
            inquiry=state["inquiry"],
            evidence=state["evidence"],
            syndrome=state["syndrome"],
        )
        return {
            "answer": answer,
            "agent_trace": [
                {"agent": "AnswerAgent", "stage": "draft", "model_call": True}
            ],
        }

    async def safety_initial_node(state: WorkflowState) -> dict[str, Any]:
        safety = await safety_agent.review(
            draft_answer=state["answer"].draft_answer,
            inquiry=state["inquiry"],
            evidence=state["evidence"],
            syndrome=state["syndrome"],
        )
        return {
            "safety": safety,
            "agent_trace": [
                {
                    "agent": "SafetyAgent",
                    "stage": "initial",
                    "model_call": True,
                    "final_safety_level": safety.final_safety_level,
                    "rewrite_required": safety.rewrite_required,
                }
            ],
        }

    async def answer_rewrite_node(state: WorkflowState) -> dict[str, Any]:
        answer = await answer_agent.compose(
            user_text=state["user_text"],
            inquiry=state["inquiry"],
            evidence=state["evidence"],
            syndrome=state["syndrome"],
            safety_review=state["safety"],
        )
        return {
            "answer": answer,
            "agent_trace": [
                {"agent": "AnswerAgent", "stage": "rewrite", "model_call": True}
            ],
        }

    async def safety_rewrite_node(state: WorkflowState) -> dict[str, Any]:
        safety = await safety_agent.review(
            draft_answer=state["answer"].draft_answer,
            inquiry=state["inquiry"],
            evidence=state["evidence"],
            syndrome=state["syndrome"],
        )
        return {
            "safety": safety,
            "agent_trace": [
                {
                    "agent": "SafetyAgent",
                    "stage": "rewrite",
                    "model_call": True,
                    "final_safety_level": safety.final_safety_level,
                    "rewrite_required": safety.rewrite_required,
                }
            ],
        }

    async def safe_fallback_node(state: WorkflowState) -> dict[str, Any]:
        answer = answer_agent.safe_fallback(state["inquiry"])
        safety = await safety_agent.review(
            draft_answer=answer.draft_answer,
            inquiry=state["inquiry"],
            evidence=state["evidence"],
            syndrome=state["syndrome"],
        )
        return {
            "answer": answer,
            "safety": safety,
            "agent_trace": [
                {"agent": "AnswerAgent", "stage": "safe_fallback"},
                {
                    "agent": "SafetyAgent",
                    "stage": "safe_fallback",
                    "model_call": True,
                    "final_safety_level": safety.final_safety_level,
                    "rewrite_required": safety.rewrite_required,
                },
            ],
        }

    async def finalize_node(state: WorkflowState) -> dict[str, Any]:
        answer = state["answer"]
        return {
            "needs_clarification": False,
            "final_text": answer.draft_answer,
            "messages": [
                AIMessage(content=answer.draft_answer, id="workflow-final-ai-1")
            ],
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("inquiry", inquiry_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("syndrome", syndrome_node)
    graph.add_node("answer_draft", answer_draft_node)
    graph.add_node("safety_initial", safety_initial_node)
    graph.add_node("answer_rewrite", answer_rewrite_node)
    graph.add_node("safety_rewrite", safety_rewrite_node)
    graph.add_node("safe_fallback", safe_fallback_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "inquiry")
    graph.add_conditional_edges(
        "inquiry",
        route_after_inquiry,
        {"clarification": "clarification", "evidence": "evidence"},
    )
    graph.add_edge("clarification", END)
    graph.add_edge("evidence", "syndrome")
    graph.add_edge("syndrome", "answer_draft")
    graph.add_edge("answer_draft", "safety_initial")
    graph.add_conditional_edges(
        "safety_initial",
        route_after_initial_safety,
        {"answer_rewrite": "answer_rewrite", "finalize": "finalize"},
    )
    graph.add_edge("answer_rewrite", "safety_rewrite")
    graph.add_conditional_edges(
        "safety_rewrite",
        route_after_rewrite_safety,
        {"safe_fallback": "safe_fallback", "finalize": "finalize"},
    )
    graph.add_edge("safe_fallback", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
```

- [ ] **Step 2: Import the graph builder in workflow.py**

Add to `app/agents/workflow_agent/workflow.py`:

```python
from app.agents.workflow_agent.graph import build_workflow_graph
```

- [ ] **Step 3: Run tests and verify graph still not wired**

Run:

```powershell
python -m unittest tests.test_workflow_agent_graph
```

Expected:

```text
FAILED
AttributeError: 'TCMWorkflow' object has no attribute 'graph'
```

The graph builder exists, but `TCMWorkflow` has not been changed to use it yet.

---

### Task 4: Convert TCMWorkflow to Compiled Graph Execution

**Files:**
- Modify: `app/agents/workflow_agent/workflow.py`
- Test: `tests/test_workflow_agent_graph.py`, `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Update TCMWorkflow constructor**

Replace the current constructor body with:

```python
        if (
            model is None
            and (
                inquiry_agent is None
                or syndrome_agent is None
                or answer_agent is None
                or safety_agent is None
            )
        ):
            raise ValueError(
                "TCMWorkflow requires a ChatOpenAI-compatible model when LLM agents "
                "are not supplied explicitly."
            )

        self.inquiry_agent = inquiry_agent or InquiryAgent(model)
        self.evidence_agent = evidence_agent or EvidenceAgent()
        self.syndrome_agent = syndrome_agent or SyndromeAgent(model)
        self.answer_agent = answer_agent or AnswerAgent(model)
        self.safety_agent = safety_agent or SafetyAgent(model)
        self.graph = build_workflow_graph(
            inquiry_agent=self.inquiry_agent,
            evidence_agent=self.evidence_agent,
            syndrome_agent=self.syndrome_agent,
            answer_agent=self.answer_agent,
            safety_agent=self.safety_agent,
            checkpointer=None,
        )
```

This first version intentionally uses `checkpointer=None` so graph behavior can be proven before wiring runtime persistence.

- [ ] **Step 2: Replace run() with graph invocation**

Replace `TCMWorkflow.run()` with:

```python
    async def run(
        self,
        user_text: str,
        conversation: Sequence[object] | None = None,
        config: dict[str, Any] | None = None,
    ) -> WorkflowRunResult:
        result = await self.graph.ainvoke(
            {
                "user_text": user_text,
                "conversation": [
                    item for item in conversation or [] if isinstance(item, dict)
                ],
                "messages": [],
                "agent_trace": [],
            },
            config=config,
        )
        return WorkflowRunResult(
            messages=list(result.get("messages", [])),
            final_text=str(result.get("final_text", "")),
            needs_clarification=bool(result.get("needs_clarification")),
            agent_trace=list(result.get("agent_trace", [])),
        )
```

- [ ] **Step 3: Run graph tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_graph
```

Expected:

```text
OK
```

- [ ] **Step 4: Run existing workflow tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_components tests.test_workflow_agent_llm tests.test_workflow_agent_flow tests.test_workflow_agent_models tests.test_workflow_agent_registry
```

Expected:

```text
OK
```

If ordering differs because LangGraph reducers append trace in a different order, inspect the actual `agent_trace` and fix graph edge order rather than weakening the test.

---

### Task 5: Wire Runtime Checkpointer Into Workflow Graph

**Files:**
- Modify: `app/agents/workflow_agent/workflow.py`
- Modify: `app/agents/workflow_agent/agent.py`
- Test: `tests/test_workflow_agent_graph.py`, `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Add checkpointer parameter to TCMWorkflow**

Change the constructor signature in `app/agents/workflow_agent/workflow.py`:

```python
        checkpointer: Any | None = None,
```

Pass it into graph construction:

```python
        self.graph = build_workflow_graph(
            inquiry_agent=self.inquiry_agent,
            evidence_agent=self.evidence_agent,
            syndrome_agent=self.syndrome_agent,
            answer_agent=self.answer_agent,
            safety_agent=self.safety_agent,
            checkpointer=checkpointer,
        )
```

- [ ] **Step 2: Make factory use runtime checkpointer**

Update `app/agents/workflow_agent/agent.py`:

```python
def make_workflow_agent(context: dict[str, Any] | None = None) -> WorkflowAgent:
    model = build_workflow_model(context)
    return WorkflowAgent(
        workflow=TCMWorkflow(
            model=model,
            checkpointer=runtime_state.state.checkpointer,
        )
    )
```

- [ ] **Step 3: Pass LangGraph config into workflow.run**

Update `WorkflowAgent.astream()` in `app/agents/workflow_agent/agent.py`:

```python
        workflow_result = await self.workflow.run(
            user_text=user_text,
            conversation=conversation,
            config=config,
        )
```

- [ ] **Step 4: Add test for factory checkpointer wiring**

Append to `tests/test_workflow_agent_graph.py`:

```python
class WorkflowFactoryCheckpointerTests(unittest.TestCase):
    def test_make_workflow_agent_passes_runtime_checkpointer_to_workflow(self):
        from app.agents.workflow_agent import agent as workflow_agent_module
        from app.runtime import state as runtime_state

        original_checkpointer = runtime_state.state.checkpointer
        sentinel = object()
        runtime_state.state.checkpointer = sentinel
        try:
            with unittest.mock.patch.object(
                workflow_agent_module,
                "build_workflow_model",
                return_value=FakeWorkflowModel(),
            ), unittest.mock.patch.object(
                workflow_agent_module,
                "TCMWorkflow",
            ) as workflow_cls:
                workflow_agent_module.make_workflow_agent({})

            self.assertIs(workflow_cls.call_args.kwargs["checkpointer"], sentinel)
        finally:
            runtime_state.state.checkpointer = original_checkpointer
```

Also add at the top:

```python
import unittest.mock
```

- [ ] **Step 5: Run checkpointer tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_graph
```

Expected:

```text
OK
```

---

### Task 6: Delegate WorkflowAgent State Methods to the Compiled Graph

**Files:**
- Modify: `app/agents/workflow_agent/agent.py`
- Test: `tests/test_workflow_agent_flow.py`, `tests/test_workflow_agent_graph.py`

- [ ] **Step 1: Add graph-backed state delegation tests**

Add to `tests/test_workflow_agent_graph.py`:

```python
class WorkflowAgentStateDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_agent_state_methods_delegate_to_graph_when_available(self):
        from types import SimpleNamespace
        from app.agents.workflow_agent.agent import WorkflowAgent

        calls = []

        class FakeGraph:
            async def aget_state(self, config):
                calls.append(("aget_state", config))
                return SimpleNamespace(values={"messages": [{"content": "from graph"}]}, next=())

            async def aupdate_state(self, config, values):
                calls.append(("aupdate_state", config, values))

        class FakeWorkflow:
            graph = FakeGraph()

        agent = WorkflowAgent(workflow=FakeWorkflow(), thread_store=object())
        config = {"configurable": {"thread_id": "thread-1"}}

        snapshot = await agent.aget_state(config)
        await agent.aupdate_state(config, {"messages": []})

        self.assertEqual(snapshot.values["messages"][0]["content"], "from graph")
        self.assertEqual(calls[0], ("aget_state", config))
        self.assertEqual(calls[1], ("aupdate_state", config, {"messages": []}))
```

- [ ] **Step 2: Verify red**

Run:

```powershell
python -m unittest tests.test_workflow_agent_graph
```

Expected:

```text
FAILED
```

Current `WorkflowAgent.aget_state()` reads `thread_store`, so it will not delegate to `workflow.graph`.

- [ ] **Step 3: Implement delegation with fallback**

Update `WorkflowAgent.aget_state()`:

```python
    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        graph = getattr(self.workflow, "graph", None)
        if graph is not None and hasattr(graph, "aget_state"):
            return await graph.aget_state(config)
        return SimpleNamespace(
            values={"messages": await self._read_messages(self._thread_id(config))},
            next=(),
        )
```

Update `WorkflowAgent.aupdate_state()`:

```python
    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        graph = getattr(self.workflow, "graph", None)
        if graph is not None and hasattr(graph, "aupdate_state"):
            await graph.aupdate_state(config, values)
            return

        thread_id = self._thread_id(config)
        messages = list(await self._read_messages(thread_id))
        ...
```

Keep the existing thread-store fallback body unchanged after the early return.

- [ ] **Step 4: Run delegation tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_graph
```

Expected:

```text
OK
```

---

### Task 7: Preserve Runtime Projection and Clarification Flow

**Files:**
- Modify: `app/agents/workflow_agent/agent.py`
- Modify: `app/runtime/runs/worker.py` only if tests prove projection drift
- Test: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Keep WorkflowAgent.astream projection behavior**

Ensure `WorkflowAgent.astream()` still:

```python
current_messages = [
    HumanMessage(content=user_text),
    *workflow_result.messages,
]
serialized_messages = [
    *previous_messages,
    *(serialize_message(message) for message in current_messages),
]
await self.thread_store.update_values(
    thread_id,
    {
        "messages": serialized_messages,
        "last_agent_trace": workflow_result.agent_trace,
    },
)
```

This preserves product projection even though graph/checkpointer owns execution state.

- [ ] **Step 2: Run runtime workflow tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow
```

Expected:

```text
OK
```

Pay particular attention to:

- `test_workflow_agent_runs_through_existing_runtime_final_event`
- `test_workflow_agent_clarification_uses_existing_waiting_flow`

These prove the existing `final` event and waiting clarification status are preserved.

---

### Task 8: Verify Retrieval Ownership and Safety Conditional Edges

**Files:**
- Test: `tests/test_workflow_agent_flow.py`
- Test: `tests/test_workflow_agent_graph.py`

- [ ] **Step 1: Add route-unit tests**

Append to `tests/test_workflow_agent_graph.py`:

```python
class WorkflowGraphRouteTests(unittest.TestCase):
    def test_route_after_inquiry_pauses_only_when_inquiry_requires_clarification(self):
        from app.agents.workflow_agent.graph import route_after_inquiry

        self.assertEqual(
            route_after_inquiry(
                {
                    "inquiry": InquiryState(
                        chief_complaint="胃胀",
                        information_sufficiency="insufficient",
                        clarification_questions=["持续多久了？"],
                        should_pause_for_clarification=True,
                    )
                }
            ),
            "clarification",
        )
        self.assertEqual(
            route_after_inquiry(
                {
                    "inquiry": InquiryState(
                        chief_complaint="胃胀",
                        information_sufficiency="sufficient",
                    )
                }
            ),
            "evidence",
        )

    def test_safety_routes_to_rewrite_only_when_required(self):
        from app.agents.workflow_agent.graph import (
            route_after_initial_safety,
            route_after_rewrite_safety,
        )

        self.assertEqual(
            route_after_initial_safety(
                {"safety": SafetyReview(rewrite_required=True)}
            ),
            "answer_rewrite",
        )
        self.assertEqual(
            route_after_initial_safety(
                {"safety": SafetyReview(rewrite_required=False)}
            ),
            "finalize",
        )
        self.assertEqual(
            route_after_rewrite_safety(
                {"safety": SafetyReview(rewrite_required=True)}
            ),
            "safe_fallback",
        )
```

- [ ] **Step 2: Run route tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_graph
```

Expected:

```text
OK
```

- [ ] **Step 3: Run retrieval ownership and rewrite tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow.WorkflowLLMBackedTests.test_evidence_agent_is_only_component_that_calls_retrieval tests.test_workflow_agent_flow.WorkflowLLMBackedTests.test_safety_rewrite_runs_answer_again_and_rechecks_final_answer
```

Expected:

```text
OK
```

---

### Task 9: Full Targeted Verification

**Files:**
- Inspect only.

- [ ] **Step 1: Compile changed files**

Run:

```powershell
python -m py_compile `
  app\agents\workflow_agent\agent.py `
  app\agents\workflow_agent\workflow.py `
  app\agents\workflow_agent\graph.py `
  app\agents\workflow_agent\state.py `
  app\agents\workflow_agent\components\__init__.py `
  app\agents\workflow_agent\components\answer.py `
  app\agents\workflow_agent\components\base.py `
  app\agents\workflow_agent\components\evidence.py `
  app\agents\workflow_agent\components\inquiry.py `
  app\agents\workflow_agent\components\safety.py `
  app\agents\workflow_agent\components\syndrome.py
```

Expected:

```text
exit code 0
```

- [ ] **Step 2: Run workflow-focused tests**

Run:

```powershell
python -m unittest `
  tests.test_workflow_agent_graph `
  tests.test_workflow_agent_components `
  tests.test_workflow_agent_llm `
  tests.test_workflow_agent_flow `
  tests.test_workflow_agent_models `
  tests.test_workflow_agent_registry
```

Expected:

```text
OK
```

- [ ] **Step 3: Run diff check**

Run:

```powershell
git diff --check -- `
  app\agents\workflow_agent `
  tests\test_workflow_agent_graph.py `
  tests\test_workflow_agent_flow.py `
  tests\test_workflow_agent_components.py
```

Expected:

```text
exit code 0
```

- [ ] **Step 4: Inspect git status**

Run:

```powershell
git status --short
git diff --cached --name-status
git diff --name-status
```

Expected:

```text
.env remains local-only.
.gitignore remains unrelated unless the user explicitly asked to include it.
LangGraph workflow files are the only newly changed feature files.
No commit is created unless the user explicitly asks.
```

## Self-Review

Spec coverage:

- Real LangGraph orchestration: Tasks 1, 3, and 4.
- Checkpointer ownership of workflow execution state: Tasks 5 and 6.
- `thread_store` remains product projection: Task 7.
- EvidenceAgent-only retrieval: Task 8.
- Safety rewrite through conditional edges: Tasks 3 and 8.
- Existing SSE/runtime contract preserved: Tasks 7 and 9.

Placeholder scan:

- No deferred implementation placeholders are present.
- Every task includes concrete file paths, code snippets, and commands.

Type consistency:

- `WorkflowState` is the single graph state schema.
- `TCMWorkflow.run()` returns the existing `WorkflowRunResult`.
- `WorkflowAgent.astream()` remains the runtime adapter used by `run_agent`.
- `build_workflow_graph(..., checkpointer=...)` is the only graph construction entry point.

Risk notes:

- This plan deliberately uses the existing runtime checkpointer object instead of adding `langgraph-checkpoint-postgres`; that package is not installed locally.
- If future production wants Postgres checkpointer persistence, use the existing `app/checkpoints/factory.py` path and add/install the missing dependency through a separate dependency-management change.
- Do not replace `thread_store`; reduce it to product projection while LangGraph owns workflow execution state.
