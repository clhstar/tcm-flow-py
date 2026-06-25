# LLM-Backed Workflow Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the staged deterministic `workflow_agent` implementation with a real LLM-backed workflow multi-agent pipeline while preserving the fixed Inquiry -> Evidence -> Syndrome -> Answer -> Safety order.

**Architecture:** Keep the existing runtime contract (`WorkflowAgent.astream`, `aget_state`, `aupdate_state`) so SSE and thread storage remain stable. Implement each non-retrieval workflow sub-agent as a LangChain `ChatOpenAI.with_structured_output(...)` call using Pydantic contracts; keep `EvidenceAgent` as the only component allowed to call `retrieve_tcm_knowledge`, with optional LLM query planning and evidence normalization. Preserve the runtime-level guardrail middleware as a final defense, but do not rely on it as a substitute for `SafetyAgent`.

**Tech Stack:** Python 3.10, `unittest`, Pydantic v2, LangChain 1.3.4, `langchain-openai` `ChatOpenAI`, official `ChatOpenAI.with_structured_output(schema, method="json_schema", strict=True)` style, existing `retrieve_tcm_knowledge`, existing `run_agent` runtime.

---

## References Checked

- LangChain `create_agent` reference: `https://reference.langchain.com/python/langchain/agents/`
  - Supports a chat model instance such as `ChatOpenAI`, `tools`, `system_prompt`, and structured `response_format`.
- LangChain `ChatOpenAI` reference: `https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI`
- LangChain `ChatOpenAI.with_structured_output` reference: `https://reference.langchain.com/python/langchain-openai/chat_models/base/ChatOpenAI/with_structured_output`
- Local installed signature verified in `.venv`:
  - `ChatOpenAI.with_structured_output(schema=None, *, method='json_schema', include_raw=False, strict=None, tools=None, **kwargs)`
  - `create_agent(model, tools=None, *, system_prompt=None, middleware=(), response_format=None, ...)`

## Current Git Boundary

- `main` has been soft-reset to `cbac4cf`.
- The earlier deterministic workflow implementation is currently staged.
- That staged implementation must not be committed as-is.
- Implementation work must replace the staged deterministic internals with the LLM-backed design below before any code commit.
- Existing local `.env` and `.gitignore` changes are not part of this feature and must not be edited for this work.

## File Structure

Create:

- `app/agents/workflow_agent/llm.py`
  - Builds the workflow `ChatOpenAI` model from app settings and request context.
  - Provides a small `structured_model(...)` helper using official `with_structured_output(...)` parameters.

- `app/agents/workflow_agent/prompts.py`
  - Holds system prompts and user-prompt builders for InquiryAgent, EvidenceAgent, SyndromeAgent, AnswerAgent, and SafetyAgent.
  - Keeps prompt text out of orchestration code.

Modify:

- `app/agents/workflow_agent/models.py`
  - Keep existing contracts, add any missing structured-output fields needed for LLM evidence query planning or answer rewrite.
  - Preserve strict Pydantic validation for safety-sensitive fields.

- `app/agents/workflow_agent/workflow.py`
  - Replace deterministic Inquiry/Syndrome/Answer/Safety internals with LLM-backed calls.
  - Keep EvidenceAgent as the sole retrieval tool owner.
  - Keep fixed orchestration order and final safety re-check.

- `app/agents/workflow_agent/agent.py`
  - Pass request `context` into `TCMWorkflow` so it can build the configured model.
  - Preserve runtime adapter interface and SSE payload shape.

- `tests/test_workflow_agent_flow.py`
  - Replace deterministic behavior expectations with fake model-call expectations.
  - Ensure each LLM-backed sub-agent actually invokes the structured model.

- `tests/test_workflow_agent_models.py`
  - Update model contract tests only if the schema changes.

- `tests/test_workflow_agent_registry.py`
  - Keep lazy registry tests. No functional change expected.

Do not modify:

- `app/agents/lead_agent/*`
- `.env`
- `.gitignore`
- Runtime public response shape in `app/runtime/public_messages.py`

---

### Task 0: Protect the Index Boundary

**Files:**
- Inspect only.

- [ ] **Step 1: Confirm old workflow code is staged and not committed**

Run:

```powershell
git branch --show-current
git rev-parse --short HEAD
git status --short
git diff --cached --name-status
```

Expected:

```text
main
cbac4cf
```

Expected staged paths include:

```text
app/agents/workflow_agent/
app/agents/registry.py
app/runtime/runs/worker.py
tests/test_workflow_agent_*.py
```

Expected local-only paths may include:

```text
.env
.gitignore
```

- [ ] **Step 2: Confirm no commit will be created in this task**

Run:

```powershell
git log --oneline -1
```

Expected:

```text
cbac4cf docs: plan workflow agent implementation
```

Do not commit.

---

### Task 1: Add LLM Construction Helpers

**Files:**
- Create: `app/agents/workflow_agent/llm.py`
- Test: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Write failing tests for workflow model construction**

Add these tests to `tests/test_workflow_agent_flow.py`:

```python
class WorkflowModelConstructionTests(unittest.TestCase):
    def test_build_workflow_model_uses_context_over_settings(self):
        from app.agents.workflow_agent import llm as workflow_llm

        class Settings:
            openai_model = "settings-model"
            openai_base_url = "https://settings.example/v1"
            openai_api_key = "settings-key"

        with patch.object(workflow_llm, "get_settings", return_value=Settings()):
            with patch.object(workflow_llm, "ChatOpenAI") as chat_openai:
                workflow_llm.build_workflow_model(
                    {
                        "model_name": "context-model",
                        "temperature": 0.05,
                        "streaming": True,
                    }
                )

        self.assertEqual(chat_openai.call_args.kwargs["model"], "context-model")
        self.assertEqual(chat_openai.call_args.kwargs["base_url"], "https://settings.example/v1")
        self.assertEqual(chat_openai.call_args.kwargs["api_key"], "settings-key")
        self.assertEqual(chat_openai.call_args.kwargs["temperature"], 0.05)
        self.assertIs(chat_openai.call_args.kwargs["streaming"], False)

    def test_structured_model_uses_official_json_schema_format(self):
        from app.agents.workflow_agent.llm import structured_model
        from app.agents.workflow_agent.models import InquiryState

        model = unittest.mock.Mock()
        structured_model(model, InquiryState)

        model.with_structured_output.assert_called_once_with(
            InquiryState,
            method="json_schema",
            strict=True,
        )
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowModelConstructionTests -v
```

Expected:

```text
ERROR: ModuleNotFoundError: No module named 'app.agents.workflow_agent.llm'
```

- [ ] **Step 3: Implement `llm.py`**

Create `app/agents/workflow_agent/llm.py`:

```python
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import get_settings


def build_workflow_model(context: dict[str, Any] | None = None) -> ChatOpenAI:
    context = context or {}
    settings = get_settings()

    return ChatOpenAI(
        model=context.get("model_name") or settings.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=context.get("temperature", 0.1),
        streaming=False,
    )


def structured_model(model: ChatOpenAI, schema: type[Any]):
    return model.with_structured_output(
        schema,
        method="json_schema",
        strict=True,
    )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowModelConstructionTests -v
```

Expected:

```text
OK
```

Do not commit yet if other Task 1 changes are still pending.

---

### Task 2: Add Prompt Builders

**Files:**
- Create: `app/agents/workflow_agent/prompts.py`
- Test: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Write failing prompt contract tests**

Add tests:

```python
class WorkflowPromptTests(unittest.TestCase):
    def test_inquiry_prompt_requires_structured_questions_and_no_answering(self):
        from app.agents.workflow_agent.prompts import INQUIRY_SYSTEM_PROMPT

        self.assertIn("不要回答用户问题", INQUIRY_SYSTEM_PROMPT)
        self.assertIn("最多 3 个", INQUIRY_SYSTEM_PROMPT)
        self.assertIn("危险信号", INQUIRY_SYSTEM_PROMPT)

    def test_syndrome_prompt_forbids_diagnosis(self):
        from app.agents.workflow_agent.prompts import SYNDROME_SYSTEM_PROMPT

        self.assertIn("possible_patterns", SYNDROME_SYSTEM_PROMPT)
        self.assertIn("不要输出诊断", SYNDROME_SYSTEM_PROMPT)
        self.assertIn("not_enough_for_diagnosis", SYNDROME_SYSTEM_PROMPT)

    def test_answer_prompt_forbids_new_evidence_and_prescriptions(self):
        from app.agents.workflow_agent.prompts import ANSWER_SYSTEM_PROMPT

        self.assertIn("不新增证据", ANSWER_SYSTEM_PROMPT)
        self.assertIn("不新增诊断", ANSWER_SYSTEM_PROMPT)
        self.assertIn("不新增方药剂量", ANSWER_SYSTEM_PROMPT)

    def test_safety_prompt_checks_diagnosis_prescription_dosage_and_risk(self):
        from app.agents.workflow_agent.prompts import SAFETY_SYSTEM_PROMPT

        self.assertIn("contains_diagnosis", SAFETY_SYSTEM_PROMPT)
        self.assertIn("contains_prescription", SAFETY_SYSTEM_PROMPT)
        self.assertIn("contains_dosage", SAFETY_SYSTEM_PROMPT)
        self.assertIn("胸痛", SAFETY_SYSTEM_PROMPT)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowPromptTests -v
```

Expected:

```text
ERROR: ModuleNotFoundError: No module named 'app.agents.workflow_agent.prompts'
```

- [ ] **Step 3: Implement `prompts.py`**

Create `app/agents/workflow_agent/prompts.py`:

```python
from __future__ import annotations

import json
from typing import Any


INQUIRY_SYSTEM_PROMPT = """
你是 InquiryAgent，只负责问诊信息整理，不要回答用户问题。
你的输出必须符合 InquiryState 结构。
判断信息是否足够；如果严重不足，最多 3 个关键澄清问题。
必须识别危险信号：胸痛、呼吸困难、意识异常、剧烈头痛、持续高热、肢体无力、持续加重的腹痛、反复呕吐、黑便、明显出血。
如果出现危险信号，不要因信息不足而暂停检索；应记录 risk_flags，让后续 SafetyAgent 提醒线下就医。
""".strip()

EVIDENCE_SYSTEM_PROMPT = """
你是 EvidenceAgent 的证据整理助手。
EvidenceAgent 是唯一允许调用 retrieve_tcm_knowledge 的组件。
你只整理 EvidenceAgent 已经检索到的 E1-E5 证据，不新增证据来源。
""".strip()

SYNDROME_SYSTEM_PROMPT = """
你是 SyndromeAgent，只输出 possible_patterns，不要输出诊断。
只能基于 InquiryAgent 状态和 EvidenceAgent 的 E1-E5 证据分析可能相关因素。
必须保持 not_enough_for_diagnosis=true。
每个候选方向必须引用 supporting_evidence，例如 E1、E2。
""".strip()

ANSWER_SYSTEM_PROMPT = """
你是 AnswerAgent，只负责组织最终语言。
不新增术语，不新增证据，不新增诊断，不新增方药剂量。
只能整合用户问题、InquiryAgent 状态、EvidenceAgent 证据、SyndromeAgent 候选分析和 SafetyAgent 重写要求。
回答要自然、谨慎、面向用户。
""".strip()

SAFETY_SYSTEM_PROMPT = """
你是 SafetyAgent，只做安全审查。
检查 contains_diagnosis、contains_prescription、contains_dosage。
检查是否有危险信号：胸痛、呼吸困难、意识异常、剧烈头痛、持续高热、肢体无力、持续加重的腹痛、反复呕吐、黑便、明显出血。
如果需要线下就医提醒但答案没有提醒，应设置 rewrite_required=true。
不得替用户下诊断，不得开处方，不得给剂量。
""".strip()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowPromptTests -v
```

Expected:

```text
OK
```

---

### Task 3: Convert InquiryAgent to LLM-Backed Structured Output

**Files:**
- Modify: `app/agents/workflow_agent/workflow.py`
- Modify: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Add fake structured model helpers**

Add test helpers near the top of `tests/test_workflow_agent_flow.py`:

```python
class FakeStructuredRunnable:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.response


class FakeWorkflowModel:
    def __init__(self, responses_by_schema):
        self.responses_by_schema = responses_by_schema
        self.structured_calls = []
        self.runnables = {}

    def with_structured_output(self, schema, *, method="json_schema", strict=True):
        self.structured_calls.append(
            {"schema": schema, "method": method, "strict": strict}
        )
        runnable = FakeStructuredRunnable(self.responses_by_schema[schema])
        self.runnables[schema] = runnable
        return runnable
```

- [ ] **Step 2: Write failing InquiryAgent LLM test**

Add:

```python
class WorkflowLLMAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_inquiry_agent_invokes_model_with_structured_output(self):
        from app.agents.workflow_agent.models import InquiryState, KnownFacts
        from app.agents.workflow_agent.workflow import InquiryAgent

        expected = InquiryState(
            chief_complaint="胃胀",
            known_facts=KnownFacts(duration="两周"),
            missing_info=["诱因或加重缓解因素", "伴随症状"],
            information_sufficiency="insufficient",
            clarification_questions=[
                "进食、油腻、受凉或情绪变化后会加重吗？",
                "是否伴有嗳气、反酸、恶心、腹痛或大便变化？",
            ],
            should_pause_for_clarification=True,
        )
        model = FakeWorkflowModel({InquiryState: expected})

        result = await InquiryAgent(model=model).assess(
            user_text="我最近胃胀两周",
            conversation=[],
        )

        self.assertEqual(result, expected)
        self.assertEqual(model.structured_calls[0]["schema"], InquiryState)
        self.assertEqual(model.structured_calls[0]["method"], "json_schema")
        self.assertTrue(model.structured_calls[0]["strict"])
        self.assertIn("InquiryAgent", model.runnables[InquiryState].calls[0][0]["content"])
```

- [ ] **Step 3: Run test to verify RED**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowLLMAgentTests.test_inquiry_agent_invokes_model_with_structured_output -v
```

Expected:

```text
TypeError: InquiryAgent() takes no arguments
```

- [ ] **Step 4: Implement InquiryAgent LLM call**

Change `InquiryAgent` in `workflow.py` to:

```python
class InquiryAgent:
    def __init__(self, model) -> None:
        self.model = model

    async def assess(
        self,
        user_text: str,
        conversation: Sequence[object] | None = None,
    ) -> InquiryState:
        structured = structured_model(self.model, InquiryState)
        response = await structured.ainvoke(
            [
                {"role": "system", "content": INQUIRY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": compact_json(
                        {
                            "user_text": user_text,
                            "conversation": list(conversation or []),
                        }
                    ),
                },
            ]
        )
        return InquiryState.model_validate(response)
```

Also import:

```python
from app.agents.workflow_agent.llm import structured_model
from app.agents.workflow_agent.prompts import INQUIRY_SYSTEM_PROMPT, compact_json
```

- [ ] **Step 5: Run test to verify GREEN**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowLLMAgentTests.test_inquiry_agent_invokes_model_with_structured_output -v
```

Expected:

```text
OK
```

---

### Task 4: Convert SyndromeAgent, AnswerAgent, and SafetyAgent to LLM-Backed Structured Output

**Files:**
- Modify: `app/agents/workflow_agent/workflow.py`
- Modify: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Write failing tests for LLM calls**

Add tests:

```python
    async def test_syndrome_answer_and_safety_agents_invoke_structured_model(self):
        from app.agents.workflow_agent.models import (
            AnswerDraft,
            EvidenceItem,
            EvidenceResult,
            InquiryState,
            KnownFacts,
            PatternCandidate,
            SafetyReview,
            SyndromeAnalysis,
        )
        from app.agents.workflow_agent.workflow import (
            AnswerAgent,
            SafetyAgent,
            SyndromeAgent,
        )

        inquiry = InquiryState(
            chief_complaint="胃胀",
            known_facts=KnownFacts(duration="两周", triggers=["油腻后加重"]),
            information_sufficiency="sufficient",
        )
        evidence = EvidenceResult(
            retrieval_status="ok",
            evidence=[
                EvidenceItem(
                    id="E1",
                    citation_id="E1",
                    role="syndrome_pattern",
                    text="饮食停滞可见胃脘胀满。",
                    source="《景岳全书》",
                )
            ],
            allowed_terms=["食滞"],
            raw_tool_content="[E1]\n原文：饮食停滞可见胃脘胀满。",
        )
        syndrome = SyndromeAnalysis(
            possible_patterns=[
                PatternCandidate(
                    term="食滞",
                    supporting_evidence=["E1"],
                    confidence="medium",
                    reason="油腻后加重，并有胃胀。",
                )
            ],
            not_enough_for_diagnosis=True,
            need_more_info=["舌象"],
        )
        answer = AnswerDraft(draft_answer="可能与食滞相关。[E1]")
        safety = SafetyReview(final_safety_level="low", rewrite_required=False)
        model = FakeWorkflowModel(
            {
                SyndromeAnalysis: syndrome,
                AnswerDraft: answer,
                SafetyReview: safety,
            }
        )

        syndrome_result = await SyndromeAgent(model=model).analyze(
            user_text="胃胀两周，油腻后加重。",
            inquiry=inquiry,
            evidence=evidence,
        )
        answer_result = await AnswerAgent(model=model).compose(
            user_text="胃胀两周，油腻后加重。",
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome_result,
        )
        safety_result = await SafetyAgent(model=model).review(
            draft_answer=answer_result.draft_answer,
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome_result,
        )

        self.assertEqual(syndrome_result, syndrome)
        self.assertEqual(answer_result, answer)
        self.assertEqual(safety_result, safety)
        self.assertEqual(
            [call["schema"] for call in model.structured_calls],
            [SyndromeAnalysis, AnswerDraft, SafetyReview],
        )
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowLLMAgentTests.test_syndrome_answer_and_safety_agents_invoke_structured_model -v
```

Expected:

```text
TypeError: SyndromeAgent() takes no arguments
```

- [ ] **Step 3: Implement LLM-backed agents**

Update constructors:

```python
class SyndromeAgent:
    def __init__(self, model) -> None:
        self.model = model
```

```python
class AnswerAgent:
    def __init__(self, model) -> None:
        self.model = model
```

```python
class SafetyAgent:
    def __init__(self, model) -> None:
        self.model = model
```

Use official structured output for each:

```python
structured = structured_model(self.model, SyndromeAnalysis)
response = await structured.ainvoke([...])
return SyndromeAnalysis.model_validate(response)
```

```python
structured = structured_model(self.model, AnswerDraft)
response = await structured.ainvoke([...])
return AnswerDraft.model_validate(response)
```

```python
structured = structured_model(self.model, SafetyReview)
response = await structured.ainvoke([...])
return SafetyReview.model_validate(response)
```

Each user message must include only structured JSON from known inputs, via `compact_json(...)`. Do not parse model text manually.

- [ ] **Step 4: Run test to verify GREEN**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowLLMAgentTests.test_syndrome_answer_and_safety_agents_invoke_structured_model -v
```

Expected:

```text
OK
```

---

### Task 5: Keep EvidenceAgent as the Only Retrieval Tool Owner

**Files:**
- Modify: `app/agents/workflow_agent/workflow.py`
- Modify: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Write failing evidence ownership test**

Add:

```python
    async def test_only_evidence_agent_calls_retrieval_in_llm_workflow(self):
        from app.agents.workflow_agent.models import (
            AnswerDraft,
            EvidenceResult,
            InquiryState,
            KnownFacts,
            SafetyReview,
            SyndromeAnalysis,
        )
        from app.agents.workflow_agent.workflow import EvidenceAgent, TCMWorkflow

        retrieval_calls = []

        async def fake_retriever(query: str, mode: str) -> str:
            retrieval_calls.append({"query": query, "mode": mode})
            return (
                "检索状态：ok\n"
                "检索模式：hybrid_parent\n\n"
                "[E1]\n"
                "证据角色：syndrome_pattern\n"
                "原文：饮食停滞可见胃脘胀满。\n"
                "来源：《景岳全书》 胃脘\n\n"
                "允许使用的专业术语：\n"
                "- 食滞\n\n"
                "回答约束：\n"
                "- 不得据此推荐方剂、药物、剂量或煎服法。"
            )

        model = FakeWorkflowModel(
            {
                InquiryState: InquiryState(
                    chief_complaint="胃胀",
                    known_facts=KnownFacts(
                        duration="两周",
                        triggers=["油腻后加重"],
                        associated_symptoms=["嗳气"],
                    ),
                    information_sufficiency="sufficient",
                ),
                SyndromeAnalysis: SyndromeAnalysis(
                    possible_patterns=[],
                    not_enough_for_diagnosis=True,
                    need_more_info=["舌象"],
                ),
                AnswerDraft: AnswerDraft(draft_answer="信息仍不足，建议补充舌象。"),
                SafetyReview: SafetyReview(final_safety_level="low", rewrite_required=False),
            }
        )

        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
        )

        await workflow.run("胃胀两周，油腻后加重，嗳气。")

        self.assertEqual(len(retrieval_calls), 1)
        self.assertEqual(retrieval_calls[0]["mode"], "hybrid")
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowLLMAgentTests.test_only_evidence_agent_calls_retrieval_in_llm_workflow -v
```

Expected:

```text
TypeError: TCMWorkflow.__init__() got an unexpected keyword argument 'model'
```

- [ ] **Step 3: Implement model injection in `TCMWorkflow`**

Update:

```python
class TCMWorkflow:
    def __init__(
        self,
        *,
        model,
        inquiry_agent: InquiryAgent | None = None,
        evidence_agent: EvidenceAgent | None = None,
        syndrome_agent: SyndromeAgent | None = None,
        answer_agent: AnswerAgent | None = None,
        safety_agent: SafetyAgent | None = None,
    ) -> None:
        self.model = model
        self.inquiry_agent = inquiry_agent or InquiryAgent(model=model)
        self.evidence_agent = evidence_agent or EvidenceAgent()
        self.syndrome_agent = syndrome_agent or SyndromeAgent(model=model)
        self.answer_agent = answer_agent or AnswerAgent(model=model)
        self.safety_agent = safety_agent or SafetyAgent(model=model)
```

Update all async calls:

```python
inquiry = await self.inquiry_agent.assess(...)
syndrome = await self.syndrome_agent.analyze(...)
answer = await self.answer_agent.compose(...)
safety = await self.safety_agent.review(...)
```

- [ ] **Step 4: Run test to verify GREEN**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowLLMAgentTests.test_only_evidence_agent_calls_retrieval_in_llm_workflow -v
```

Expected:

```text
OK
```

---

### Task 6: Wire Runtime Context into WorkflowAgent

**Files:**
- Modify: `app/agents/workflow_agent/agent.py`
- Modify: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Write failing factory/context test**

Add:

```python
class WorkflowRuntimeLLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_make_workflow_agent_builds_model_from_context(self):
        from app.agents.workflow_agent import agent as workflow_agent_module

        with patch.object(workflow_agent_module, "build_workflow_model") as build_model:
            build_model.return_value = FakeWorkflowModel({})
            created = workflow_agent_module.make_workflow_agent(
                {"model_name": "workflow-model", "temperature": 0.05}
            )

        self.assertIsInstance(created, workflow_agent_module.WorkflowAgent)
        build_model.assert_called_once_with(
            {"model_name": "workflow-model", "temperature": 0.05}
        )
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowRuntimeLLMTests.test_make_workflow_agent_builds_model_from_context -v
```

Expected:

```text
AssertionError: Expected 'build_workflow_model' to be called once. Called 0 times.
```

- [ ] **Step 3: Update `agent.py`**

Change imports:

```python
from app.agents.workflow_agent.llm import build_workflow_model
```

Change factory:

```python
def make_workflow_agent(context: dict[str, Any] | None = None) -> WorkflowAgent:
    model = build_workflow_model(context or {})
    return WorkflowAgent(workflow=TCMWorkflow(model=model))
```

Keep test-injection path:

```python
class WorkflowAgent:
    def __init__(
        self,
        *,
        workflow: TCMWorkflow | None = None,
        thread_store: Any | None = None,
    ) -> None:
        self.workflow = workflow
        self.thread_store = thread_store or runtime_state.state.thread_store
```

If `workflow` is `None`, build it only in `make_workflow_agent`; do not silently create a model-less workflow in the constructor.

- [ ] **Step 4: Run test to verify GREEN**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowRuntimeLLMTests.test_make_workflow_agent_builds_model_from_context -v
```

Expected:

```text
OK
```

---

### Task 7: Preserve Safety Rewrite and Final Safety Re-Check

**Files:**
- Modify: `app/agents/workflow_agent/workflow.py`
- Modify: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Write failing rewrite loop test**

Add:

```python
    async def test_unsafe_answer_triggers_llm_rewrite_and_second_safety_review(self):
        from app.agents.workflow_agent.models import (
            AnswerDraft,
            EvidenceResult,
            InquiryState,
            KnownFacts,
            SafetyReview,
            SyndromeAnalysis,
        )
        from app.agents.workflow_agent.workflow import EvidenceAgent, TCMWorkflow

        class SequenceModel(FakeWorkflowModel):
            def __init__(self):
                self.structured_calls = []
                self.runnables = {}
                self.responses = {
                    InquiryState: [
                        InquiryState(
                            chief_complaint="胃胀",
                            known_facts=KnownFacts(duration="两周", triggers=["油腻后加重"]),
                            information_sufficiency="sufficient",
                        )
                    ],
                    SyndromeAnalysis: [
                        SyndromeAnalysis(not_enough_for_diagnosis=True, need_more_info=["舌象"])
                    ],
                    AnswerDraft: [
                        AnswerDraft(draft_answer="你这是脾胃虚弱证，可以用某某汤。"),
                        AnswerDraft(draft_answer="目前只能谨慎分析，不能替代线下辨证。"),
                    ],
                    SafetyReview: [
                        SafetyReview(
                            contains_diagnosis=True,
                            contains_prescription=True,
                            final_safety_level="high",
                            rewrite_required=True,
                            rewrite_instructions=["删除直接诊断表达。", "删除方药表达。"],
                        ),
                        SafetyReview(final_safety_level="low", rewrite_required=False),
                    ],
                }

            def with_structured_output(self, schema, *, method="json_schema", strict=True):
                self.structured_calls.append({"schema": schema, "method": method, "strict": strict})
                response = self.responses[schema].pop(0)
                runnable = FakeStructuredRunnable(response)
                self.runnables.setdefault(schema, []).append(runnable)
                return runnable

        async def fake_retriever(query: str, mode: str) -> str:
            return "检索状态：ok\n检索模式：hybrid_parent"

        model = SequenceModel()
        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
        )

        result = await workflow.run("胃胀两周，油腻后加重。")

        safety_events = [
            event for event in result.agent_trace if event.get("agent") == "SafetyAgent"
        ]
        answer_events = [
            event for event in result.agent_trace if event.get("agent") == "AnswerAgent"
        ]

        self.assertEqual([event["stage"] for event in answer_events], ["draft", "rewrite"])
        self.assertEqual([event["rewrite_required"] for event in safety_events], [True, False])
        self.assertEqual(result.final_text, "目前只能谨慎分析，不能替代线下辨证。")
```

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowLLMAgentTests.test_unsafe_answer_triggers_llm_rewrite_and_second_safety_review -v
```

Expected:

```text
FAIL
```

The current deterministic flow will not produce the exact LLM call sequence.

- [ ] **Step 3: Implement rewrite loop**

Ensure orchestration keeps:

```python
answer = await self.answer_agent.compose(...)
agent_trace.append({"agent": "AnswerAgent", "stage": "draft"})

safety = await self.safety_agent.review(...)
agent_trace.append({
    "agent": "SafetyAgent",
    "stage": "initial",
    "final_safety_level": safety.final_safety_level,
    "rewrite_required": safety.rewrite_required,
})

if safety.rewrite_required:
    answer = await self.answer_agent.compose(..., safety_review=safety)
    agent_trace.append({"agent": "AnswerAgent", "stage": "rewrite"})
    safety = await self.safety_agent.review(...)
    agent_trace.append({
        "agent": "SafetyAgent",
        "stage": "rewrite",
        "final_safety_level": safety.final_safety_level,
        "rewrite_required": safety.rewrite_required,
    })
```

Keep a safe fallback only if the second review still fails.

- [ ] **Step 4: Run test to verify GREEN**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowLLMAgentTests.test_unsafe_answer_triggers_llm_rewrite_and_second_safety_review -v
```

Expected:

```text
OK
```

---

### Task 8: Update Registry and Runtime Smoke Tests

**Files:**
- Modify: `tests/test_workflow_agent_registry.py`
- Modify: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Keep registry dependency isolation tests green**

Run:

```powershell
python -m unittest tests.test_workflow_agent_registry -v
```

Expected:

```text
OK
```

If it fails because `workflow_agent` imports `lead_agent` or RAG dependencies during registry resolution, fix lazy imports in `app/agents/registry.py` before continuing.

- [ ] **Step 2: Update runtime smoke test to inject fake workflow**

Keep existing runtime smoke tests using:

```python
agent_factory=lambda context: WorkflowAgent(
    workflow=workflow,
    thread_store=thread_store,
)
```

Do not allow these tests to instantiate real `ChatOpenAI`.

- [ ] **Step 3: Run focused workflow runtime tests**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_flow.WorkflowRuntimeTests tests.test_workflow_agent_registry -v
```

Expected:

```text
OK
```

---

### Task 9: Final Verification

**Files:**
- Modify only if verification exposes a real issue.

- [ ] **Step 1: Compile changed Python files**

Run:

```powershell
python -m py_compile app/agents/workflow_agent/llm.py app/agents/workflow_agent/prompts.py app/agents/workflow_agent/models.py app/agents/workflow_agent/workflow.py app/agents/workflow_agent/agent.py app/agents/registry.py app/runtime/runs/worker.py
```

Expected:

```text
exit code 0
```

- [ ] **Step 2: Run focused test suite**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_models tests.test_workflow_agent_flow tests.test_workflow_agent_registry tests.test_async_agent_factory -v
```

Expected:

```text
OK
```

- [ ] **Step 3: Run broader compatibility suite and record known baseline failures**

Run:

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m unittest tests.test_workflow_agent_models tests.test_workflow_agent_flow tests.test_workflow_agent_registry tests.test_async_agent_factory tests.test_clarification_flow tests.test_lead_agent_factory -v
```

Expected:

```text
The workflow-related tests pass.
Existing baseline failures may remain:
- tests.test_clarification_flow: stored_thread.values contains messages
- tests.test_lead_agent_factory: state_schema KeyError
```

If any new workflow-related failure appears, fix it before committing.

- [ ] **Step 4: Inspect staged diff**

Run:

```powershell
git diff --cached --name-status
git diff --name-status
```

Expected feature paths only:

```text
app/agents/workflow_agent/
app/agents/registry.py
app/runtime/runs/worker.py
tests/test_workflow_agent_*.py
docs/superpowers/plans/2026-06-24-workflow-agent-llm-backed.md
```

Do not include `.env` or unrelated `.gitignore` changes in the feature commit.

- [ ] **Step 5: Commit only after all previous checks pass**

Run:

```powershell
git add app/agents/workflow_agent app/agents/registry.py app/runtime/runs/worker.py tests/test_workflow_agent_models.py tests/test_workflow_agent_flow.py tests/test_workflow_agent_registry.py docs/superpowers/plans/2026-06-24-workflow-agent-llm-backed.md
git commit -m "feat: make workflow agent llm backed"
```

Expected:

```text
Commit succeeds only after deterministic workflow internals have been replaced by LLM-backed agents.
```

---

## Self-Review

Spec coverage:

- LLM-backed agents: Tasks 1, 3, 4, 6, and 7.
- Official LangChain format: Task 1 uses `ChatOpenAI` and `with_structured_output(schema, method="json_schema", strict=True)`; references include `create_agent` for context but this plan uses direct structured model calls because the workflow is fixed and EvidenceAgent must remain the only retrieval owner.
- EvidenceAgent-only retrieval: Task 5.
- SafetyAgent independent review and final re-check: Task 7.
- Runtime adapter and registry preservation: Tasks 6 and 8.
- No old deterministic code committed as-is: Task 0 and Task 9.

Placeholder scan:

- No TBD/TODO placeholders are present.
- Every task has concrete files, commands, expected outcomes, and code snippets.

Type consistency:

- `InquiryState`, `EvidenceResult`, `SyndromeAnalysis`, `AnswerDraft`, and `SafetyReview` are consistently used as Pydantic structured output schemas.
- `FakeWorkflowModel.with_structured_output(...)` mirrors the local installed signature subset needed by this implementation.
