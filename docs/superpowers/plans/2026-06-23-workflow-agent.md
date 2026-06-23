# Workflow Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `workflow_agent` assistant that runs a fixed Inquiry -> Evidence -> Syndrome -> Answer -> Safety pipeline while leaving the existing `lead_agent` path unchanged.

**Architecture:** Implement the workflow as small deterministic Python agent components with Pydantic contracts and injected dependencies for retrieval. Wrap the workflow in a minimal async agent adapter that satisfies the current `run_agent` runtime interface (`astream`, `aget_state`, `aupdate_state`) so SSE, conversation storage, clarification handling, guardrails, and trace extraction keep working.

**Tech Stack:** Python 3, `unittest`, Pydantic v2, LangChain Core messages, existing `retrieve_tcm_knowledge`, existing `run_agent` runtime and thread store.

---

## File Structure

Create these files:

- `app/agents/workflow_agent/__init__.py`: public exports for the new assistant package.
- `app/agents/workflow_agent/models.py`: Pydantic data contracts shared by fixed workflow agents.
- `app/agents/workflow_agent/workflow.py`: InquiryAgent, EvidenceAgent, SyndromeAgent, AnswerAgent, SafetyAgent, and TCMWorkflow orchestration.
- `app/agents/workflow_agent/agent.py`: runtime-compatible WorkflowAgent adapter and `make_workflow_agent` factory.
- `tests/test_workflow_agent_models.py`: model validation and safety contract tests.
- `tests/test_workflow_agent_flow.py`: fixed workflow behavior and runtime smoke tests.
- `tests/test_workflow_agent_registry.py`: assistant registry tests.

Modify these files:

- `app/agents/registry.py`: register `assistant_id="workflow_agent"` without changing `lead_agent`.
- `app/runtime/runs/worker.py`: preserve `workflow_trace` emitted by the adapter when saving `last_agent_trace`.

Do not modify:

- `app/agents/lead_agent/*`
- `.env`
- `.gitignore`
- SSE router payload shape

---

### Task 1: Workflow Agent Data Contracts

**Files:**
- Create: `app/agents/workflow_agent/__init__.py`
- Create: `app/agents/workflow_agent/models.py`
- Test: `tests/test_workflow_agent_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/test_workflow_agent_models.py`:

```python
import unittest
from pydantic import ValidationError

from app.agents.workflow_agent.models import (
    AnswerDraft,
    EvidenceItem,
    EvidenceResult,
    InquiryState,
    KnownFacts,
    PatternCandidate,
    SafetyReview,
    SyndromeAnalysis,
    filter_allowed_patterns,
)


class WorkflowAgentModelTests(unittest.TestCase):
    def test_inquiry_state_caps_and_normalizes_clarification_questions(self):
        state = InquiryState(
            chief_complaint="胃胀",
            known_facts=KnownFacts(triggers=["油腻后加重"]),
            missing_info=["持续时间", "大便情况", "反酸烧心", "食欲"],
            information_sufficiency="insufficient",
            clarification_questions=[
                "胃胀持续多久了？",
                "大便情况如何？",
                "是否伴有反酸、烧心或腹痛？",
                "食欲怎么样？",
            ],
            should_pause_for_clarification=True,
        )

        self.assertEqual(
            state.clarification_questions,
            [
                "胃胀持续多久了？",
                "大便情况如何？",
                "是否伴有反酸、烧心或腹痛？",
            ],
        )

    def test_inquiry_pause_requires_questions(self):
        with self.assertRaisesRegex(ValidationError, "clarification"):
            InquiryState(
                chief_complaint="胃胀",
                information_sufficiency="insufficient",
                clarification_questions=[],
                should_pause_for_clarification=True,
            )

    def test_completed_syndrome_analysis_filters_unauthorized_terms(self):
        analysis = SyndromeAnalysis(
            possible_patterns=[
                PatternCandidate(
                    term="食滞",
                    supporting_evidence=["E1"],
                    confidence="medium",
                    reason="油腻后胃胀加重，并伴有嗳气。",
                ),
                PatternCandidate(
                    term="脾胃虚弱",
                    supporting_evidence=["E2"],
                    confidence="low",
                    reason="该术语不在本次证据允许范围内。",
                ),
            ],
            not_enough_for_diagnosis=True,
            need_more_info=["舌象", "大便"],
        )

        filtered = filter_allowed_patterns(analysis, ["食滞"])

        self.assertEqual(len(filtered.possible_patterns), 1)
        self.assertEqual(filtered.possible_patterns[0].term, "食滞")

    def test_safety_review_marks_rewrite_required_for_unsafe_content(self):
        review = SafetyReview(
            has_risk_flags=True,
            risk_flags=["胸痛"],
            contains_diagnosis=True,
            contains_prescription=True,
            contains_dosage=True,
            needs_offline_medical_advice=True,
            final_safety_level="high",
            rewrite_required=True,
            rewrite_instructions=[
                "删除直接诊断表达。",
                "删除方药和剂量表达。",
                "加入线下就医提醒。",
            ],
        )

        self.assertTrue(review.rewrite_required)
        self.assertEqual(review.final_safety_level, "high")

    def test_evidence_result_keeps_raw_tool_content_for_guardrails(self):
        result = EvidenceResult(
            retrieval_status="ok",
            retrieval_mode="hybrid_parent",
            degraded=False,
            evidence=[
                EvidenceItem(
                    id="E1",
                    citation_id="E1",
                    role="syndrome_pattern",
                    text="因食而胀。",
                    source="《景岳全书》 卷一 / 胃脘",
                )
            ],
            allowed_terms=["食滞"],
            raw_tool_content="允许使用的专业术语：\n- 食滞",
        )

        self.assertEqual(result.evidence[0].id, "E1")
        self.assertIn("食滞", result.raw_tool_content)

    def test_answer_draft_strips_outer_whitespace(self):
        draft = AnswerDraft(draft_answer="  当前只能做谨慎分析。  ")

        self.assertEqual(draft.draft_answer, "当前只能做谨慎分析。")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_workflow_agent_models -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.agents.workflow_agent'`.

- [ ] **Step 3: Create package exports**

Create `app/agents/workflow_agent/__init__.py`:

```python
__all__: list[str] = []
```

- [ ] **Step 4: Implement workflow data contracts**

Create `app/agents/workflow_agent/models.py`:

```python
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.middlewares.clarification_controller import normalize_question_items


InformationSufficiency = Literal["sufficient", "insufficient"]
Confidence = Literal["low", "medium", "high"]
SafetyLevel = Literal["low", "medium", "high"]


class KnownFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: str = ""
    triggers: list[str] = Field(default_factory=list)
    associated_symptoms: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class InquiryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chief_complaint: str = ""
    known_facts: KnownFacts = Field(default_factory=KnownFacts)
    missing_info: list[str] = Field(default_factory=list)
    information_sufficiency: InformationSufficiency = "insufficient"
    clarification_questions: list[str] = Field(default_factory=list)
    should_pause_for_clarification: bool = False

    @field_validator("chief_complaint")
    @classmethod
    def normalize_chief_complaint(cls, value: str) -> str:
        return value.strip()

    @field_validator("missing_info")
    @classmethod
    def normalize_missing_info(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=max(len(value), 1))

    @field_validator("clarification_questions")
    @classmethod
    def normalize_questions(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=3)

    @model_validator(mode="after")
    def validate_pause_contract(self) -> "InquiryState":
        if self.should_pause_for_clarification and not self.clarification_questions:
            raise ValueError("clarification questions are required when pausing")
        if (
            self.information_sufficiency == "sufficient"
            and self.should_pause_for_clarification
        ):
            raise ValueError("sufficient inquiry state cannot pause for clarification")
        return self


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    citation_id: str
    role: str = ""
    text: str
    source: str = ""

    @field_validator("id", "citation_id", "role", "text", "source")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class EvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval_status: str = "insufficient_evidence"
    retrieval_mode: str = "hybrid_parent"
    degraded: bool = False
    evidence: list[EvidenceItem] = Field(default_factory=list)
    allowed_terms: list[str] = Field(default_factory=list)
    raw_tool_content: str = ""

    @field_validator("allowed_terms")
    @classmethod
    def normalize_allowed_terms(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=max(len(value), 1))


class PatternCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"
    reason: str

    @field_validator("term", "reason")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class SyndromeAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    possible_patterns: list[PatternCandidate] = Field(default_factory=list)
    not_enough_for_diagnosis: bool = True
    need_more_info: list[str] = Field(default_factory=list)

    @field_validator("need_more_info")
    @classmethod
    def normalize_need_more_info(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=max(len(value), 1))


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_answer: str

    @field_validator("draft_answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        return value.strip()


class SafetyReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_risk_flags: bool = False
    risk_flags: list[str] = Field(default_factory=list)
    contains_diagnosis: bool = False
    contains_prescription: bool = False
    contains_dosage: bool = False
    needs_offline_medical_advice: bool = False
    final_safety_level: SafetyLevel = "low"
    rewrite_required: bool = False
    rewrite_instructions: list[str] = Field(default_factory=list)

    @field_validator("risk_flags", "rewrite_instructions")
    @classmethod
    def normalize_list(cls, value: list[str]) -> list[str]:
        return normalize_question_items(value, max_questions=max(len(value), 1))


def filter_allowed_patterns(
    analysis: SyndromeAnalysis,
    allowed_terms: list[str],
) -> SyndromeAnalysis:
    allowed = set(allowed_terms)
    return SyndromeAnalysis(
        possible_patterns=[
            pattern
            for pattern in analysis.possible_patterns
            if pattern.term in allowed
        ],
        not_enough_for_diagnosis=analysis.not_enough_for_diagnosis,
        need_more_info=analysis.need_more_info,
    )
```

- [ ] **Step 5: Run tests to verify Task 1 passes**

Run:

```powershell
python -m unittest tests.test_workflow_agent_models -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

Run:

```powershell
git add app/agents/workflow_agent/__init__.py app/agents/workflow_agent/models.py tests/test_workflow_agent_models.py
git commit -m "feat: add workflow agent contracts"
```

---

### Task 2: Inquiry, Answer, and Safety Agents

**Files:**
- Modify: `app/agents/workflow_agent/workflow.py`
- Test: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Write failing tests for inquiry, answer, and safety behavior**

Create `tests/test_workflow_agent_flow.py` with these initial tests:

```python
import unittest

from app.agents.workflow_agent.models import (
    EvidenceItem,
    EvidenceResult,
    InquiryState,
    KnownFacts,
    PatternCandidate,
    SyndromeAnalysis,
)
from app.agents.workflow_agent.workflow import (
    AnswerAgent,
    InquiryAgent,
    SafetyAgent,
)


class WorkflowFixedAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_inquiry_agent_pauses_when_key_information_is_missing(self):
        state = InquiryAgent().assess("我最近胃胀")

        self.assertEqual(state.information_sufficiency, "insufficient")
        self.assertTrue(state.should_pause_for_clarification)
        self.assertLessEqual(len(state.clarification_questions), 3)
        self.assertIn("胃胀", state.chief_complaint)

    def test_inquiry_agent_records_risk_flags(self):
        state = InquiryAgent().assess("我胃胀，还胸痛、呼吸困难")

        self.assertIn("胸痛", state.known_facts.risk_flags)
        self.assertIn("呼吸困难", state.known_facts.risk_flags)

    def test_answer_agent_uses_only_supplied_terms_and_evidence(self):
        inquiry = InquiryState(
            chief_complaint="胃胀",
            known_facts=KnownFacts(
                duration="两周",
                triggers=["油腻后加重"],
                associated_symptoms=["嗳气"],
            ),
            missing_info=["舌象"],
            information_sufficiency="sufficient",
        )
        evidence = EvidenceResult(
            retrieval_status="ok",
            retrieval_mode="hybrid_parent",
            degraded=False,
            evidence=[
                EvidenceItem(
                    id="E1",
                    citation_id="E1",
                    role="syndrome_pattern",
                    text="饮食停滞可见胃脘胀满。",
                    source="《景岳全书》 胃脘",
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
                    reason="油腻后胃胀加重，并伴有嗳气。",
                )
            ],
            not_enough_for_diagnosis=True,
            need_more_info=["舌象"],
        )

        answer = AnswerAgent().compose(
            user_text="胃胀两周，油腻后加重，嗳气。",
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome,
        )

        self.assertIn("食滞", answer.draft_answer)
        self.assertIn("[E1]", answer.draft_answer)
        self.assertNotIn("脾胃虚弱", answer.draft_answer)
        self.assertNotIn("处方", answer.draft_answer)

    def test_safety_agent_detects_diagnosis_prescription_dosage_and_risk(self):
        inquiry = InquiryState(
            chief_complaint="胃胀",
            known_facts=KnownFacts(risk_flags=["胸痛"]),
            information_sufficiency="sufficient",
        )

        review = SafetyAgent().review(
            draft_answer="你这是脾胃虚弱证，可以用某某汤，每次10克。",
            inquiry=inquiry,
            evidence=EvidenceResult(),
            syndrome=SyndromeAnalysis(),
        )

        self.assertTrue(review.contains_diagnosis)
        self.assertTrue(review.contains_prescription)
        self.assertTrue(review.contains_dosage)
        self.assertTrue(review.needs_offline_medical_advice)
        self.assertEqual(review.final_safety_level, "high")
        self.assertTrue(review.rewrite_required)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.agents.workflow_agent.workflow'`.

- [ ] **Step 3: Implement deterministic InquiryAgent, AnswerAgent, and SafetyAgent**

Create `app/agents/workflow_agent/workflow.py` with this initial content:

```python
import re
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.agents.workflow_agent.models import (
    AnswerDraft,
    EvidenceResult,
    InquiryState,
    KnownFacts,
    PatternCandidate,
    SafetyReview,
    SyndromeAnalysis,
)


RISK_FLAG_TERMS = [
    "胸痛",
    "呼吸困难",
    "意识异常",
    "剧烈头痛",
    "持续高热",
    "肢体无力",
    "持续加重的腹痛",
    "反复呕吐",
    "黑便",
    "明显出血",
]

SYMPTOM_HINTS = [
    "胃胀",
    "腹痛",
    "胃痛",
    "嗳气",
    "反酸",
    "烧心",
    "头痛",
    "眩晕",
    "咳嗽",
    "失眠",
    "心悸",
    "便秘",
    "泄泻",
]

DURATION_PATTERN = re.compile(r"(持续)?[一二三四五六七八九十半两0-9]+(天|日|周|个月|月|年|小时)")
DOSAGE_PATTERN = re.compile(r"\d+\s*(克|g|毫克|mg|片|粒|袋|剂|次)")
DIAGNOSIS_PATTERNS = [
    re.compile(r"你(这|就是|属于|是).{0,12}(证|病)"),
    re.compile(r"(诊断为|确诊为|可以诊断)"),
]
PRESCRIPTION_TERMS = [
    "处方",
    "方剂",
    "某某汤",
    "汤剂",
    "中成药",
    "煎服",
    "每日一剂",
]


def _contains_any(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term in text]


def _extract_duration(text: str) -> str:
    match = DURATION_PATTERN.search(text)
    return match.group(0) if match else ""


class InquiryAgent:
    def assess(
        self,
        user_text: str,
        conversation: list[dict[str, str]] | None = None,
    ) -> InquiryState:
        text = user_text.strip()
        risk_flags = _contains_any(text, RISK_FLAG_TERMS)
        duration = _extract_duration(text)
        triggers = []
        if "油腻" in text:
            triggers.append("油腻后加重")
        if "饭后" in text or "进食后" in text:
            triggers.append("进食后相关")

        associated_symptoms = [
            symptom
            for symptom in ["嗳气", "反酸", "烧心", "腹痛", "呕吐", "发热", "黑便"]
            if symptom in text
        ]
        chief = next((symptom for symptom in SYMPTOM_HINTS if symptom in text), "")
        if not chief:
            chief = text[:20]

        missing_info: list[str] = []
        questions: list[str] = []
        if not duration:
            missing_info.append("持续时间")
            questions.append(f"{chief or '这种情况'}持续多久了？")
        if "大便" not in text and "便秘" not in text and "泄泻" not in text and "黑便" not in text:
            missing_info.append("大便情况")
            questions.append("大便情况如何，是否有便秘、腹泻或黑便？")
        if "食欲" not in text and "饭后" not in text and "油腻" not in text:
            missing_info.append("饮食和食欲情况")
            questions.append("食欲和进食后的变化如何？")
        if "反酸" not in text and "烧心" not in text:
            missing_info.append("反酸烧心情况")
            questions.append("是否伴有反酸、烧心或腹痛？")

        should_pause = len(missing_info) >= 2 and not risk_flags
        sufficiency = "insufficient" if should_pause else "sufficient"

        return InquiryState(
            chief_complaint=chief,
            known_facts=KnownFacts(
                duration=duration,
                triggers=triggers,
                associated_symptoms=associated_symptoms,
                risk_flags=risk_flags,
            ),
            missing_info=missing_info,
            information_sufficiency=sufficiency,
            clarification_questions=questions[:3] if should_pause else [],
            should_pause_for_clarification=should_pause,
        )


class AnswerAgent:
    def compose(
        self,
        *,
        user_text: str,
        inquiry: InquiryState,
        evidence: EvidenceResult,
        syndrome: SyndromeAnalysis,
        safety_review: SafetyReview | None = None,
    ) -> AnswerDraft:
        lines: list[str] = []
        chief = inquiry.chief_complaint or "你描述的情况"
        lines.append(f"从你描述的{chief}来看，目前只能做健康咨询层面的谨慎分析，不能替代面诊诊断。")

        if evidence.retrieval_status != "ok" or not evidence.evidence:
            lines.append("目前检索依据有限，暂不宜据此判断具体证候。")
        elif syndrome.possible_patterns:
            pattern_parts = []
            for pattern in syndrome.possible_patterns:
                citations = "、".join(pattern.supporting_evidence)
                pattern_parts.append(f"{pattern.term}（{citations}）")
            lines.append(
                "结合本次检索证据，可先从"
                + "、".join(pattern_parts)
                + "等可能相关因素理解，但这不等同于诊断。"
            )
        else:
            lines.append("本次证据不足以支持明确的候选辨证方向。")

        if evidence.evidence:
            citation_ids = "、".join(item.id for item in evidence.evidence[:5])
            lines.append(f"可参考的检索证据包括 {citation_ids}。")

        if inquiry.missing_info or syndrome.need_more_info:
            missing = list(dict.fromkeys([*inquiry.missing_info, *syndrome.need_more_info]))
            lines.append("后续可继续补充：" + "、".join(missing[:5]) + "。")

        lines.append("日常上可先注意清淡饮食，避免油腻、过饱和明显诱发因素。")

        if inquiry.known_facts.risk_flags:
            lines.append("你提到的危险信号需要重视，建议及时线下就医评估。")
        else:
            lines.append("如果出现持续加重的腹痛、反复呕吐、黑便、发热、胸痛或呼吸困难，应及时线下就医。")

        if safety_review and safety_review.rewrite_required:
            lines = [
                line
                for line in lines
                if not any(term in line for term in PRESCRIPTION_TERMS)
            ]

        return AnswerDraft(draft_answer="\n".join(lines))

    def safe_fallback(self, inquiry: InquiryState) -> AnswerDraft:
        risk_text = ""
        if inquiry.known_facts.risk_flags:
            risk_text = "你提到的危险信号需要重视，建议及时线下就医评估。"
        return AnswerDraft(
            draft_answer=(
                "目前信息和安全审查结果不足以支持继续展开中医辨证分析。"
                "我不能替代医生面诊，也不能提供处方、药物或剂量建议。"
                f"{risk_text}"
            )
        )


class SafetyAgent:
    def review(
        self,
        draft_answer: str,
        *,
        inquiry: InquiryState,
        evidence: EvidenceResult,
        syndrome: SyndromeAnalysis,
    ) -> SafetyReview:
        contains_diagnosis = any(
            pattern.search(draft_answer) for pattern in DIAGNOSIS_PATTERNS
        )
        contains_prescription = bool(_contains_any(draft_answer, PRESCRIPTION_TERMS))
        contains_dosage = bool(DOSAGE_PATTERN.search(draft_answer))
        risk_flags = list(inquiry.known_facts.risk_flags)
        needs_offline = bool(risk_flags)
        rewrite_required = (
            contains_diagnosis
            or contains_prescription
            or contains_dosage
            or (needs_offline and "就医" not in draft_answer)
        )

        instructions: list[str] = []
        if contains_diagnosis:
            instructions.append("删除直接诊断表达，改为可能相关因素。")
        if contains_prescription or contains_dosage:
            instructions.append("删除方药、药物、剂量和煎服法表达。")
        if needs_offline and "就医" not in draft_answer:
            instructions.append("加入线下就医提醒。")

        if contains_prescription or contains_dosage or contains_diagnosis or risk_flags:
            level = "high" if risk_flags or contains_dosage else "medium"
        else:
            level = "low"

        return SafetyReview(
            has_risk_flags=bool(risk_flags),
            risk_flags=risk_flags,
            contains_diagnosis=contains_diagnosis,
            contains_prescription=contains_prescription,
            contains_dosage=contains_dosage,
            needs_offline_medical_advice=needs_offline,
            final_safety_level=level,
            rewrite_required=rewrite_required,
            rewrite_instructions=instructions,
        )
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow -v
```

Expected: PASS for the four tests in this file.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add app/agents/workflow_agent/workflow.py tests/test_workflow_agent_flow.py
git commit -m "feat: add workflow inquiry answer safety agents"
```

---

### Task 3: EvidenceAgent and SyndromeAgent

**Files:**
- Modify: `app/agents/workflow_agent/workflow.py`
- Modify: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Add failing evidence and syndrome tests**

Append these tests inside `WorkflowFixedAgentTests` in `tests/test_workflow_agent_flow.py`:

```python
    async def test_evidence_agent_is_only_component_that_calls_retrieval(self):
        calls = []

        async def fake_retriever(query: str, mode: str) -> str:
            calls.append({"query": query, "mode": mode})
            return (
                "检索状态：ok\n\n"
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

        inquiry = InquiryState(
            chief_complaint="胃胀",
            known_facts=KnownFacts(duration="两周", triggers=["油腻后加重"]),
            information_sufficiency="sufficient",
        )

        result = await EvidenceAgent(retriever=fake_retriever).retrieve(
            user_text="胃胀两周，油腻后加重。",
            inquiry=inquiry,
        )

        self.assertEqual(calls[0]["mode"], "hybrid")
        self.assertIn("胃胀", calls[0]["query"])
        self.assertEqual(result.evidence[0].id, "E1")
        self.assertEqual(result.allowed_terms, ["食滞"])
        self.assertIn("允许使用的专业术语", result.raw_tool_content)

    def test_syndrome_agent_uses_only_allowed_terms_and_evidence_ids(self):
        evidence = EvidenceResult(
            retrieval_status="ok",
            retrieval_mode="hybrid_parent",
            degraded=False,
            evidence=[
                EvidenceItem(
                    id="E1",
                    citation_id="E1",
                    role="syndrome_pattern",
                    text="饮食停滞可见胃脘胀满。",
                    source="《景岳全书》 胃脘",
                )
            ],
            allowed_terms=["食滞"],
            raw_tool_content="",
        )
        inquiry = InquiryState(
            chief_complaint="胃胀",
            known_facts=KnownFacts(
                duration="两周",
                triggers=["油腻后加重"],
                associated_symptoms=["嗳气"],
            ),
            information_sufficiency="sufficient",
        )

        analysis = SyndromeAgent().analyze(
            user_text="胃胀两周，油腻后加重，嗳气。",
            inquiry=inquiry,
            evidence=evidence,
        )

        self.assertEqual(len(analysis.possible_patterns), 1)
        self.assertEqual(analysis.possible_patterns[0].term, "食滞")
        self.assertEqual(analysis.possible_patterns[0].supporting_evidence, ["E1"])
        self.assertTrue(analysis.not_enough_for_diagnosis)
```

Also update the import list in `tests/test_workflow_agent_flow.py`:

```python
from app.agents.workflow_agent.workflow import (
    AnswerAgent,
    EvidenceAgent,
    InquiryAgent,
    SafetyAgent,
    SyndromeAgent,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow -v
```

Expected: FAIL with `ImportError` for `EvidenceAgent` or `SyndromeAgent`.

- [ ] **Step 3: Add retrieval parsing helpers and fixed agents**

Add these imports near the top of `app/agents/workflow_agent/workflow.py`:

```python
from app.agents.workflow_agent.models import EvidenceItem, filter_allowed_patterns
from app.tools.builtins.retrieval_tool import retrieve_tcm_knowledge
```

Add this type alias near the existing constants:

```python
Retriever = Callable[[str, str], Awaitable[str]]
```

Add these helpers and classes after `InquiryAgent`:

```python
async def _default_retriever(query: str, mode: str) -> str:
    return await retrieve_tcm_knowledge.ainvoke(
        {
            "query": query,
            "mode": mode,
        }
    )


def _line_value(block: str, prefix: str) -> str:
    for line in block.splitlines():
        text = line.strip()
        if text.startswith(prefix):
            return text.replace(prefix, "", 1).strip()
    return ""


def _parse_allowed_terms(raw_text: str) -> list[str]:
    terms: list[str] = []
    collecting = False
    for line in raw_text.splitlines():
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
    return list(dict.fromkeys(terms))


def parse_retrieval_text(raw_text: str) -> EvidenceResult:
    status = _line_value(raw_text, "检索状态：") or "insufficient_evidence"
    mode = _line_value(raw_text, "检索模式：") or "hybrid_parent"
    degraded = "降级检索：是" in raw_text
    evidence: list[EvidenceItem] = []

    for block in raw_text.split("\n\n"):
        stripped = block.strip()
        if not stripped.startswith("[E"):
            continue
        citation = stripped.splitlines()[0].strip().strip("[]")
        evidence.append(
            EvidenceItem(
                id=citation,
                citation_id=citation,
                role=_line_value(stripped, "证据角色："),
                text=_line_value(stripped, "原文："),
                source=_line_value(stripped, "来源："),
            )
        )

    return EvidenceResult(
        retrieval_status=status,
        retrieval_mode=mode,
        degraded=degraded,
        evidence=evidence[:5],
        allowed_terms=_parse_allowed_terms(raw_text),
        raw_tool_content=raw_text,
    )


class EvidenceAgent:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or _default_retriever

    async def retrieve(
        self,
        *,
        user_text: str,
        inquiry: InquiryState,
    ) -> EvidenceResult:
        query_parts = [
            user_text.strip(),
            inquiry.chief_complaint,
            inquiry.known_facts.duration,
            " ".join(inquiry.known_facts.triggers),
            " ".join(inquiry.known_facts.associated_symptoms),
        ]
        query = " ".join(part for part in query_parts if part).strip()
        raw_text = await self.retriever(query, "hybrid")
        return parse_retrieval_text(raw_text)


class SyndromeAgent:
    def analyze(
        self,
        *,
        user_text: str,
        inquiry: InquiryState,
        evidence: EvidenceResult,
    ) -> SyndromeAnalysis:
        candidates: list[PatternCandidate] = []
        for term in evidence.allowed_terms:
            supporting_ids = [
                item.id
                for item in evidence.evidence
                if term in item.text or term in item.role or term in user_text
            ]
            if not supporting_ids and evidence.evidence and term in evidence.raw_tool_content:
                supporting_ids = [evidence.evidence[0].id]
            if not supporting_ids:
                continue
            candidates.append(
                PatternCandidate(
                    term=term,
                    supporting_evidence=supporting_ids[:5],
                    confidence="medium" if inquiry.known_facts.triggers else "low",
                    reason=(
                        "用户描述与本次检索证据存在一定相关性；"
                        "该项只能作为可能相关因素，不等同于诊断。"
                    ),
                )
            )

        analysis = SyndromeAnalysis(
            possible_patterns=candidates,
            not_enough_for_diagnosis=True,
            need_more_info=list(
                dict.fromkeys(
                    [
                        *inquiry.missing_info,
                        "舌象",
                        "疼痛性质",
                    ]
                )
            ),
        )
        return filter_allowed_patterns(analysis, evidence.allowed_terms)
```

- [ ] **Step 4: Run tests to verify Task 3 passes**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow -v
```

Expected: PASS for all six tests in this file.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add app/agents/workflow_agent/workflow.py tests/test_workflow_agent_flow.py
git commit -m "feat: add workflow evidence and syndrome agents"
```

---

### Task 4: Fixed Workflow Orchestrator

**Files:**
- Modify: `app/agents/workflow_agent/workflow.py`
- Modify: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Add failing orchestration tests**

Append these tests inside `WorkflowFixedAgentTests`:

```python
    async def test_workflow_pauses_before_retrieval_when_inquiry_is_insufficient(self):
        retrieval_calls = []

        async def fake_retriever(query: str, mode: str) -> str:
            retrieval_calls.append(query)
            return "检索状态：ok"

        workflow = TCMWorkflow(
            inquiry_agent=InquiryAgent(),
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
        )

        result = await workflow.run(
            user_text="我最近胃胀",
            conversation=[],
        )

        self.assertTrue(result.needs_clarification)
        self.assertEqual(retrieval_calls, [])
        self.assertEqual(result.messages[-1].name, "ask_clarification")
        self.assertIn("InquiryAgent", [event["agent"] for event in result.agent_trace])

    async def test_workflow_runs_answer_then_safety_for_sufficient_information(self):
        async def fake_retriever(query: str, mode: str) -> str:
            return (
                "检索状态：ok\n\n"
                "检索模式：hybrid_parent\n\n"
                "[E1]\n"
                "证据角色：syndrome_pattern\n"
                "原文：食滞可见胃脘胀满。\n"
                "来源：《景岳全书》 胃脘\n\n"
                "允许使用的专业术语：\n"
                "- 食滞\n\n"
                "回答约束：\n"
                "- 不得据此推荐方剂、药物、剂量或煎服法。"
            )

        workflow = TCMWorkflow(
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
        )

        result = await workflow.run(
            user_text="胃胀持续两周，油腻后加重，嗳气，大便正常，无反酸烧心。",
            conversation=[],
        )

        self.assertFalse(result.needs_clarification)
        self.assertIn("食滞", result.final_text)
        self.assertIn("[E1]", result.final_text)
        self.assertIn("SafetyAgent", [event["agent"] for event in result.agent_trace])
        self.assertEqual(result.messages[-1].type, "ai")
```

Also update the import list:

```python
from app.agents.workflow_agent.workflow import (
    AnswerAgent,
    EvidenceAgent,
    InquiryAgent,
    SafetyAgent,
    SyndromeAgent,
    TCMWorkflow,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow -v
```

Expected: FAIL with `ImportError` for `TCMWorkflow`.

- [ ] **Step 3: Implement orchestration result and TCMWorkflow**

Add these imports near the top of `app/agents/workflow_agent/workflow.py`:

```python
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.middlewares.clarification_controller import format_clarification_questions
```

Add this dataclass before `InquiryAgent`:

```python
@dataclass
class WorkflowRunResult:
    messages: list[BaseMessage]
    final_text: str
    needs_clarification: bool
    agent_trace: list[dict]
```

Add this class after `SafetyAgent`:

```python
class TCMWorkflow:
    def __init__(
        self,
        *,
        inquiry_agent: InquiryAgent | None = None,
        evidence_agent: EvidenceAgent | None = None,
        syndrome_agent: SyndromeAgent | None = None,
        answer_agent: AnswerAgent | None = None,
        safety_agent: SafetyAgent | None = None,
    ):
        self.inquiry_agent = inquiry_agent or InquiryAgent()
        self.evidence_agent = evidence_agent or EvidenceAgent()
        self.syndrome_agent = syndrome_agent or SyndromeAgent()
        self.answer_agent = answer_agent or AnswerAgent()
        self.safety_agent = safety_agent or SafetyAgent()

    async def run(
        self,
        *,
        user_text: str,
        conversation: list[dict[str, str]] | None = None,
    ) -> WorkflowRunResult:
        trace: list[dict] = []
        inquiry = self.inquiry_agent.assess(
            user_text,
            conversation=conversation or [],
        )
        trace.append(
            {
                "agent": "InquiryAgent",
                "action": "assess_information",
                "status": (
                    "needs_clarification"
                    if inquiry.should_pause_for_clarification
                    else "completed"
                ),
                "summary": "；".join(inquiry.missing_info) or "问诊信息基本可继续。",
                "payload": inquiry.model_dump(),
            }
        )

        if inquiry.should_pause_for_clarification:
            tool_call_id = "workflow-clarification-1"
            ai_message = AIMessage(
                id="workflow-clarification-ai-1",
                content="为了更准确地帮您分析，请先补充以下关键信息：",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "ask_clarification",
                        "args": {
                            "questions": inquiry.clarification_questions,
                        },
                    }
                ],
            )
            tool_message = ToolMessage(
                id=f"clarification:{tool_call_id}",
                name="ask_clarification",
                tool_call_id=tool_call_id,
                content=format_clarification_questions(
                    inquiry.clarification_questions
                ),
            )
            return WorkflowRunResult(
                messages=[ai_message, tool_message],
                final_text="",
                needs_clarification=True,
                agent_trace=trace,
            )

        evidence = await self.evidence_agent.retrieve(
            user_text=user_text,
            inquiry=inquiry,
        )
        trace.append(
            {
                "agent": "EvidenceAgent",
                "action": "retrieve",
                "tool": "retrieve_tcm_knowledge",
                "status": evidence.retrieval_status,
                "summary": "完成中医知识检索。",
                "retrieval": {
                    "retrieval_mode": evidence.retrieval_mode,
                    "allowed_terms": evidence.allowed_terms,
                    "degraded": evidence.degraded,
                },
            }
        )

        syndrome = self.syndrome_agent.analyze(
            user_text=user_text,
            inquiry=inquiry,
            evidence=evidence,
        )
        trace.append(
            {
                "agent": "SyndromeAgent",
                "action": "analyze_possible_patterns",
                "status": "completed",
                "summary": "整理可能相关因素，不输出诊断。",
                "payload": syndrome.model_dump(),
            }
        )

        draft = self.answer_agent.compose(
            user_text=user_text,
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome,
        )
        safety = self.safety_agent.review(
            draft.draft_answer,
            inquiry=inquiry,
            evidence=evidence,
            syndrome=syndrome,
        )

        if safety.rewrite_required:
            draft = self.answer_agent.compose(
                user_text=user_text,
                inquiry=inquiry,
                evidence=evidence,
                syndrome=syndrome,
                safety_review=safety,
            )
            safety = self.safety_agent.review(
                draft.draft_answer,
                inquiry=inquiry,
                evidence=evidence,
                syndrome=syndrome,
            )

        if safety.rewrite_required:
            draft = self.answer_agent.safe_fallback(inquiry)
            safety = self.safety_agent.review(
                draft.draft_answer,
                inquiry=inquiry,
                evidence=evidence,
                syndrome=syndrome,
            )

        trace.append(
            {
                "agent": "AnswerAgent",
                "action": "compose_answer",
                "status": "completed",
                "summary": "已基于前序 Agent 结果生成最终语言。",
            }
        )
        trace.append(
            {
                "agent": "SafetyAgent",
                "action": "review",
                "status": "rewrite_required" if safety.rewrite_required else "passed",
                "summary": "完成诊断、处方、剂量和危险信号检查。",
                "payload": safety.model_dump(),
            }
        )

        retrieve_call_id = "workflow-retrieval-1"
        retrieval_ai = AIMessage(
            id="workflow-retrieval-ai-1",
            content="",
            tool_calls=[
                {
                    "id": retrieve_call_id,
                    "name": "retrieve_tcm_knowledge",
                    "args": {
                        "query": user_text,
                        "mode": "hybrid",
                    },
                }
            ],
        )
        retrieval_tool = ToolMessage(
            id=f"retrieval:{retrieve_call_id}",
            name="retrieve_tcm_knowledge",
            tool_call_id=retrieve_call_id,
            content=evidence.raw_tool_content,
        )
        final_ai = AIMessage(
            id="workflow-final-ai-1",
            content=draft.draft_answer,
        )

        return WorkflowRunResult(
            messages=[retrieval_ai, retrieval_tool, final_ai],
            final_text=draft.draft_answer,
            needs_clarification=False,
            agent_trace=trace,
        )
```

- [ ] **Step 4: Run tests to verify Task 4 passes**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow -v
```

Expected: PASS for all eight tests in this file.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add app/agents/workflow_agent/workflow.py tests/test_workflow_agent_flow.py
git commit -m "feat: orchestrate fixed workflow agent"
```

---

### Task 5: Runtime-Compatible WorkflowAgent Adapter

**Files:**
- Create: `app/agents/workflow_agent/agent.py`
- Modify: `tests/test_workflow_agent_flow.py`

- [ ] **Step 1: Add failing runtime smoke tests**

Append these imports to `tests/test_workflow_agent_flow.py`:

```python
import json
from unittest.mock import AsyncMock, patch

from app.agents.workflow_agent.agent import WorkflowAgent
from app.runtime.runs.worker import run_agent
from app.runtime.stream import StreamBridge
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore
```

Append these helper and test classes to the bottom of `tests/test_workflow_agent_flow.py`, before the `if __name__ == "__main__"` block:

```python
class WorkflowRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def drain_events(self, bridge: StreamBridge, run_id: str) -> list[str]:
        events = []
        async for event in bridge.subscribe(run_id):
            events.append(event)
        return events

    def parse_events(self, events: list[str]) -> list[tuple[str, dict]]:
        parsed = []
        for raw_event in events:
            event_name = ""
            data = ""
            for line in raw_event.splitlines():
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data = line[len("data:"):].strip()
            if event_name and data:
                parsed.append((event_name, json.loads(data)))
        return parsed

    async def test_workflow_agent_runs_through_existing_runtime_final_event(self):
        async def fake_retriever(query: str, mode: str) -> str:
            return (
                "检索状态：ok\n\n"
                "检索模式：hybrid_parent\n\n"
                "[E1]\n"
                "证据角色：syndrome_pattern\n"
                "原文：食滞可见胃脘胀满。\n"
                "来源：《景岳全书》 胃脘\n\n"
                "允许使用的专业术语：\n"
                "- 食滞\n\n"
                "回答约束：\n"
                "- 不得据此推荐方剂、药物、剂量或煎服法。"
            )

        workflow = TCMWorkflow(
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
        )
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "workflow_agent")
        bridge.create(run.run_id)

        with patch(
            "app.runtime.runs.worker.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": "经过 Guardrail 的 workflow 答案",
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": True,
                    "allowed_terms": ["食滞"],
                }
            ),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                thread_store=thread_store,
                record=run,
                agent_factory=lambda context: WorkflowAgent(
                    workflow=workflow,
                    thread_store=thread_store,
                ),
                input_data={
                    "messages": [
                        {
                            "type": "human",
                            "content": "胃胀持续两周，油腻后加重，嗳气，大便正常，无反酸烧心。",
                        }
                    ]
                },
                context={"stream_mode": ["messages"]},
            )

        parsed_events = self.parse_events(await self.drain_events(bridge, run.run_id))
        final_events = [data for event, data in parsed_events if event == "final"]
        stored_thread = await thread_store.get(thread.thread_id)

        self.assertEqual((await run_manager.get(run.run_id)).status, "success")
        self.assertTrue(final_events)
        self.assertEqual(
            final_events[-1]["assistant_message"],
            "经过 Guardrail 的 workflow 答案",
        )
        stored_agents = [
            item.get("agent")
            for item in stored_thread.values.get("last_agent_trace", [])
        ]
        self.assertIn("SafetyAgent", stored_agents)

    async def test_workflow_agent_clarification_uses_existing_waiting_flow(self):
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()
        run = await run_manager.create(thread.thread_id, "workflow_agent")
        bridge.create(run.run_id)

        await run_agent(
            bridge=bridge,
            run_manager=run_manager,
            thread_store=thread_store,
            record=run,
            agent_factory=lambda context: WorkflowAgent(thread_store=thread_store),
            input_data={
                "messages": [
                    {
                        "type": "human",
                        "content": "我最近胃胀",
                    }
                ]
            },
            context={"stream_mode": ["messages"]},
        )

        parsed_events = self.parse_events(await self.drain_events(bridge, run.run_id))
        final_events = [data for event, data in parsed_events if event == "final"]

        self.assertEqual(
            (await run_manager.get(run.run_id)).status,
            "waiting_clarification",
        )
        self.assertTrue(final_events)
        self.assertEqual(final_events[-1]["status"], "need_clarification")
        self.assertLessEqual(
            len(final_events[-1]["pending_clarification"]),
            3,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.agents.workflow_agent.agent'`.

- [ ] **Step 3: Implement runtime-compatible adapter**

Create `app/agents/workflow_agent/agent.py`:

```python
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessageChunk, HumanMessage

from app.agents.workflow_agent.workflow import TCMWorkflow
from app.runtime.serialization import serialize_message
from app.runtime.state import state as runtime_state


def _extract_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    return str(content or "")


def _latest_user_text(input_data: dict[str, Any]) -> str:
    for message in input_data.get("messages", []):
        if message.get("role") == "user":
            return _extract_text(message.get("content", "")).strip()
    return ""


class WorkflowAgent:
    def __init__(
        self,
        *,
        workflow: TCMWorkflow | None = None,
        thread_store: Any | None = None,
    ):
        self.workflow = workflow or TCMWorkflow()
        self.thread_store = thread_store or runtime_state.thread_store

    def _thread_id(self, config: dict[str, Any]) -> str:
        return str(config.get("configurable", {}).get("thread_id", ""))

    async def _read_thread_values(self, thread_id: str) -> dict[str, Any]:
        thread = await self.thread_store.get(thread_id)
        return dict(thread.values) if thread else {}

    async def _read_messages(self, thread_id: str) -> list[Any]:
        values = await self._read_thread_values(thread_id)
        messages = values.get("messages", [])
        return list(messages) if isinstance(messages, list) else []

    async def _read_conversation(self, thread_id: str) -> list[dict[str, str]]:
        values = await self._read_thread_values(thread_id)
        conversation = values.get("conversation", [])
        return list(conversation) if isinstance(conversation, list) else []

    async def aget_state(self, config: dict[str, Any]):
        thread_id = self._thread_id(config)
        return SimpleNamespace(
            values={
                "messages": await self._read_messages(thread_id),
            },
            next=(),
        )

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ):
        thread_id = self._thread_id(config)
        messages = await self._read_messages(thread_id)
        updates = values.get("messages", [])

        for update in updates:
            update_payload = serialize_message(update)
            update_id = update_payload.get("id")
            if not update_id:
                continue
            for index in range(len(messages) - 1, -1, -1):
                message = messages[index]
                if isinstance(message, dict) and message.get("id") == update_id:
                    messages[index] = {
                        **message,
                        "content": update_payload.get("content", ""),
                    }
                    break

        await self.thread_store.update_values(
            thread_id,
            {
                "messages": messages,
            },
        )

    async def astream(
        self,
        input_data: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode,
    ):
        thread_id = self._thread_id(config)
        previous_messages = await self._read_messages(thread_id)
        conversation = await self._read_conversation(thread_id)
        user_text = _latest_user_text(input_data)

        workflow_result = await self.workflow.run(
            user_text=user_text,
            conversation=conversation,
        )

        current_messages = [HumanMessage(content=user_text), *workflow_result.messages]
        final_messages = [*previous_messages, *current_messages]
        serialized_messages = [
            message if isinstance(message, dict) else serialize_message(message)
            for message in final_messages
        ]

        await self.thread_store.update_values(
            thread_id,
            {
                "messages": serialized_messages,
                "last_agent_trace": workflow_result.agent_trace,
            },
        )

        modes = stream_mode if isinstance(stream_mode, list) else [stream_mode]
        if "messages" in modes and workflow_result.final_text:
            yield (
                "messages",
                (
                    AIMessageChunk(content=workflow_result.final_text),
                    {
                        "langgraph_node": "workflow_agent",
                        "thread_id": thread_id,
                    },
                ),
            )

        yield (
            "values",
            {
                "messages": serialized_messages,
                "workflow_trace": workflow_result.agent_trace,
            },
        )


def make_workflow_agent(context: dict[str, Any] | None = None) -> WorkflowAgent:
    return WorkflowAgent()
```

Update `app/agents/workflow_agent/__init__.py` after `agent.py` exists:

```python
from app.agents.workflow_agent.agent import make_workflow_agent

__all__ = ["make_workflow_agent"]
```

- [ ] **Step 4: Preserve workflow trace in the existing worker**

Modify `app/runtime/runs/worker.py`.

Add this variable after `clarification_to_emit = ""`:

```python
        workflow_trace: list[dict[str, Any]] = []
```

Add this block immediately after `serialized_values = serialize(chunk, mode="values")`:

```python
            if isinstance(serialized_values, dict):
                candidate_workflow_trace = serialized_values.get("workflow_trace")
                if isinstance(candidate_workflow_trace, list):
                    workflow_trace = [
                        item
                        for item in candidate_workflow_trace
                        if isinstance(item, dict)
                    ]
```

Replace the final message-trace assignment shown below:

```python
        agent_trace = extract_agent_trace_from_messages(
            final_messages[message_start_index:]
        )
```

with:

```python
        message_trace = extract_agent_trace_from_messages(
            final_messages[message_start_index:]
        )
        agent_trace = workflow_trace or message_trace
```

- [ ] **Step 5: Run tests to verify Task 5 passes**

Run:

```powershell
python -m unittest tests.test_workflow_agent_flow -v
```

Expected: PASS.

- [ ] **Step 6: Run compatibility tests touched by the adapter**

Run:

```powershell
python -m unittest tests.test_async_agent_factory tests.test_clarification_flow -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

Run:

```powershell
git add app/agents/workflow_agent/__init__.py app/agents/workflow_agent/agent.py app/runtime/runs/worker.py tests/test_workflow_agent_flow.py
git commit -m "feat: add runtime workflow agent adapter"
```

---

### Task 6: Registry Integration

**Files:**
- Modify: `app/agents/registry.py`
- Test: `tests/test_workflow_agent_registry.py`

- [ ] **Step 1: Write failing registry tests**

Create `tests/test_workflow_agent_registry.py`:

```python
import unittest

from app.agents.registry import resolve_agent_factory
from app.agents.workflow_agent.agent import WorkflowAgent


class WorkflowAgentRegistryTests(unittest.TestCase):
    def test_resolves_workflow_agent_without_replacing_lead_agent(self):
        workflow_factory = resolve_agent_factory("workflow_agent")
        workflow_agent = workflow_factory({})

        self.assertIsInstance(workflow_agent, WorkflowAgent)

        lead_factory = resolve_agent_factory("lead_agent")

        self.assertEqual(lead_factory.__name__, "make_lead_agent")

    def test_unknown_assistant_still_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown assistant_id"):
            resolve_agent_factory("missing_agent")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tests.test_workflow_agent_registry -v
```

Expected: FAIL with `ValueError: Unknown assistant_id: workflow_agent`.

- [ ] **Step 3: Register `workflow_agent`**

Modify `app/agents/registry.py`:

```python
from collections.abc import Callable
from typing import Any

from app.agents.lead_agent.agent import make_lead_agent
from app.agents.workflow_agent.agent import make_workflow_agent


AgentFactory = Callable[[dict[str, Any] | None], Any]


def resolve_agent_factory(assistant_id: str) -> AgentFactory:
    """
    根据 assistant_id 解析 Agent Factory。

    lead_agent 保持旧 prompt/tools 路径。
    workflow_agent 使用固定 Inquiry -> Evidence -> Syndrome -> Answer -> Safety 流程。
    """

    if assistant_id == "lead_agent":
        return make_lead_agent

    if assistant_id == "workflow_agent":
        return make_workflow_agent

    raise ValueError(f"Unknown assistant_id: {assistant_id}")
```

- [ ] **Step 4: Run registry tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_registry -v
```

Expected: PASS.

- [ ] **Step 5: Run focused workflow suite**

Run:

```powershell
python -m unittest tests.test_workflow_agent_models tests.test_workflow_agent_flow tests.test_workflow_agent_registry -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

Run:

```powershell
git add app/agents/registry.py tests/test_workflow_agent_registry.py
git commit -m "feat: register workflow agent"
```

---

### Task 7: Final Verification and Documentation Check

**Files:**
- Modify only if a focused test exposes a real issue.

- [ ] **Step 1: Compile changed Python files**

Run:

```powershell
python -m py_compile app/agents/workflow_agent/models.py app/agents/workflow_agent/workflow.py app/agents/workflow_agent/agent.py app/agents/registry.py
```

Expected: command exits 0.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
python -m unittest tests.test_workflow_agent_models tests.test_workflow_agent_flow tests.test_workflow_agent_registry tests.test_async_agent_factory tests.test_clarification_flow tests.test_lead_agent_factory -v
```

Expected: PASS.

- [ ] **Step 3: Inspect git diff for unintended files**

Run:

```powershell
git status --short
git diff -- app/agents/workflow_agent app/agents/registry.py tests/test_workflow_agent_models.py tests/test_workflow_agent_flow.py tests/test_workflow_agent_registry.py
```

Expected:

```text
Only workflow_agent files, registry.py, worker.py, and workflow_agent tests are changed.
.env and .gitignore may remain dirty from pre-existing local changes and must not be staged.
```

- [ ] **Step 4: Commit final verification adjustment if any**

If Step 2 required a small fix, run:

```powershell
git add app/agents/workflow_agent app/agents/registry.py app/runtime/runs/worker.py tests/test_workflow_agent_models.py tests/test_workflow_agent_flow.py tests/test_workflow_agent_registry.py
git commit -m "test: verify workflow agent integration"
```

If no fix was needed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- `workflow_agent` registration is covered in Task 6.
- InquiryAgent clarification-first behavior is covered in Tasks 1, 2, 4, and 5.
- EvidenceAgent exclusive retrieval ownership is covered in Task 3 and workflow pause-before-retrieval tests in Task 4.
- SyndromeAgent possible-pattern output and allowed-term filtering are covered in Tasks 1 and 3.
- AnswerAgent no-new-evidence/no-prescription behavior is covered in Task 2.
- SafetyAgent independent final review and rewrite path are covered in Tasks 2 and 4.
- Runtime compatibility with `run_agent` final and clarification paths is covered in Task 5.
- Existing `lead_agent` path preservation is covered in Task 6 and focused compatibility tests in Task 7.

Placeholder scan:

- The plan uses exact file paths, exact test bodies, exact implementation snippets, and exact commands.
- There are no unresolved placeholder sections.

Type consistency:

- `InquiryState`, `EvidenceResult`, `SyndromeAnalysis`, `AnswerDraft`, and `SafetyReview` are defined in Task 1 and used consistently by Tasks 2 through 5.
- `WorkflowRunResult` is defined before `TCMWorkflow` and consumed by `WorkflowAgent`.
- `make_workflow_agent` is defined in `agent.py` and imported by `registry.py`.
