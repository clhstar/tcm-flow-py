import json
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.workflow_agent.agent import WorkflowAgent
from app.agents.workflow_agent.models import (
    AnswerDraft,
    EvidenceItem,
    EvidenceResult,
    InquiryState,
    IntentState,
    KnownFacts,
    PatternCandidate,
    SafetyReview,
    SyndromeAnalysis,
)
from app.agents.workflow_agent.components.evidence import EvidenceAgent
from app.agents.workflow_agent.components.syndrome import SyndromeAgent
from app.agents.workflow_agent.workflow import TCMWorkflow
from app.runtime.runs.worker import run_agent
from app.runtime.stream import StreamBridge
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


DEFAULT_SCHEMA_ORDER = [
    IntentState,
    InquiryState,
    SyndromeAnalysis,
    AnswerDraft,
    SafetyReview,
]


def workflow_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


class FakeStructuredRunnable:
    def __init__(self, model, schema):
        self.model = model
        self.schema = schema

    async def ainvoke(self, messages):
        self.model.invocations.append({"schema": self.schema, "messages": messages})
        responses = self.model.responses_by_schema.get(self.schema, [])
        if not responses:
            raise AssertionError(f"no fake response configured for {self.schema.__name__}")
        return responses.pop(0)


class FakeWorkflowModel:
    def __init__(self, responses_by_schema, schema_order=None):
        self.responses_by_schema = {
            schema: list(responses) if isinstance(responses, list) else [responses]
            for schema, responses in responses_by_schema.items()
        }
        self.schema_queue = [
            *(schema_order or self.responses_by_schema),
            *(
                schema
                for schema in DEFAULT_SCHEMA_ORDER
                if schema not in (schema_order or self.responses_by_schema)
            ),
        ]
        self.structured_calls = []
        self.invocations = []

    def with_structured_output(self, schema=None, *, method, strict=None):
        schema = schema or self.schema_queue.pop(0)
        self.structured_calls.append(
            {"schema": schema, "method": method, "strict": strict}
        )
        return FakeStructuredRunnable(self, schema)


def workflow_model(responses_by_schema):
    return FakeWorkflowModel(
        responses_by_schema,
        schema_order=DEFAULT_SCHEMA_ORDER,
    )


def inquiry_intent():
    return IntentState(
        primary_intent="symptom_consultation",
        confidence="high",
        is_tcm_domain_query=True,
        is_personal_health_query=True,
        requires_retrieval=True,
        should_enter_inquiry=True,
        route_hint="inquiry",
    )


def sufficient_inquiry(risk_flags=None):
    return InquiryState(
        chief_complaint="胃胀",
        known_facts=KnownFacts(
            duration="两周",
            triggers=["油腻后加重"],
            associated_symptoms=["嗳气"],
            risk_flags=risk_flags or [],
        ),
        missing_info=["舌象", "大便"],
        information_sufficiency="sufficient",
    )


def insufficient_inquiry():
    return InquiryState(
        chief_complaint="胃胀",
        known_facts=KnownFacts(),
        missing_info=["持续时间", "诱因", "伴随症状"],
        information_sufficiency="insufficient",
        clarification_questions=[
            "这种胃胀持续多久了？",
            "进食、油腻或受凉后会加重吗？",
            "是否伴有嗳气、反酸、腹痛或大便变化？",
        ],
        should_pause_for_clarification=True,
    )


def evidence_text():
    return (
        "检索状态：ok\n\n"
        "检索模式：hybrid_parent\n\n"
        "[E1]\n"
        "证据角色：syndrome_pattern\n"
        "原文：饮食停滞可见胃脘胀满。\n"
        "来源：《景岳全书》胃脘\n\n"
        "允许使用的专业术语：\n"
        "- 食滞\n\n"
        "回答约束：\n"
        "- 不得据此推荐方剂、药物、剂量或煎服法。"
    )


def syndrome_analysis():
    return SyndromeAnalysis(
        possible_patterns=[
            PatternCandidate(
                term="食滞",
                supporting_evidence=["E1"],
                confidence="medium",
                reason="用户提到油腻后胃胀加重，并伴有嗳气。",
            )
        ],
        not_enough_for_diagnosis=True,
        need_more_info=["舌象", "大便"],
    )


def safe_review():
    return SafetyReview(
        final_safety_level="low",
        rewrite_required=False,
    )


class WorkflowLLMBackedTests(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_pauses_before_retrieval_when_inquiry_llm_requests_it(self):
        retrieval_calls = []

        async def fake_retriever(query: str, mode: str) -> str:
            retrieval_calls.append({"query": query, "mode": mode})
            return evidence_text()

        model = workflow_model(
            {
                IntentState: inquiry_intent(),
                InquiryState: insufficient_inquiry(),
            }
        )
        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
            checkpointer=InMemorySaver(),
        )

        result = await workflow.run(
            user_text="我最近胃胀",
            conversation=[],
            config=workflow_config("pause-before-retrieval"),
        )

        self.assertTrue(result.needs_clarification)
        self.assertEqual(retrieval_calls, [])
        self.assertEqual(result.messages[-1].name, "ask_clarification")
        self.assertEqual(
            [call["schema"] for call in model.invocations],
            [IntentState, InquiryState],
        )
        self.assertNotIn("4.", result.messages[-1].content)

    async def test_cause_question_context_continues_despite_inquiry_pause_flag(self):
        retrieval_calls = []

        async def fake_retriever(query: str, mode: str) -> str:
            retrieval_calls.append({"query": query, "mode": mode})
            return evidence_text()

        model = workflow_model(
            {
                IntentState: IntentState(
                    primary_intent="cause_explanation",
                    confidence="high",
                    is_tcm_domain_query=True,
                    is_personal_health_query=True,
                    requires_retrieval=True,
                    should_enter_inquiry=True,
                    route_hint="inquiry",
                ),
                InquiryState: InquiryState(
                    chief_complaint="\u8d77\u5e8a\u80c3\u75db",
                    known_facts=KnownFacts(
                        duration="\u4e24\u5929",
                        triggers=[
                            "\u7a7a\u8179\u65f6\u75db",
                            "\u4e0e\u996e\u98df\u65e0\u5173",
                        ],
                        associated_symptoms=[
                            "\u6076\u5fc3",
                            "\u65e0\u53cd\u9178",
                            "\u65e0\u8179\u80c0",
                        ],
                    ),
                    missing_info=[
                        "\u662f\u5426\u53d1\u70ed",
                        "\u662f\u5426\u5927\u4fbf\u989c\u8272\u6539\u53d8",
                    ],
                    information_sufficiency="insufficient",
                    clarification_questions=[
                        "\u662f\u5426\u8fd8\u6709\u5455\u5410\uff1f",
                        "\u662f\u5426\u591c\u95f4\u53d1\u4f5c\uff1f",
                    ],
                    should_pause_for_clarification=True,
                ),
                SyndromeAnalysis: syndrome_analysis(),
                AnswerDraft: AnswerDraft(
                    draft_answer=(
                        "\u76ee\u524d\u53ea\u80fd\u8bf4\u53ef\u80fd\u4e0e\u80c3\u90e8\u523a\u6fc0\u7b49\u56e0\u7d20\u76f8\u5173\uff0c"
                        "\u8fd8\u4e0d\u80fd\u8bca\u65ad\u3002[E1]"
                    )
                ),
                SafetyReview: safe_review(),
            }
        )
        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
            checkpointer=InMemorySaver(),
        )

        result = await workflow.run(
            user_text=(
                "\u80c3\u504f\u4e0a\uff0c\u9690\u9690\u75db\uff0c"
                "\u7a7a\u8179\u65f6\u75db\uff0c\u6ca1\u6709\u53cd\u9178"
            ),
            conversation=[
                {
                    "role": "user",
                    "content": "\u6700\u8fd1\u8d77\u5e8a\u7ecf\u5e38\u80c3\u75db\u4ec0\u4e48\u539f\u56e0\u5bfc\u81f4\u7684",
                },
                {
                    "role": "assistant",
                    "content": "\u8bf7\u8865\u5145\u75bc\u75db\u90e8\u4f4d\u3001\u53cd\u9178\u548c\u6301\u7eed\u65f6\u95f4\u3002",
                },
            ],
            config=workflow_config("cause-question-context"),
        )

        self.assertFalse(result.needs_clarification)
        self.assertEqual(len(retrieval_calls), 1)
        self.assertIn("[E1]", result.final_text)
        self.assertEqual(
            [call["schema"] for call in model.invocations],
            [IntentState, InquiryState, SyndromeAnalysis, AnswerDraft, SafetyReview],
        )
        self.assertEqual(
            result.agent_trace[1]["information_sufficiency"],
            "sufficient",
        )
        self.assertFalse(result.agent_trace[1]["should_pause_for_clarification"])

    async def test_workflow_runs_llm_agents_in_fixed_order_for_sufficient_information(self):
        retrieval_calls = []

        async def fake_retriever(query: str, mode: str) -> str:
            retrieval_calls.append({"query": query, "mode": mode})
            return evidence_text()

        model = workflow_model(
            {
                IntentState: inquiry_intent(),
                InquiryState: sufficient_inquiry(),
                SyndromeAnalysis: syndrome_analysis(),
                AnswerDraft: AnswerDraft(
                    draft_answer=(
                        "从你描述的胃胀、嗳气、油腻后加重来看，中医上可能与食滞等因素有关，"
                        "但仅凭这些信息还不能判断具体证候。[E1]"
                    )
                ),
                SafetyReview: safe_review(),
            }
        )
        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
            checkpointer=InMemorySaver(),
        )

        result = await workflow.run(
            user_text="胃胀持续两周，油腻后加重，嗳气，大便正常。",
            conversation=[],
            config=workflow_config("fixed-order"),
        )

        self.assertFalse(result.needs_clarification)
        self.assertEqual(retrieval_calls[0]["mode"], "hybrid")
        self.assertIn("胃胀", retrieval_calls[0]["query"])
        self.assertIn("食滞", result.final_text)
        self.assertIn("[E1]", result.final_text)
        self.assertEqual(
            [call["schema"] for call in model.invocations],
            [IntentState, InquiryState, SyndromeAnalysis, AnswerDraft, SafetyReview],
        )
        self.assertEqual(
            {
                (call["schema"], call["method"], call["strict"])
                for call in model.structured_calls
            },
            {
                (IntentState, "json_mode", None),
                (InquiryState, "json_mode", None),
                (SyndromeAnalysis, "json_mode", None),
                (AnswerDraft, "json_mode", None),
                (SafetyReview, "json_mode", None),
            },
        )

    async def test_safety_rewrite_runs_answer_again_and_rechecks_final_answer(self):
        async def fake_retriever(query: str, mode: str) -> str:
            return evidence_text()

        model = workflow_model(
            {
                IntentState: inquiry_intent(),
                InquiryState: sufficient_inquiry(risk_flags=["胸痛"]),
                SyndromeAnalysis: syndrome_analysis(),
                AnswerDraft: [
                    AnswerDraft(draft_answer="你这是食滞证，可以用某某汤，每次10克。"),
                    AnswerDraft(
                        draft_answer=(
                            "目前只能做谨慎分析，不能给出诊断或用药建议。"
                            "你提到胸痛，建议及时线下就医评估。[E1]"
                        )
                    ),
                ],
                SafetyReview: [
                    SafetyReview(
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
                    ),
                    SafetyReview(
                        has_risk_flags=True,
                        risk_flags=["胸痛"],
                        needs_offline_medical_advice=True,
                        final_safety_level="high",
                        rewrite_required=False,
                    ),
                ],
            }
        )
        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
            checkpointer=InMemorySaver(),
        )

        result = await workflow.run(
            user_text="胃胀持续两周，油腻后加重，嗳气，同时胸痛。",
            conversation=[],
            config=workflow_config("safety-rewrite"),
        )

        self.assertIn("线下就医", result.final_text)
        self.assertNotIn("某某汤", result.final_text)
        self.assertEqual(
            [call["schema"] for call in model.invocations],
            [
                IntentState,
                InquiryState,
                SyndromeAnalysis,
                AnswerDraft,
                SafetyReview,
                AnswerDraft,
                SafetyReview,
            ],
        )
        self.assertEqual(
            [event.get("stage") for event in result.agent_trace if event["agent"] == "SafetyAgent"],
            ["initial", "rewrite"],
        )

    async def test_syndrome_agent_filters_unallowed_terms_from_llm_output(self):
        model = FakeWorkflowModel(
            {
                SyndromeAnalysis: SyndromeAnalysis(
                    possible_patterns=[
                        PatternCandidate(
                            term="食滞",
                            supporting_evidence=["E1", "E9"],
                            confidence="medium",
                            reason="与饮食后加重有关。",
                        ),
                        PatternCandidate(
                            term="脾胃虚弱",
                            supporting_evidence=["E1"],
                            confidence="low",
                            reason="这个术语不在检索允许范围内。",
                        ),
                    ],
                    not_enough_for_diagnosis=True,
                    need_more_info=["舌象"],
                )
            }
        )
        agent = SyndromeAgent(model)
        analysis = await agent.analyze(
            user_text="胃胀两周，油腻后加重。",
            inquiry=sufficient_inquiry(),
            evidence=EvidenceResult(
                retrieval_status="ok",
                retrieval_mode="hybrid_parent",
                evidence=[
                    EvidenceItem(
                        id="E1",
                        citation_id="E1",
                        role="syndrome_pattern",
                        text="饮食停滞可见胃脘胀满。",
                        source="《景岳全书》胃脘",
                    )
                ],
                allowed_terms=["食滞"],
                raw_tool_content=evidence_text(),
            ),
        )

        self.assertEqual([item.term for item in analysis.possible_patterns], ["食滞"])
        self.assertEqual(analysis.possible_patterns[0].supporting_evidence, ["E1"])

    async def test_evidence_agent_is_only_component_that_calls_retrieval(self):
        calls = []

        async def fake_retriever(query: str, mode: str) -> str:
            calls.append({"query": query, "mode": mode})
            return evidence_text()

        result = await EvidenceAgent(retriever=fake_retriever).retrieve(
            user_text="胃胀两周，油腻后加重。",
            inquiry=sufficient_inquiry(),
        )

        self.assertEqual(calls, [{"query": "胃胀两周，油腻后加重。 胃胀 两周 油腻后加重 嗳气", "mode": "hybrid"}])
        self.assertEqual(result.evidence[0].id, "E1")
        self.assertEqual(result.allowed_terms, ["食滞"])
        self.assertIn("允许使用的专业术语", result.raw_tool_content)

    async def test_evidence_agent_parses_degraded_line_and_ignores_e6(self):
        async def fake_retriever(query: str, mode: str) -> str:
            return (
                "检索状态：ok\n"
                "检索模式：keyword_parent\n"
                "降级检索：是（模型不可用）\n\n"
                "[E6]\n"
                "证据角色：syndrome_pattern\n"
                "原文：不应保留。\n"
                "来源：B\n\n"
                "[E1]\n"
                "证据角色：syndrome_pattern\n"
                "原文：应保留。\n"
                "来源：A\n\n"
                "允许使用的专业术语：\n"
                "- 食滞\n\n"
                "回答约束：\n"
                "- 不得推荐方药。"
            )

        result = await EvidenceAgent(retriever=fake_retriever).retrieve(
            user_text="胃胀两周，油腻后加重。",
            inquiry=sufficient_inquiry(),
        )

        self.assertTrue(result.degraded)
        self.assertEqual(result.retrieval_mode, "keyword_parent")
        self.assertEqual([(item.id, item.text, item.source) for item in result.evidence], [("E1", "应保留。", "A")])


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
                    event_name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data = line[len("data:") :].strip()
            if event_name and data:
                parsed.append((event_name, json.loads(data)))
        return parsed

    async def test_workflow_agent_runs_through_existing_runtime_final_event(self):
        async def fake_retriever(query: str, mode: str) -> str:
            return evidence_text()

        final_answer = "从描述看可能与食滞等因素有关，但还不能判断具体证候。[E1]"
        model = workflow_model(
            {
                IntentState: inquiry_intent(),
                InquiryState: sufficient_inquiry(),
                SyndromeAnalysis: syndrome_analysis(),
                AnswerDraft: AnswerDraft(draft_answer=final_answer),
                SafetyReview: safe_review(),
            }
        )
        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
            checkpointer=InMemorySaver(),
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
                    "final_text": final_answer,
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
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
                            "content": "胃胀持续两周，油腻后加重，嗳气，大便正常。",
                        }
                    ]
                },
                context={"stream_mode": ["messages"]},
            )

        parsed_events = self.parse_events(await self.drain_events(bridge, run.run_id))
        final_events = [data for event, data in parsed_events if event == "final"]
        stored_thread = await thread_store.get(thread.thread_id)

        self.assertEqual((await run_manager.get(run.run_id)).status, "success")
        self.assertEqual(final_events[-1]["assistant_message"], final_answer)
        self.assertTrue(
            {
                "last_validation",
                "last_allowed_terms",
                "last_rewritten",
                "last_agent_trace",
            }.isdisjoint(stored_thread.values)
        )

    async def test_workflow_agent_clarification_uses_existing_waiting_flow(self):
        model = workflow_model(
            {
                IntentState: inquiry_intent(),
                InquiryState: insufficient_inquiry(),
            }
        )
        workflow = TCMWorkflow(model=model, checkpointer=InMemorySaver())
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
            agent_factory=lambda context: WorkflowAgent(
                workflow=workflow,
                thread_store=thread_store,
            ),
            input_data={"messages": [{"type": "human", "content": "我最近胃胀"}]},
            context={"stream_mode": ["messages"]},
        )

        parsed_events = self.parse_events(await self.drain_events(bridge, run.run_id))
        final_events = [data for event, data in parsed_events if event == "final"]
        stored_thread = await thread_store.get(thread.thread_id)

        self.assertEqual(
            (await run_manager.get(run.run_id)).status,
            "waiting_clarification",
        )
        self.assertEqual(final_events[-1]["status"], "need_clarification")
        self.assertLessEqual(len(final_events[-1]["pending_clarification"]), 3)
        self.assertTrue(stored_thread.values["messages"])
        self.assertTrue(
            any(
                message.get("tool_calls")
                for message in stored_thread.values["messages"]
            )
        )
        self.assertEqual(
            stored_thread.values["conversation"][-2:],
            [
                {"role": "user", "content": "我最近胃胀"},
                {
                    "role": "assistant",
                    "content": final_events[-1]["assistant_message"],
                },
            ],
        )
        self.assertNotIn("last_agent_trace", stored_thread.values)

    async def test_followup_after_clarification_does_not_reuse_old_question(self):
        async def fake_retriever(query: str, mode: str) -> str:
            return evidence_text()

        old_question = "进食、油腻或受凉后会加重吗？"
        final_answer = "根据补充信息，饭后缓解但仍不能直接判断具体证候。[E1]"
        model = workflow_model(
            {
                IntentState: [
                    inquiry_intent(),
                    IntentState(
                        primary_intent="followup_clarification",
                        confidence="high",
                        is_tcm_domain_query=True,
                        is_personal_health_query=True,
                        requires_retrieval=True,
                        should_enter_inquiry=True,
                        route_hint="inquiry",
                    ),
                ],
                InquiryState: [
                    insufficient_inquiry(),
                    InquiryState(
                        chief_complaint="胃痛",
                        known_facts=KnownFacts(
                            duration="2天",
                            triggers=["饭后缓解"],
                            associated_symptoms=["没有其他感觉"],
                        ),
                        missing_info=["疼痛性质", "大便情况"],
                        information_sufficiency="sufficient",
                    ),
                ],
                SyndromeAnalysis: syndrome_analysis(),
                AnswerDraft: AnswerDraft(draft_answer=final_answer),
                SafetyReview: safe_review(),
            }
        )
        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
            checkpointer=InMemorySaver(),
        )
        bridge = StreamBridge()
        run_manager = RunManager()
        thread_store = ThreadStore()
        thread = await thread_store.create()

        first_run = await run_manager.create(thread.thread_id, "workflow_agent")
        bridge.create(first_run.run_id)
        await run_agent(
            bridge=bridge,
            run_manager=run_manager,
            thread_store=thread_store,
            record=first_run,
            agent_factory=lambda context: WorkflowAgent(
                workflow=workflow,
                thread_store=thread_store,
            ),
            input_data={"messages": [{"type": "human", "content": "我最近胃痛"}]},
            context={"stream_mode": ["messages"]},
        )

        second_run = await run_manager.create(thread.thread_id, "workflow_agent")
        bridge.create(second_run.run_id)
        with patch(
            "app.runtime.runs.worker.apply_guardrails",
            new=AsyncMock(
                return_value={
                    "final_text": final_answer,
                    "validation": {"passed": True},
                    "validation_before_rewrite": {"passed": True},
                    "rewritten": False,
                    "allowed_terms": ["胃脘痛"],
                }
            ),
        ):
            await run_agent(
                bridge=bridge,
                run_manager=run_manager,
                thread_store=thread_store,
                record=second_run,
                agent_factory=lambda context: WorkflowAgent(
                    workflow=workflow,
                    thread_store=thread_store,
                ),
                input_data={
                    "messages": [
                        {
                            "type": "human",
                            "content": (
                                "疼痛是间歇性的，吃完饭会好一点，没有其他感觉，"
                                "最近吃维生素B2、维生素C。"
                            ),
                        }
                    ]
                },
                context={"stream_mode": ["messages", "values"]},
            )

        parsed_events = self.parse_events(
            await self.drain_events(bridge, second_run.run_id)
        )
        final_events = [data for event, data in parsed_events if event == "final"]
        value_events = [data for event, data in parsed_events if event == "values"]

        self.assertEqual((await run_manager.get(second_run.run_id)).status, "success")
        self.assertEqual(final_events[-1]["status"], "completed")
        self.assertEqual(final_events[-1]["assistant_message"], final_answer)
        self.assertNotIn(old_question, final_events[-1]["assistant_message"])
        self.assertEqual(
            [
                value
                for value in value_events
                if value.get("needs_clarification") is True
                or old_question in str(value.get("final_text", ""))
            ],
            [],
        )
        self.assertFalse(value_events[-1]["needs_clarification"])
        self.assertEqual(value_events[-1]["final_text"], final_answer)

    async def test_workflow_agent_reads_visible_conversation_from_shared_thread_store(self):
        from app.runtime import state as runtime_state

        original_store = runtime_state.state.thread_store
        try:
            class RecordingWorkflow:
                def __init__(self):
                    self.calls = []

                async def astream(self, *, user_text, conversation, config, stream_mode):
                    self.calls.append(
                        {
                            "user_text": user_text,
                            "conversation": conversation,
                            "config": config,
                            "stream_mode": stream_mode,
                        }
                    )
                    yield ("values", {"messages": []})

            thread_store = ThreadStore()
            runtime_state.state.thread_store = thread_store
            workflow = RecordingWorkflow()
            agent = WorkflowAgent(workflow=workflow)

            thread = await thread_store.create()
            await thread_store.update_values(
                thread.thread_id,
                {
                    "conversation": [
                        {"role": "user", "content": "previous"},
                        {"role": "assistant", "content": "previous answer"},
                    ],
                    "messages": [{"id": "m1", "type": "ai", "content": "hello"}],
                },
            )

            config = {"configurable": {"thread_id": thread.thread_id}}
            events = [
                event
                async for event in agent.astream(
                    {"messages": [{"type": "human", "content": "new question"}]},
                    config=config,
                    stream_mode=["values"],
                )
            ]

            self.assertEqual(
                workflow.calls,
                [
                    {
                        "user_text": "new question",
                        "conversation": [
                            {"role": "user", "content": "previous"},
                            {"role": "assistant", "content": "previous answer"},
                        ],
                        "config": config,
                        "stream_mode": ["values"],
                    }
                ],
            )
            self.assertEqual(events, [("values", {"messages": []})])
        finally:
            runtime_state.state.thread_store = original_store

    async def test_workflow_agent_astream_extracts_input_and_forwards_stream(self):
        class RecordingThreadStore(ThreadStore):
            def __init__(self):
                super().__init__()
                self.value_updates = []

            async def update_values(self, thread_id, values, run_id=None):
                self.value_updates.append(
                    {"thread_id": thread_id, "values": values, "run_id": run_id}
                )
                return await super().update_values(thread_id, values, run_id=run_id)

        class StreamingWorkflow:
            graph = None

            def __init__(self):
                self.calls = []

            async def run(self, *args, **kwargs):
                raise AssertionError("WorkflowAgent.astream should call workflow.astream")

            async def astream(self, *, user_text, conversation, config, stream_mode):
                self.calls.append(
                    {
                        "user_text": user_text,
                        "conversation": conversation,
                        "config": config,
                        "stream_mode": stream_mode,
                    }
                )
                yield (
                    "messages",
                    (
                        AIMessageChunk(content="partial"),
                        {"agent": "workflow_agent"},
                    ),
                )
                yield (
                    "values",
                    {
                        "messages": [],
                        "agent_trace": [{"agent": "StreamingWorkflow"}],
                    },
                )

        thread_store = RecordingThreadStore()
        thread = await thread_store.create()
        await thread_store.update_values(
            thread.thread_id,
            {
                "conversation": [
                    {"role": "user", "content": "previous"},
                    {"role": "assistant", "content": "previous answer"},
                ]
            },
        )
        thread_store.value_updates.clear()

        workflow = StreamingWorkflow()
        agent = WorkflowAgent(workflow=workflow, thread_store=thread_store)
        config = {"configurable": {"thread_id": thread.thread_id}}

        events = [
            event
            async for event in agent.astream(
                {
                    "messages": [
                        {"type": "human", "content": "new question"},
                    ]
                },
                config=config,
                stream_mode=["messages", "values"],
            )
        ]

        self.assertEqual(thread_store.value_updates, [])
        self.assertEqual(
            workflow.calls,
            [
                {
                    "user_text": "new question",
                    "conversation": [
                        {"role": "user", "content": "previous"},
                        {"role": "assistant", "content": "previous answer"},
                    ],
                    "config": config,
                    "stream_mode": ["messages", "values"],
                }
            ],
        )
        self.assertEqual([event[0] for event in events], ["messages", "values"])


if __name__ == "__main__":
    unittest.main()
