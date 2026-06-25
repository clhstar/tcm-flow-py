import unittest
import unittest.mock

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
                    chief_complaint="stomach distension",
                    known_facts=KnownFacts(
                        duration="two weeks",
                        triggers=["worse after oily food"],
                        associated_symptoms=["belching"],
                    ),
                    information_sufficiency="sufficient",
                )
            ],
            SyndromeAnalysis: [
                SyndromeAnalysis(
                    possible_patterns=[
                        PatternCandidate(
                            term="Food stagnation",
                            supporting_evidence=["E1"],
                            confidence="medium",
                            reason="The user reports bloating after oily food.",
                        )
                    ],
                    not_enough_for_diagnosis=True,
                    need_more_info=["tongue appearance"],
                )
            ],
            AnswerDraft: [
                AnswerDraft(
                    draft_answer=(
                        "This may be related to food stagnation, but it is not "
                        "enough for a diagnosis. [E1]"
                    )
                )
            ],
            SafetyReview: [
                SafetyReview(final_safety_level="low", rewrite_required=False)
            ],
        }
        self.invocations = []
        self.schema_queue = list(self.responses_by_schema)

    def with_structured_output(self, schema=None, *, method, strict=None):
        schema = schema or self.schema_queue.pop(0)
        return FakeStructuredRunnable(self, schema)


async def fake_retriever(query: str, mode: str) -> str:
    return (
        "Search status: ok\n"
        "Search mode: hybrid_parent\n\n"
        "[E1]\n"
        "Role: syndrome_pattern\n"
        "Text: Food stagnation may present with gastric distension.\n"
        "Source: Test source\n"
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
            user_text="stomach distension for two weeks, worse after oily food",
            conversation=[],
        )

        self.assertFalse(result.needs_clarification)
        self.assertIn("[E1]", result.final_text)
        self.assertEqual(
            [event["agent"] for event in result.agent_trace],
            [
                "InquiryAgent",
                "EvidenceAgent",
                "SyndromeAgent",
                "AnswerAgent",
                "SafetyAgent",
            ],
        )

    async def test_workflow_run_returns_only_current_messages_with_checkpointer(self):
        from langgraph.checkpoint.memory import InMemorySaver

        from app.agents.workflow_agent.components.evidence import EvidenceAgent

        model = FakeWorkflowModel()
        for responses in model.responses_by_schema.values():
            responses.append(responses[0].model_copy(deep=True))
        workflow = TCMWorkflow(
            model=model,
            evidence_agent=EvidenceAgent(retriever=fake_retriever),
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "thread-1"}}

        first = await workflow.run(
            user_text="stomach distension for two weeks",
            conversation=[],
            config=config,
        )
        second = await workflow.run(
            user_text="stomach distension continues",
            conversation=[],
            config=config,
        )
        snapshot = await workflow.graph.aget_state(config)

        self.assertEqual(len(first.messages), 3)
        self.assertEqual(len(second.messages), 3)
        self.assertEqual(len(first.agent_trace), 5)
        self.assertEqual(len(second.agent_trace), 5)
        self.assertEqual(len(snapshot.values["messages"]), 8)
        self.assertTrue(
            all(getattr(message, "type", "") != "human" for message in second.messages)
        )


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


class WorkflowAgentStateDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_workflow_agent_state_methods_delegate_to_graph_when_available(self):
        from types import SimpleNamespace

        from app.agents.workflow_agent.agent import WorkflowAgent

        calls = []

        class FakeGraph:
            async def aget_state(self, config):
                calls.append(("aget_state", config))
                return SimpleNamespace(
                    values={"messages": [{"content": "from graph"}]},
                    next=(),
                )

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


class WorkflowGraphRouteTests(unittest.TestCase):
    def test_route_after_inquiry_pauses_only_when_inquiry_requires_clarification(self):
        from app.agents.workflow_agent.graph import route_after_inquiry

        self.assertEqual(
            route_after_inquiry(
                {
                    "inquiry": InquiryState(
                        chief_complaint="stomach distension",
                        information_sufficiency="insufficient",
                        clarification_questions=["How long has it lasted?"],
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
                        chief_complaint="stomach distension",
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
            route_after_initial_safety({"safety": SafetyReview(rewrite_required=True)}),
            "answer_rewrite",
        )
        self.assertEqual(
            route_after_initial_safety({"safety": SafetyReview(rewrite_required=False)}),
            "finalize",
        )
        self.assertEqual(
            route_after_rewrite_safety({"safety": SafetyReview(rewrite_required=True)}),
            "safe_fallback",
        )
