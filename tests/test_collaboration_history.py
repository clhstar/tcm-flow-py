import unittest

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.workflow_agent.agent import WorkflowAgent
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


class RecordingWorkflow:
    def __init__(self):
        self.conversation = None

    async def astream(
        self,
        *,
        user_text,
        conversation,
        config,
        stream_mode,
    ):
        self.conversation = conversation
        yield "values", {"messages": [], "agent_trace": []}


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
        self.assertEqual(assistant_turn.get("run_id"), run.run_id)
        self.assertEqual(
            [step["agent"] for step in assistant_turn.get("agent_trace", [])],
            ["IntentAgent", "InquiryAgent"],
        )


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
                {"configurable": {"thread_id": thread.thread_id}},
                ["values"],
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


if __name__ == "__main__":
    unittest.main()
