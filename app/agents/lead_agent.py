import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.prompt import SYSTEM_PROMPT
from app.tools import get_available_tools

load_dotenv()

_checkpointer = InMemorySaver()


def make_lead_agent(context: dict | None = None):
    context = context or {}

    model_name = context.get("model_name") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        temperature=0.3,
    )

    tools = get_available_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )

    return agent