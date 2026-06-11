from langchain.tools import tool


@tool("ask_clarification", return_direct=True)
def ask_clarification(questions: list[str]) -> str:
    """
    信息不足时向用户提出澄清问题。

    questions 必须包含 1 到 3 个清晰、独立的问题。
    不要重复询问用户已经回答过的信息。
    """
    return questions
