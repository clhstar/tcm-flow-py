from langchain.tools import tool


@tool("ask_clarification", return_direct=True)
def ask_clarification(question: str) -> str:
    """
    当用户信息不足、症状描述不清楚、需要补充病史或需要确认风险时调用。

    输入 question 应该是一个明确的问题，用于向用户补充询问。
    调用该工具后，本轮对话应该暂停，等待用户回答。
    """
    return question