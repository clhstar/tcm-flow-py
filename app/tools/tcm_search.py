from langchain.tools import tool


TCM_KNOWLEDGE = {
    "胃胀": "胃胀常见于饮食不节、情志不畅、脾胃虚弱等情况。中医问诊应关注食欲、大便、嗳气、反酸、舌苔等信息。",
    "失眠": "失眠可与心脾两虚、肝郁化火、心肾不交等相关。应追问入睡困难、易醒、多梦、情绪、心悸等表现。",
    "头痛": "头痛需要关注部位、性质、持续时间、诱因、是否伴随发热、呕吐、肢体麻木等危险信号。",
}


@tool("search_tcm_knowledge")
def search_tcm_knowledge(query: str) -> str:
    """检索简化版中医知识库，返回和用户问题相关的知识。"""
    for key, value in TCM_KNOWLEDGE.items():
        if key in query:
            return value
    return "未检索到精确知识。请结合用户症状继续追问，不要直接下诊断。"