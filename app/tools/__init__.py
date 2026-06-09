from app.tools.clarification import ask_clarification
from app.tools.present_files import present_files
from app.tools.tcm_search import search_tcm_knowledge


def get_available_tools():
    """返回所有可用的Agent工具列表"""
    return [
        ask_clarification,
        present_files,
        search_tcm_knowledge,
    ]