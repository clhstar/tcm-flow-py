from app.tools.clarification import ask_clarification
from app.tools.present_files import present_files
from app.tools.tcm_search import search_tcm_knowledge


def get_available_tools():
    return [
        ask_clarification,
        present_files,
        search_tcm_knowledge,
    ]