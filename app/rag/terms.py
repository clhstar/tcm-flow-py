from app.rag.ancient_books.query import (
    SEARCH_ALIASES,
    detect_chief_symptom,
    rewrite_query,
)


TCM_TERMS = list(
    dict.fromkeys(
        term
        for terms in SEARCH_ALIASES.values()
        for term in terms
    )
) + [
    # 症状
    "胃胀",
    "胃脘胀满",
    "嗳气",
    "反酸",
    "烧心",
    "恶心",
    "便秘",
    "腹泻",
    "大便不成形",
    "食欲不振",
    "入睡困难",
    "易醒",
    "多梦",
    "头痛",

    # 病因病机 / 证候相关
    "饮食不节",
    "情志不畅",
    "脾胃虚弱",
    "气机不畅",
    "心脾两虚",
    "肝郁化火",
    "心肾不交",
    "痰热扰心",

    # 危险信号
    "突发剧烈头痛",
    "意识异常",
    "肢体无力",
    "呕吐",
    "黑便",
    "消瘦",
    "持续高热",
]


QUERY_EXPANSIONS = SEARCH_ALIASES


def deduplicate_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)

    return result


def extract_terms(text: str) -> list[str]:
    """
    从文本中抽取命中的中医术语。
    第一版用词表匹配，后面可以升级成 NER / 医学术语抽取模型。
    """
    if not text:
        return []

    matched = []

    for term in TCM_TERMS:
        if term in text:
            matched.append(term)

    return deduplicate_keep_order(matched)


def rewrite_tcm_query(query: str) -> str:
    """
    将用户口语化问题扩展成更适合中医知识库检索的 query。
    """
    return rewrite_query(query)


def detect_topic(query: str) -> str | None:
    """
    根据 query 粗略识别主题。
    后面 metadata 过滤可以用。
    """
    return detect_chief_symptom(query)
