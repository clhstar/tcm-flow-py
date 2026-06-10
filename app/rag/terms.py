TCM_TERMS = [
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


QUERY_EXPANSIONS = {
    "胃胀": [
        "胃脘胀满",
        "饭后加重",
        "嗳气",
        "反酸",
        "烧心",
        "恶心",
        "大便不成形",
        "饮食不节",
        "情志不畅",
        "脾胃虚弱",
        "气机不畅",
        "问诊要点",
        "日常调护",
    ],
    "失眠": [
        "入睡困难",
        "易醒",
        "多梦",
        "早醒",
        "心悸",
        "健忘",
        "心脾两虚",
        "肝郁化火",
        "心肾不交",
        "痰热扰心",
        "问诊要点",
    ],
    "头痛": [
        "头痛部位",
        "头痛性质",
        "持续时间",
        "诱因",
        "呕吐",
        "意识异常",
        "肢体无力",
        "言语不清",
        "危险信号",
    ],
}


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
    expansions = []

    for keyword, extra_terms in QUERY_EXPANSIONS.items():
        if keyword in query:
            expansions.extend(extra_terms)

    # 如果用户没明确提到知识库里的标准词，也可以根据常见表达做简单补充
    if "饭后" in query and "胃胀" not in query:
        expansions.extend(QUERY_EXPANSIONS.get("胃胀", []))

    if "睡不着" in query and "失眠" not in query:
        expansions.extend(QUERY_EXPANSIONS.get("失眠", []))

    all_terms = deduplicate_keep_order([query] + expansions)

    return " ".join(all_terms)


def detect_topic(query: str) -> str | None:
    """
    根据 query 粗略识别主题。
    后面 metadata 过滤可以用。
    """
    for topic in QUERY_EXPANSIONS.keys():
        if topic in query:
            return topic

    if "饭后" in query or "嗳气" in query or "反酸" in query:
        return "胃胀"

    if "睡不着" in query or "多梦" in query or "易醒" in query:
        return "失眠"

    return None