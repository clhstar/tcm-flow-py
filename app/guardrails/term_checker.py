from app.rag.terms import TCM_TERMS

EXTRA_TCM_TERMS = [
    # 常见证候 / 病机，主要用于检测模型是否乱引入
    "肝胃不和",
    "食积内停",
    "寒湿困脾",
    "脾虚湿困",
    "胃气上逆",
    "肝郁气滞",
    "肝气犯胃",
    "湿热中阻",
    "胃阴不足",
    "脾阳虚",
    "肾阳虚",
    "肾阴虚",
    "气血不足",
    "气滞血瘀",
    "痰湿内阻",
    # 治法类
    "健脾和胃",
    "理气和中",
    "疏肝理气",
    "消食导滞",
    "温中散寒",
    "清热化湿",
    "滋阴养胃",
    "胃气壅滞",
    "胃气不降",
    "胃气当降不降",
    "胃失和降",
    "中焦气滞",
    "脾失健运",
]


def get_all_tcm_terms() -> list[str]:
    """
    合并 RAG 术语表和额外校验术语表。
    """
    terms = list(TCM_TERMS) + EXTRA_TCM_TERMS

    seen = set()
    result = []

    # 长词优先，避免“胃胀”先匹配影响“胃脘胀满”这类长词
    for term in sorted(terms, key=len, reverse=True):
        if term not in seen:
            seen.add(term)
            result.append(term)

    return result


def extract_terms(text: str) -> list[str]:
    """
    从文本中抽取中医专业术语。
    第一版使用词表匹配，后续可以替换为 NER 模型。
    """
    if not text:
        return []

    matched = []

    for term in get_all_tcm_terms():
        if term in text:
            matched.append(term)

    return matched


def normalize_terms(terms: list[str]) -> list[str]:
    """
    去重并保持顺序。
    """
    seen = set()
    result = []

    for term in terms:
        if term and term not in seen:
            seen.add(term)
            result.append(term)

    return result
