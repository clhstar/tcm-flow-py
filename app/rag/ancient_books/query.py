ALIASES = {
    "头痛": ["突发剧烈头痛", "头疼", "头痛"],
    "眩晕": ["天旋地转", "头晕", "眩晕"],
    "咳嗽": ["咳嗽", "咳"],
    "喘促": ["呼吸困难", "气喘", "喘促", "喘"],
    "心悸": ["怦怦跳", "心慌", "心悸"],
    "不寐": ["睡不着", "容易醒", "易醒", "失眠", "不寐"],
    "胃脘痛": ["胃脘痛", "胃疼", "心口痛"],
    "腹痛": ["肚子痛", "腹痛"],
    "泄泻": ["拉肚子", "腹泻", "泄泻"],
    "便秘": ["大便干结", "排便困难", "便秘"],
}

SEARCH_ALIASES = {
    "头痛": ["头痛", "头风"],
    "眩晕": ["眩晕", "眩运", "头眩"],
    "咳嗽": ["咳嗽", "咳逆"],
    "喘促": ["喘促", "喘逆", "上气", "短气"],
    "心悸": ["心悸", "惊悸", "怔忡", "怔仲"],
    "不寐": ["不寐", "不得卧"],
    "胃脘痛": ["胃脘痛", "胃脘", "心痛"],
    "腹痛": ["腹痛", "腹满", "心腹痛"],
    "泄泻": ["泄泻", "下利"],
    "便秘": ["便秘", "秘结", "大便不通"],
}


def detect_chief_symptom(query: str) -> str | None:
    aliases = sorted(
        (
            (alias, symptom)
            for symptom, symptom_aliases in ALIASES.items()
            for alias in symptom_aliases
        ),
        key=lambda item: -len(item[0]),
    )
    for alias, symptom in aliases:
        if alias in query:
            return symptom
    return None


def rewrite_query(query: str) -> str:
    symptom = detect_chief_symptom(query)
    terms = [query]
    if symptom:
        terms.extend(SEARCH_ALIASES[symptom])
    return " ".join(dict.fromkeys(term for term in terms if term))
