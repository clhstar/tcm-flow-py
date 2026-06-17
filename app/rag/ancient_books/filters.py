import re


_SUBTITLE_PATTERN = re.compile(r"\\x(?P<title>[^\\\r\n]+)\\x")
_EXCLUDED_SUBTITLES = {
    "用药",
    "方剂",
    "选方",
    "附方",
    "快捷方式法",
    "制法",
    "服法",
    "煎服法",
}
_SENTENCE_PATTERN = re.compile(r"([^。！？!?；;\n]*)([。！？!?；;\n]+|$)")
_DOSE_PATTERN = re.compile(
    r"(?:[（(][^）)]*(?:钱|两|分|枚)[^）)]*[）)])"
    r"|(?:\d+(?:\.\d+)?|[一二三四五六七八九十百半]+)\s*(?:钱|两|分|枚)"
)
_FORMULA_NAME_PATTERN = re.compile(
    r"^[\u3400-\u9fff]{1,12}(?:汤|散|丸|饮|膏|丹)(?:方)?$"
)
_PREPARATION_PATTERN = re.compile(
    r"为末|为丸|水煎|主之|治|服|煎|"
    r"(?:空心|食前|食后|温水|白汤|米汤|姜汤|酒|姜汁)下|送下|下之"
)
_DIAGNOSTIC_CUES = ("因", "者", "痛", "恶", "喜", "脉", "证", "症")
_COMMON_HERBS = (
    "人参",
    "党参",
    "黄芪",
    "白术",
    "茯苓",
    "甘草",
    "当归",
    "川芎",
    "白芍",
    "赤芍",
    "熟地",
    "生地",
    "黄芩",
    "黄连",
    "黄柏",
    "栀子",
    "连翘",
    "丹皮",
    "牡丹皮",
    "桑叶",
    "菊花",
    "薄荷",
    "柴胡",
    "白芷",
    "羌活",
    "独活",
    "防风",
    "麻黄",
    "桂枝",
    "细辛",
    "半夏",
    "陈皮",
    "枳实",
    "枳壳",
    "厚朴",
    "大黄",
    "芒硝",
    "附子",
    "干姜",
    "生姜",
    "石膏",
    "知母",
    "麦冬",
    "天冬",
    "贝母",
    "桔梗",
    "杏仁",
    "桃仁",
    "红花",
    "牛膝",
    "木香",
    "砂仁",
    "香附",
    "苍术",
    "泽泻",
    "猪苓",
    "车前子",
    "龙骨",
    "牡蛎",
    "酸枣仁",
    "远志",
)


def _remove_excluded_subtitles(text: str) -> str:
    retained: list[str] = []
    cursor = 0
    excluded = False

    for match in _SUBTITLE_PATTERN.finditer(text):
        if not excluded:
            retained.append(text[cursor : match.start()])
        excluded = match.group("title").strip() in _EXCLUDED_SUBTITLES
        cursor = match.end()

    if not excluded:
        retained.append(text[cursor:])
    return "".join(retained)


def _is_unsafe_clause(clause: str) -> bool:
    candidate = clause.strip(" \t\r\n，,、：:")
    if not candidate:
        return True
    if _DOSE_PATTERN.search(candidate) or _PREPARATION_PATTERN.search(candidate):
        return True
    if _FORMULA_NAME_PATTERN.fullmatch(candidate) and not any(
        cue in candidate for cue in _DIAGNOSTIC_CUES
    ):
        return True
    herb_count = sum(herb in candidate for herb in _COMMON_HERBS)
    return herb_count >= 2


def _filter_sentence(body: str, terminator: str) -> str:
    clauses = re.split(r"[，,]", body)
    unsafe = [_is_unsafe_clause(clause) for clause in clauses]
    if not any(unsafe):
        return body + terminator
    retained = [
        clause.strip()
        for clause, is_unsafe in zip(clauses, unsafe)
        if not is_unsafe
    ]
    if not retained:
        return ""
    punctuation = terminator if terminator else ""
    return "，".join(retained) + punctuation


def filter_retrievable_text(text: str) -> str:
    """Remove formula, medication, dosage, and preparation content."""
    if not text or not text.strip():
        return ""

    safe_regions = _remove_excluded_subtitles(text)
    filtered: list[str] = []
    for match in _SENTENCE_PATTERN.finditer(safe_regions):
        body, terminator = match.groups()
        if not body and not terminator:
            continue
        sentence = _filter_sentence(body, terminator)
        if sentence:
            filtered.append(sentence)
    return "".join(filtered).strip()
