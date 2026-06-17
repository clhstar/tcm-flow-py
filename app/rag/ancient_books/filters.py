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
    r"(?:[（(][^）)]*(?:毫升|钱|两|分|枚|升|合|斗|撮|铢|斤|克|片|粒|盏)"
    r"[^）)]*[）)])"
    r"|(?:\d+(?:\.\d+)?|[一二三四五六七八九十百半]+)\s*"
    r"(?:毫升|钱|两|分|枚|升|合|斗|撮|铢|斤|克|片|粒|盏)"
)
_FORMULA_NAME_PATTERN = re.compile(
    r"^[\u3400-\u9fff]{1,12}(?:汤|散|丸|饮|膏|丹)(?:方)?$"
)
_MODIFIED_FORMULA_PATTERN = re.compile(
    r"^[\u3400-\u9fff]{1,12}(?:汤|散|丸|膏|丹)(?:方)?"
    r"(?:加减|加味|主之|治|服|合用|化裁)"
)
_PREPARATION_PATTERN = re.compile(
    r"为末|为丸|水煎|主之|治|服|煎|煮|熬|炙|炒|焙|研|捣|浸|泡|"
    r"上[一二三四五六七八九十百\d]+味|"
    r"(?:空心|食前|食后|温水|白汤|米汤|姜汤|酒|姜汁)下|送下|下之|"
    r"(?:[一二三四五六七八九十百\d]+上)?[一二三四五六七八九十百\d]+下"
)
_DIAGNOSTIC_CUES = ("因", "者", "恶", "喜", "脉", "证", "症")
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
    "天麻",
    "钩藤",
    "葛根",
    "升麻",
    "荆芥",
    "前胡",
    "紫苏",
    "藿香",
    "佩兰",
    "金银花",
    "蒲公英",
    "板蓝根",
    "玄参",
    "木通",
    "通草",
    "滑石",
    "薏苡仁",
    "五味子",
    "山茱萸",
    "杜仲",
    "续断",
    "肉桂",
    "吴茱萸",
    "丁香",
    "小茴香",
    "郁金",
    "延胡索",
    "丹参",
    "益母草",
    "地龙",
    "全蝎",
    "蜈蚣",
    "僵蚕",
    "磁石",
    "朱砂",
    "琥珀",
    "瓜蒌",
    "竹茹",
    "旋覆花",
    "代赭石",
    "款冬花",
    "紫菀",
    "百部",
    "白前",
    "苏子",
    "莱菔子",
    "麦芽",
    "山楂",
    "神曲",
    "鸡内金",
    "阿胶",
    "何首乌",
    "白扁豆",
    "山药",
    "莲子",
    "芡实",
)


def _remove_excluded_subtitles(text: str) -> str:
    retained: list[str] = []
    cursor = 0
    excluded = False

    for match in _SUBTITLE_PATTERN.finditer(text):
        if not excluded:
            retained.append(text[cursor : match.start()])
        title = match.group("title").strip()
        excluded = any(keyword in title for keyword in _EXCLUDED_SUBTITLES)
        cursor = match.end()
        if not excluded:
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1

    if not excluded:
        retained.append(text[cursor:])
    return "".join(retained)


def _herb_count(text: str) -> int:
    return sum(herb in text for herb in _COMMON_HERBS)


def _is_unsafe_clause(clause: str) -> bool:
    candidate = clause.strip(" \t\r\n，,、：:")
    if not candidate:
        return True
    if _DOSE_PATTERN.search(candidate) or _PREPARATION_PATTERN.search(candidate):
        return True
    if _MODIFIED_FORMULA_PATTERN.search(candidate):
        return True
    if _FORMULA_NAME_PATTERN.fullmatch(candidate) and not any(
        cue in candidate for cue in _DIAGNOSTIC_CUES
    ):
        return True
    return _herb_count(candidate) >= 2


def _filter_sentence(body: str, terminator: str) -> str:
    clauses = re.split(r"[，,]", body)
    sentence_herb_count = sum(_herb_count(clause) for clause in clauses)
    unsafe = [
        _is_unsafe_clause(clause)
        or (sentence_herb_count >= 2 and _herb_count(clause) > 0)
        for clause in clauses
    ]
    if not any(unsafe):
        return body + terminator
    retained_indices = [
        index for index, is_unsafe in enumerate(unsafe) if not is_unsafe
    ]
    if not retained_indices:
        return ""

    separators = re.findall(r"[，,]", body)
    retained = clauses[retained_indices[0]]
    for index in retained_indices[1:]:
        retained += separators[index - 1] + clauses[index]
    return retained + terminator


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
