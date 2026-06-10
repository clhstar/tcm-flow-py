import re
from datetime import datetime
from typing import Any

# 注意：
# 这些关键词不是澄清中断的主机制。
# 主机制是 ask_clarification 工具调用。
# 这里仅用于兜底：当模型没有调用工具，
# 但 final answer 明显包含正式追问时，
# Runtime 将其转换为 clarification interrupt。
FORMAL_CLARIFICATION_KEYWORDS = [
    "请您先补充",
    "请先补充",
    "需要您补充",
    "还需要了解",
    "我需要先向您追问",
    "请先告诉我",
    "请告诉我",
    "为了更准确",
    "为了更全面",
    "尚需补充",
    "缺失的关键信息",
    "进一步了解",
    "确认一下",
    "补充一下",
]


CLARIFICATION_STRUCTURE_PHRASES = [
    "以下信息",
    "几个问题",
    "关键信息",
    "进一步了解",
    "确认一下",
    "补充一下",
    "请回答",
    "请提供",
]


FIELD_RULES = [
    ("持续时间", ["持续多久", "多久", "几天", "几周", "几个月"]),
    ("大便情况", ["大便", "便秘", "腹泻", "不成形"]),
    ("反酸烧心", ["反酸", "烧心"]),
    ("疼痛情况", ["腹痛", "胃痛", "疼痛"]),
    ("饮食情况", ["饮食", "过饱", "油腻", "生冷", "辛辣"]),
    ("食欲情况", ["食欲", "胃口"]),
    ("情绪压力", ["情绪", "压力", "焦虑", "紧张"]),
    ("风险信号", ["呕吐", "黑便", "消瘦", "体重"]),
    ("睡眠情况", ["睡眠", "失眠", "多梦"]),
    ("口干口苦", ["口干", "口苦"]),
    ("小便情况", ["小便", "尿频", "尿急", "尿黄"]),
    ("既往病史", ["既往", "病史", "慢性病"]),
    ("用药情况", ["用药", "药物", "正在吃"]),
]


DEFAULT_FIELD_QUESTIONS = {
    "持续时间": "这种情况持续多久了？",
    "大便情况": "大便情况如何？是否有便秘、腹泻或大便不成形？",
    "反酸烧心": "是否伴有反酸或烧心？",
    "疼痛情况": "是否有明显腹痛或胃痛？",
    "饮食情况": "近期饮食是否过饱、油腻、生冷或辛辣？",
    "食欲情况": "食欲情况如何？",
    "情绪压力": "最近情绪压力是否较大？",
    "风险信号": "是否有呕吐、黑便、消瘦或体重明显下降等情况？",
    "睡眠情况": "睡眠情况如何？",
    "口干口苦": "是否有口干或口苦？",
    "小便情况": "小便情况是否正常？",
    "既往病史": "是否有既往病史或慢性病？",
    "用药情况": "近期是否正在用药？",
}


def extract_latest_clarification_question(messages: list[dict[str, Any]]) -> str:
    """
    主路径：只在 ask_clarification 工具执行完成后中断。

    """
    if not messages:
        return ""

    latest = messages[-1]

    if latest.get("type") == "tool" and latest.get("name") == "ask_clarification":
        content = latest.get("content", "")

        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def extract_questions_from_text(text: str) -> list[str]:
    """
    从文本中提取问句。
    """
    if not text:
        return []

    questions: list[str] = []

    matches = re.findall(r"[^。！？\n]*[？?]", text)

    for item in matches:
        question = item.strip()
        question = re.sub(r"^[\-\*\d\.\、\s]+", "", question).strip()

        if question:
            questions.append(question)

    result = []
    seen = set()

    for question in questions:
        if question not in seen:
            seen.add(question)
            result.append(question)

    return result


def extract_required_fields(text: str) -> list[str]:
    """
    根据问题文本提取需要补充的字段。

    注意：
    这只是为了结构化 pending_clarification，
    不是决定是否中断的唯一依据。
    """
    if not text:
        return []

    fields: list[str] = []

    for field, keywords in FIELD_RULES:
        if any(keyword in text for keyword in keywords):
            fields.append(field)

    return list(dict.fromkeys(fields))


def build_limited_question(
    original_question: str,
    required_fields: list[str],
    max_questions: int = 3,
) -> str:
    """
    每次最多追问 max_questions 个关键问题。
    """
    selected: list[str] = []

    if required_fields:
        selected = [
            DEFAULT_FIELD_QUESTIONS[field]
            for field in required_fields
            if field in DEFAULT_FIELD_QUESTIONS
        ][:max_questions]

    if not selected:
        questions = extract_questions_from_text(original_question)
        selected = questions[:max_questions]

    if not selected and original_question.strip():
        selected = [original_question.strip()]

    lines = ["为了更准确地帮您分析，请先补充以下关键信息："]

    for index, question in enumerate(selected, start=1):
        lines.append(f"{index}. {question}")

    return "\n".join(lines)


def build_clarification_payload(
    question: str,
    run_id: str,
    source: str = "ask_clarification",
    max_questions: int = 3,
) -> dict[str, Any]:
    """
    构造标准 pending_clarification。
    """
    required_fields = extract_required_fields(question)

    limited_question = build_limited_question(
        original_question=question,
        required_fields=required_fields,
        max_questions=max_questions,
    )

    return {
        "type": "clarification",
        "run_id": run_id,
        "question": limited_question,
        "required_fields": required_fields[:max_questions],
        "source": source,
        "created_at": datetime.utcnow().isoformat(),
        "resume_count": 0,
    }


def should_convert_final_to_clarification(final_text: str) -> bool:
    """
    兜底路径：
    判断 final answer 是否实际上是在正式追问。

    三层判断：
    1. 命中正式追问关键词；
    2. 存在多个问句；
    3. 出现多个问诊字段。
    """
    if not final_text:
        return False

    questions = extract_questions_from_text(final_text)
    required_fields = extract_required_fields(final_text)

    hit_keyword = any(
        keyword in final_text for keyword in FORMAL_CLARIFICATION_KEYWORDS
    )

    has_structure = any(
        phrase in final_text for phrase in CLARIFICATION_STRUCTURE_PHRASES
    )

    has_many_questions = len(questions) >= 3
    has_medical_missing_fields = len(required_fields) >= 2

    return (
        (hit_keyword and len(questions) >= 1)
        or (has_many_questions and has_medical_missing_fields)
        or (has_structure and has_medical_missing_fields)
    )


def build_clarification_from_final(
    final_text: str,
    run_id: str,
) -> dict[str, Any] | None:
    """
    当模型没有调用 ask_clarification，
    但 final answer 明显是在正式追问时，
    Runtime 将 final 转换为 clarification。
    """
    if not should_convert_final_to_clarification(final_text):
        return None

    questions = extract_questions_from_text(final_text)

    if questions:
        question_text = "\n".join(questions)
    else:
        question_text = final_text

    return build_clarification_payload(
        question=question_text,
        run_id=run_id,
        source="final_fallback",
        max_questions=3,
    )
