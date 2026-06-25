from collections.abc import Sequence
from typing import Any

from app.agents.workflow_agent.components.base import StructuredWorkflowComponent
from app.agents.workflow_agent.components.inquiry import conversation_text
from app.agents.workflow_agent.models import IntentState
from app.agents.workflow_agent.prompts import INTENT_SYSTEM_PROMPT

CAUSE_INTENT_TERMS = (
    "什么原因",
    "为什么",
    "导致",
    "怎么回事",
    "咋回事",
    "什么引起",
    "为啥",
)

CLASSIC_INTENT_TERMS = (
    "伤寒论",
    "金匮要略",
    "黄帝内经",
    "温病条辨",
    "景岳全书",
    "条文",
    "原文",
    "出处",
    "出自",
)

FORMULA_INTENT_TERMS = (
    "方剂",
    "方子",
    "经方",
    "组成",
    "功效",
    "主治",
    "配伍",
)

GENERAL_TCM_TERMS = (
    "中医",
    "证候",
    "病机",
    "脾胃",
    "气血",
    "阴阳",
    "寒热",
    "虚实",
    "湿热",
    "痰湿",
    "气滞",
)

RISK_SIGNAL_TERMS = (
    "胸痛",
    "呼吸困难",
    "喘不上气",
    "意识不清",
    "意识异常",
    "昏迷",
    "剧烈头痛",
    "持续高热",
    "高热不退",
    "肢体无力",
    "黑便",
    "呕血",
    "便血",
    "明显出血",
    "持续加重腹痛",
    "反复呕吐",
)

GREETING_TERMS = (
    "你好",
    "您好",
    "hello",
    "hi",
    "在吗",
)

PERSONAL_HEALTH_MARKERS = (
    "我",
    "我的",
    "本人",
    "自己",
    "最近",
    "这几天",
    "这两天",
    "怎么办",
    "不舒服",
    "难受",
    "疼",
    "痛",
    "胀",
    "拉肚子",
    "便秘",
    "失眠",
    "咳嗽",
)


def _is_greeting_only(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in GREETING_TERMS


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def detect_rule_signals(user_text: str) -> dict[str, Any]:
    """
    高确定性规则信号。
    注意：这些规则不是最终分类结果，只是给 LLM 和后处理使用的辅助信号。
    """
    text = user_text.strip()

    risk_terms = _matched_terms(text, RISK_SIGNAL_TERMS)
    classic_terms = _matched_terms(text, CLASSIC_INTENT_TERMS)
    formula_terms = _matched_terms(text, FORMULA_INTENT_TERMS)
    general_tcm_terms = _matched_terms(text, GENERAL_TCM_TERMS)
    cause_terms = _matched_terms(text, CAUSE_INTENT_TERMS)
    personal_terms = _matched_terms(text, PERSONAL_HEALTH_MARKERS)

    return {
        "has_risk_signal": bool(risk_terms),
        "risk_terms": risk_terms,
        "has_classic_intent": bool(classic_terms),
        "classic_terms": classic_terms,
        "has_formula_intent": bool(formula_terms),
        "formula_terms": formula_terms,
        "has_general_tcm_terms": bool(general_tcm_terms),
        "general_tcm_terms": general_tcm_terms,
        "has_cause_intent": bool(cause_terms),
        "cause_terms": cause_terms,
        "has_personal_health_marker": bool(personal_terms),
        "personal_health_terms": personal_terms,
        "is_greeting_only": _is_greeting_only(text),
    }

def _merge_secondary(
    existing: Sequence[str],
    *items: str,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in [*existing, *items]:
        if item and item not in seen:
            seen.add(item)
            result.append(item)

    return result

def _apply_rule_policy(
    intent: IntentState,
    *,
    user_text: str,
    rule_signals: dict[str, Any],
) -> IntentState:
    """
    用确定性规则修正 LLM 分类结果。

    原则：
    1. 危险信号优先级最高。
    2. 问候可直接返回。
    3. 古籍/方剂/一般知识可直接进入 evidence。
    4. 原因类和个人症状咨询进入 inquiry。
    """
    if rule_signals["has_risk_signal"]:
        return intent.model_copy(
            update={
                "primary_intent": "high_risk",
                "confidence": "high",
                "is_tcm_domain_query": True,
                "is_personal_health_query": True,
                "has_risk_signal": True,
                "risk_flags": rule_signals["risk_terms"],
                "requires_retrieval": False,
                "should_enter_inquiry": True,
                "route_hint": "inquiry",
                "reason": "检测到危险信号，应优先进入问诊整理与安全审查流程。",
                "rule_signals": rule_signals,
            }
        )

    if rule_signals["is_greeting_only"]:
        return intent.model_copy(
            update={
                "primary_intent": "greeting_or_chitchat",
                "confidence": "high",
                "is_tcm_domain_query": False,
                "is_personal_health_query": False,
                "requires_retrieval": False,
                "should_enter_inquiry": False,
                "route_hint": "direct_response",
                "direct_response": "你好，我可以帮你做中医知识解释、症状信息整理和基于证据的健康咨询。你可以直接描述想了解的问题。",
                "reason": "用户输入为问候语，不需要进入中医问答 workflow。",
                "rule_signals": rule_signals,
            }
        )

    if rule_signals["has_classic_intent"]:
        return intent.model_copy(
            update={
                "primary_intent": "classic_explanation",
                "secondary_intents": _merge_secondary(
                    intent.secondary_intents,
                    "general_tcm_knowledge",
                ),
                "confidence": "high",
                "is_tcm_domain_query": True,
                "is_personal_health_query": False,
                "requires_retrieval": True,
                "should_enter_inquiry": False,
                "route_hint": "evidence",
                "reason": "检测到古籍、条文或出处类表达，可直接进入证据检索。",
                "rule_signals": rule_signals,
            }
        )

    if rule_signals["has_formula_intent"]:
        return intent.model_copy(
            update={
                "primary_intent": "formula_knowledge",
                "confidence": "high",
                "is_tcm_domain_query": True,
                "is_personal_health_query": False,
                "requires_retrieval": True,
                "should_enter_inquiry": False,
                "route_hint": "evidence",
                "reason": "检测到方剂知识类表达，可直接进入证据检索。",
                "rule_signals": rule_signals,
            }
        )

    if rule_signals["has_cause_intent"]:
        is_personal = (
            intent.is_personal_health_query
            or rule_signals["has_personal_health_marker"]
        )
        return intent.model_copy(
            update={
                "primary_intent": "cause_explanation",
                "secondary_intents": _merge_secondary(
                    intent.secondary_intents,
                    "symptom_consultation" if is_personal else "general_tcm_knowledge",
                ),
                "confidence": "high",
                "is_tcm_domain_query": True,
                "is_personal_health_query": is_personal,
                "requires_retrieval": True,
                "should_enter_inquiry": is_personal,
                "route_hint": "inquiry" if is_personal else "evidence",
                "reason": (
                    "检测到原因类问题；若属于个人症状咨询，则进入问诊整理，"
                    "否则可直接进入证据检索。"
                ),
                "rule_signals": rule_signals,
            }
        )

    if (
        rule_signals["has_general_tcm_terms"]
        and not rule_signals["has_personal_health_marker"]
    ):
        return intent.model_copy(
            update={
                "primary_intent": (
                    intent.primary_intent
                    if intent.primary_intent != "unknown"
                    else "general_tcm_knowledge"
                ),
                "confidence": (
                    intent.confidence if intent.confidence != "low" else "medium"
                ),
                "is_tcm_domain_query": True,
                "is_personal_health_query": False,
                "requires_retrieval": True,
                "should_enter_inquiry": False,
                "route_hint": "evidence",
                "reason": intent.reason
                or "检测到一般中医知识类问题，可直接进入证据检索。",
                "rule_signals": rule_signals,
            }
        )

    return intent.model_copy(
        update={
            "rule_signals": rule_signals,
        }
    )


class IntentAgent(StructuredWorkflowComponent[IntentState]):
    schema = IntentState
    system_prompt = INTENT_SYSTEM_PROMPT

    async def assess(
        self, user_text: str, conversation: Sequence[object] | None = None
    ) -> IntentState:
        visible_conversation = conversation_text(conversation)
        rule_signals = detect_rule_signals(user_text)

        intent = await self.invoke_structured(
            {
                "user_question": user_text,
                "visible_conversation": visible_conversation,
                "rule_signals": rule_signals,
                "output_contract": {
                    "do_not_answer": True,
                    "classify_only": True,
                    "use_route_hint": True,
                    "risk_signal_priority": True,
                },
            }
        )

        return _apply_rule_policy(
            intent,
            user_text=user_text,
            rule_signals=rule_signals,
        )
