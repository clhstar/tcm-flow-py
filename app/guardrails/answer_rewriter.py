import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_rewrite_model() -> ChatOpenAI:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        temperature=0.2,
    )


def build_rewrite_prompt(
    answer: str,
    allowed_terms: list[str],
    unsupported_terms: list[str],
    evidence_text: str,
) -> list[dict[str, str]]:
    allowed_terms_text = "、".join(allowed_terms) if allowed_terms else "无"
    unsupported_terms_text = "、".join(unsupported_terms) if unsupported_terms else "无"

    system_prompt = """
你是一个中医健康问答系统的答案重写助手。

你的任务是直接生成“面向用户展示的最终回答”。

严格要求：
1. 只能输出重写后的最终回答正文。
2. 不要说“我已重写”“重写后的答案如下”“根据您提供的证据”等过程性话语。
3. 不要输出分隔线，例如“---”。
4. 不要解释你是如何重写的。
5. 删除或替换没有证据支持的中医专业术语。
6. 不要引入新的证候、病机、治法或方剂术语。
7. 不要直接下诊断。
8. 不要开处方或给剂量。
9. 表达要谨慎，可以使用“可能相关”“可先从……角度理解”“目前依据有限”等说法。
10. 保留必要的日常调护建议和线下就医提醒。
""".strip()

    user_prompt = f"""
【检索证据】
{evidence_text}

【允许使用的专业术语】
{allowed_terms_text}

【原答案中缺少证据支持的术语】
{unsupported_terms_text}

【需要重写的原答案】
{answer}

请直接输出面向用户的最终回答正文。

要求：
- 只能使用“允许使用的专业术语”中的中医术语；
- 不要使用“原答案中缺少证据支持的术语”；
- 不要说“我已重写”“重写后的答案如下”；
- 不要输出分隔线；
- 如果依据不足，请明确说明“目前检索依据有限”；
- 保持自然、温和、谨慎的表达。
""".strip()

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


async def rewrite_answer(
    answer: str,
    allowed_terms: list[str],
    unsupported_terms: list[str],
    evidence_text: str,
) -> str:
    """
    调用 LLM 重写答案。
    """
    model = get_rewrite_model()

    messages = build_rewrite_prompt(
        answer=answer,
        allowed_terms=allowed_terms,
        unsupported_terms=unsupported_terms,
        evidence_text=evidence_text,
    )

    response = await model.ainvoke(messages)

    content = getattr(response, "content", "")

    return str(content).strip()
