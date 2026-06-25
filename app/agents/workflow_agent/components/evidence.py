from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence

from app.agents.workflow_agent.models import EvidenceItem, EvidenceResult, InquiryState


EVIDENCE_ANY_HEADER_PATTERN = re.compile(r"^\[(E[^\]]*)\]$")
EVIDENCE_HEADER_PATTERN = re.compile(r"^\[(E[1-5])\]$")
Retriever = Callable[[str, str], Awaitable[str]]


async def default_retriever(query: str, mode: str) -> str:
    from app.tools.builtins.retrieval_tool import retrieve_tcm_knowledge

    return await retrieve_tcm_knowledge.ainvoke({"query": query, "mode": mode})


def _value_after_label(line: str, label: str) -> str:
    if line.startswith(label):
        return line[len(label) :].strip()
    return ""


def _parse_allowed_terms(lines: list[str]) -> list[str]:
    allowed_terms = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("允许使用的专业术语："):
            collecting = True
            continue
        if collecting and stripped.startswith("回答约束："):
            break
        if collecting and stripped:
            term = stripped.lstrip("-").strip()
            if term:
                allowed_terms.append(term)
    return allowed_terms


def _parse_evidence_blocks(lines: list[str]) -> list[EvidenceItem]:
    evidence = []
    current: dict[str, str] | None = None

    def flush_current() -> None:
        nonlocal current
        if current and current.get("id") and current.get("text") and len(evidence) < 5:
            evidence.append(
                EvidenceItem(
                    id=current["id"],
                    citation_id=current["id"],
                    role=current.get("role", ""),
                    text=current["text"],
                    source=current.get("source", ""),
                )
            )
        current = None

    for line in lines:
        stripped = line.strip()
        any_match = EVIDENCE_ANY_HEADER_PATTERN.match(stripped)
        match = EVIDENCE_HEADER_PATTERN.match(stripped)
        if any_match:
            flush_current()
            if not match or len(evidence) >= 5:
                current = None
                continue
            current = {"id": match.group(1).strip()}
            continue
        if current is None:
            continue
        role = _value_after_label(stripped, "证据角色：")
        text = _value_after_label(stripped, "原文：")
        source = _value_after_label(stripped, "来源：")
        if role:
            current["role"] = role
        if text:
            current["text"] = text
        if source:
            current["source"] = source

    flush_current()
    return evidence


def parse_retrieval_result(raw_tool_content: str) -> EvidenceResult:
    lines = raw_tool_content.splitlines()
    status = ""
    mode = ""
    degraded = False
    for line in lines:
        stripped = line.strip()
        status = status or _value_after_label(stripped, "检索状态：")
        mode = mode or _value_after_label(stripped, "检索模式：")
        if stripped.startswith("降级检索：是"):
            degraded = True

    return EvidenceResult(
        retrieval_status=status or "insufficient_evidence",
        retrieval_mode=mode or "hybrid_parent",
        degraded=degraded,
        evidence=_parse_evidence_blocks(lines),
        allowed_terms=_parse_allowed_terms(lines),
        raw_tool_content=raw_tool_content,
    )


def _unique_non_empty(items: Sequence[str]) -> list[str]:
    values = []
    seen = set()
    for item in items:
        value = item.strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


class EvidenceAgent:
    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever or default_retriever

    async def retrieve(self, user_text: str, inquiry: InquiryState) -> EvidenceResult:
        facts = inquiry.known_facts
        query_parts = [
            user_text,
            inquiry.chief_complaint,
            facts.duration,
            *facts.triggers,
            *facts.associated_symptoms,
        ]
        raw_tool_content = await self._retriever(
            " ".join(_unique_non_empty(query_parts)),
            "hybrid",
        )
        return parse_retrieval_result(raw_tool_content)
