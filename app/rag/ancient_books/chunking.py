import hashlib
import re

from .filters import filter_retrievable_text
from .schema import EvidenceParent, EvidenceRole, RetrievalChunk, SelectedSection


_SENTENCE_BOUNDARY_PATTERN = re.compile(r".*?(?:[。！？!?；;\n]+|$)", re.DOTALL)
_DIAGNOSTIC_METHOD_TITLES = (
    "十问",
    "问病",
    "望色",
    "闻声",
    "辨息",
    "切脉",
    "合色脉",
    "问诊",
)


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _body_signature(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence_role(section: SelectedSection) -> EvidenceRole:
    title = section.section
    if any(marker in title for marker in _DIAGNOSTIC_METHOD_TITLES):
        return "diagnostic_method"
    if "脉案" in title:
        return "case"
    if "脉候" in title or "危险" in title:
        return "differential"
    if "病机" in title:
        return "pathogenesis"
    if section.source_type == "curated_markdown":
        return "symptom_feature"
    return "syndrome_pattern"


def _bounded_sentence_groups(text: str, limit: int) -> list[str]:
    sentence_units = [
        match.group(0)
        for match in _SENTENCE_BOUNDARY_PATTERN.finditer(text)
        if match.group(0)
    ]
    groups: list[str] = []
    for sentence in sentence_units:
        if len(sentence) > limit:
            groups.extend(
                sentence[offset : offset + limit]
                for offset in range(0, len(sentence), limit)
            )
        else:
            groups.append(sentence)
    return groups


def _context_window(anchors: list[str], anchor_index: int, limit: int) -> str:
    start = anchor_index
    end = anchor_index
    total_length = len(anchors[anchor_index])
    left = anchor_index - 1
    right = anchor_index + 1

    while left >= 0 or right < len(anchors):
        expanded = False
        if left >= 0:
            if total_length + len(anchors[left]) <= limit:
                start = left
                total_length += len(anchors[left])
                left -= 1
                expanded = True
            else:
                left = -1
        if right < len(anchors):
            if total_length + len(anchors[right]) <= limit:
                end = right
                total_length += len(anchors[right])
                right += 1
                expanded = True
            else:
                right = len(anchors)
        if not expanded:
            break

    return "".join(anchors[start : end + 1])


def build_parent_child(
    section: SelectedSection,
) -> tuple[list[EvidenceParent], list[RetrievalChunk]]:
    filtered_text = filter_retrievable_text(section.original_text)
    if not filtered_text:
        return [], []

    role = _evidence_role(section)
    anchor_bodies = _bounded_sentence_groups(filtered_text, 1000)
    parent_duplicate_counts: dict[str, int] = {}
    parents: list[EvidenceParent] = []
    children: list[RetrievalChunk] = []

    for anchor_index, anchor_body in enumerate(anchor_bodies):
        parent_body = _context_window(anchor_bodies, anchor_index, 1000)
        parent_signature = _body_signature(anchor_body)
        parent_ordinal = parent_duplicate_counts.get(parent_signature, 0)
        parent_duplicate_counts[parent_signature] = parent_ordinal + 1
        parent_id = _stable_id(
            "parent",
            section.section_id,
            parent_signature,
            parent_ordinal,
        )
        parents.append(
            EvidenceParent(
                parent_id=parent_id,
                source_type=section.source_type,
                book_id=section.book_id,
                book_title=section.book_title,
                source_file=section.source_file,
                source_hash=section.source_hash,
                volume=section.volume,
                chapter=section.chapter,
                section=section.section,
                symptom_tags=list(section.symptom_tags),
                evidence_role=role,
                original_text=parent_body,
                normalized_text=" ".join(parent_body.split()),
            )
        )

        child_duplicate_counts: dict[str, int] = {}
        for child_body in _bounded_sentence_groups(anchor_body, 300):
            child_signature = _body_signature(child_body)
            child_ordinal = child_duplicate_counts.get(child_signature, 0)
            child_duplicate_counts[child_signature] = child_ordinal + 1
            children.append(
                RetrievalChunk(
                    chunk_id=_stable_id(
                        "chunk",
                        parent_id,
                        child_signature,
                        child_ordinal,
                    ),
                    parent_id=parent_id,
                    text=child_body,
                    source_type=section.source_type,
                    symptom_tags=list(section.symptom_tags),
                    evidence_role=role,
                )
            )

    return parents, children
