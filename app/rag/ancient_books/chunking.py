"""
parse_tagged_book()
  ↓
解析整本书，得到所有章节 SelectedSection

select_sections()
  ↓
筛选和打标签，得到目标章节 SelectedSection

build_parent_child()
  ↓
把每个章节切成 parent 和 child

后续：
  ↓
children 建 BM25 / 向量索引
  ↓
检索时命中 child
  ↓
通过 parent_id 找回 parent
  ↓
把 parent 原文交给大模型生成答案
"""
import hashlib
import re

from .filters import filter_retrievable_text
from .schema import EvidenceParent, EvidenceRole, RetrievalChunk, SelectedSection


# 句子边界匹配规则
#
# 作用：
#   把长文本按照句号、问号、感叹号、分号、换行等切成句子单元。
#
# 正则解释：
#   .*?              非贪婪匹配任意内容
#   (?: ... | $)     匹配句子结束符，或者文本结束
#   [。！？!?；;\n]+  中文/英文标点或换行，作为句子边界
#   re.DOTALL        让 . 可以匹配换行符
#
# 注意：
#   这个正则不是 split，而是 finditer。
#   它会保留句末标点。
_SENTENCE_BOUNDARY_PATTERN = re.compile(r".*?(?:[。！？!?；;\n]+|$)", re.DOTALL)

# 用于判断某个章节是不是“诊断/问诊方法”类证据
#
# 例如：
#   十问
#   问病
#   望色
#   闻声
#   切脉
#
# 这些章节不是具体某个症状的证候，而是问诊、诊法、辨证方法。
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
    """
    根据若干字段生成稳定 ID。

    作用：
        只要输入内容不变，生成的 ID 就不变。
        这对于索引构建、结果追踪、实验复现非常重要。

    参数：
        prefix:
            ID 前缀，例如 "parent" 或 "chunk"。

        *parts:
            参与生成 ID 的字段，例如 section_id、正文 hash、重复序号等。

    返回示例：
        parent-4f8a93c1e6a2b7...
        chunk-a31c0e9d7b8f...
    """
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:24]}"


def _body_signature(text: str) -> str:
    """
    计算文本正文的 SHA256 签名。

    作用：
        用于判断文本内容是否重复。
        同样内容会得到同样的 signature。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence_role(section: SelectedSection) -> EvidenceRole:
    """
    根据章节标题判断证据类型。

    输入：
        section:
            一个已经筛选出来的古籍章节。

    输出：
        EvidenceRole 类型，表示该章节在问答系统中的证据角色。

    目前规则是基于标题关键词的简单判断。
    """
    title = section.section
    if any(marker in title for marker in _DIAGNOSTIC_METHOD_TITLES):
        return "diagnostic_method"
    if "脉案" in title:
        return "case"
    if "脉候" in title or "危险" in title:
        return "differential"
    if "病机" in title:
        return "pathogenesis"
    return "syndrome_pattern"


def _bounded_sentence_groups(text: str, limit: int) -> list[str]:
    """
    把文本切成若干个不超过 limit 长度的句子组。

    这里的“句子组”可以理解成 anchor 单元。

    为什么需要这个函数？
        古籍文本很长，不能直接整篇进入向量模型。
        所以要先按句子边界切开。

    参数：
        text:
            输入正文。

        limit:
            每个单元的最大字符长度。

    返回：
        文本片段列表。

    逻辑：
        1. 先按句子边界切分。
        2. 如果某个句子仍然超过 limit，就强制按 limit 截断。
        3. 空白片段会被跳过。
    """
    sentence_units = [
        match.group(0)
        for match in _SENTENCE_BOUNDARY_PATTERN.finditer(text)
        if match.group(0)
    ]
    groups: list[str] = []
    for sentence in sentence_units:
        if len(sentence) > limit:
            groups.extend(
                chunk
                for offset in range(0, len(sentence), limit)
                if (chunk := sentence[offset : offset + limit]).strip()
            )
        elif sentence.strip():
            groups.append(sentence)
    return groups


def _context_window(anchors: list[str], anchor_index: int, limit: int) -> str:
    """
    以某个 anchor 为中心，向左右扩展上下文，形成 parent 文本。

    参数：
        anchors:
            已经切好的句子/片段列表。

        anchor_index:
            当前中心片段的位置。

        limit:
            parent 最大长度限制。

    返回：
        由当前 anchor 加上左右上下文拼接成的 parent 文本。

    设计目的：
        child chunk 用于检索，需要短一些；
        parent chunk 用于回答，需要上下文更完整。

    举例：
        anchors = [A, B, C, D, E]
        当前 anchor 是 C

        如果长度允许，就扩展成：
        B + C + D
        或者：
        A + B + C + D

        直到接近 limit。
    """
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
    """
    为一个章节构建 Parent-Child 检索结构。

    输入：
        section:
            一个古籍章节 SelectedSection。

    输出：
        parents:
            EvidenceParent 列表。
            parent 是较大的证据上下文，用于最终回答引用。

        children:
            RetrievalChunk 列表。
            child 是较小的检索块，用于向量检索、BM25、reranker。

    整体逻辑：
        1. 清洗章节正文。
        2. 判断章节证据类型。
        3. 把正文切成较大的 anchor，最大 1000 字。
        4. 每个 anchor 扩展上下文，形成 parent。
        5. 每个 anchor 再切成 300 字以内的 child。
        6. child 通过 parent_id 关联回 parent。
    """
    
    # 先对原始正文做过滤。
    # 例如去掉不可检索内容、无意义符号、页码、注释等。
    #
    # 具体过滤规则在 .filters.filter_retrievable_text 中。
    filtered_text = filter_retrievable_text(section.original_text)
    if not filtered_text:
        return [], []

    # 判断当前章节属于哪种证据类型
    # 例如 diagnostic_method / case / pathogenesis / syndrome_pattern
    role = _evidence_role(section)
    # 先把过滤后的正文切成 anchor。
    # 每个 anchor 最大 1000 字。
    #
    # 注意：
    #   这里不是最终用于检索的小 chunk。
    #   它更像一个“中心片段”。
    anchor_bodies = _bounded_sentence_groups(filtered_text, 1000)
    
    # 用于记录 parent 文本重复出现的次数
    #
    # 例如同一个 anchor_body 在文本中重复出现，
    # 就通过 ordinal 区分，避免生成完全相同的 ID。
    parent_duplicate_counts: dict[str, int] = {}
    parents: list[EvidenceParent] = []
    children: list[RetrievalChunk] = []

    for anchor_index, anchor_body in enumerate(anchor_bodies):
        # 以当前 anchor 为中心，向左右扩展上下文，形成 parent body。
        #
        # parent_body 最大 1000 字。
        # 这样 parent 比 child 更完整，适合最终给 LLM 作为引用证据。
        parent_body = _context_window(anchor_bodies, anchor_index, 1000)
        # 计算当前 anchor 的签名
        #
        # 注意：
        #   这里用的是 anchor_body，不是 parent_body。
        #   所以 parent_id 主要绑定的是“中心 anchor”，
        #   而 parent 的 original_text 则是扩展后的上下文窗口。
        parent_signature = _body_signature(anchor_body)
        # 处理重复 anchor
        parent_ordinal = parent_duplicate_counts.get(parent_signature, 0)
        parent_duplicate_counts[parent_signature] = parent_ordinal + 1
        # 生成 parent_id
        parent_id = _stable_id(
            "parent",
            section.section_id,
            parent_signature,
            parent_ordinal,
        )
        # 构造 EvidenceParent
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

        # 每个 parent 下面有自己的 child 去重计数器
        child_duplicate_counts: dict[str, int] = {}
        
        # 把当前 anchor 再切成更小的 child chunk
        #
        # 注意：
        #   child 是从 anchor_body 切的，不是从 parent_body 切的。
        #   也就是说：用于检索的是中心 anchor 的内容，
        #   不是扩展后的整个 parent 上下文。
        for child_body in _bounded_sentence_groups(anchor_body, 300):
            
            # 计算 child 正文签名
            child_signature = _body_signature(child_body)
            # 处理同一个 parent 下重复 child 的情况
            child_ordinal = child_duplicate_counts.get(child_signature, 0)
            child_duplicate_counts[child_signature] = child_ordinal + 1
            # 构造 RetrievalChunk
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
