"""
第一步：parse_tagged_book()
读取原始 txt
  ↓
根据 <目录> 和 <篇名> 切分章节
  ↓
生成 SelectedSection 列表

第二步：select_sections()
拿到所有章节
  ↓
根据 method_sections / fixed_sections 精确选章节
  ↓
根据 symptoms 扫描症状相关章节
  ↓
根据 exclude_title_patterns 排除不需要的章节
  ↓
返回最终 selected sections
"""

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .schema import SelectedSection

DIRECTORY_PATTERN = re.compile(r"(?m)^<目录>(?P<value>[^\r\n]*)\r?$")
TITLE_PATTERN = re.compile(r"(?m)^<篇名>(?P<value>[^\r\n]*)\r?$")


def _sha256(raw_bytes: bytes) -> str:
    """
    计算原始字节内容的 SHA256 哈希值。

    作用：
    1. 用于标识源文件是否发生变化。
    2. 用于保证实验可复现。
    3. 用于后续追踪数据来源。

    返回值转成大写，是为了格式统一。
    """
    return hashlib.sha256(raw_bytes).hexdigest().upper()


def _stable_id(prefix: str, *parts: object) -> str:
    """
    根据多个字段生成稳定 ID。

    prefix:
        ID 前缀，通常是 book_id，例如 jing_yue_quan_shu。

    *parts:
        用来生成 ID 的关键字段，比如目录、篇名、正文哈希等。

    为什么叫 stable_id？
        因为只要输入字段不变，生成出来的 ID 就不变。
        这对于实验复现、索引构建、结果追踪很重要。

    示例：
        _stable_id("jing_yue_quan_shu", "卷一", "十问篇", "xxx")
        可能返回：
        jing_yue_quan_shu-a1b2c3d4e5f6...
    """
    # 用 \0 作为分隔符，把多个字段拼接成一个字符串
    # \0 很少出现在普通文本中，可以降低字段拼接冲突的概率
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    # 取 SHA256 的前 24 位作为短 ID
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _title_symptom_tags(
    title: str,
    symptom_aliases: Mapping[str, Sequence[str]],
) -> list[str]:
    """
    根据标题或层级文本，判断这个章节属于哪些症状标签。

    参数：
        title:
            实际上传入的不一定只是篇名，也可能是：
            卷名 + 章节名 + 篇名 拼起来的层级字符串。

        symptom_aliases:
            症状标准名和别名映射。

            例如：
            {
                "头痛": ["头痛", "头风"],
                "眩晕": ["眩晕", "眩运", "头眩"]
            }

    返回：
        命中的标准症状名列表。

    示例：
        title = "卷之一\n头风门\n头风"
        symptom_aliases = {"头痛": ["头痛", "头风"]}

        返回：
        ["头痛"]
    """
    tags: list[str] = []
    for symptom, aliases in symptom_aliases.items():
        candidates = (symptom, *aliases)
        if any(alias and alias in title for alias in candidates):
            tags.append(symptom)
    return tags


def parse_tagged_book(
    path: Path,
    book_id: str,
    book_title: str,
    encoding: str,
) -> list[SelectedSection]:
    """
    解析带标签的古籍文本文件。

    目标：
        从 txt 中提取出一个个章节，转换成 SelectedSection 对象。

    输入文本格式大概类似：

        <目录>卷之一\\入集
        <篇名>十问篇（九）
        正文内容......

        <目录>卷之二\\杂证谟\\头痛
        <篇名>头痛论
        正文内容......

    参数：
        path:
            古籍 txt 文件路径。

        book_id:
            书籍内部 ID，例如 jing_yue_quan_shu。

        book_title:
            书名，例如 景岳全书。

        encoding:
            文件编码，例如 cp936。

    返回：
        所有解析出来的章节列表。
    """
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode(encoding, errors="strict")
    source_hash = _sha256(raw_bytes)
    directory_matches = list(DIRECTORY_PATTERN.finditer(text))
    sections: list[SelectedSection] = []
    duplicate_counts: dict[tuple[str, str, str, str], int] = {}

    for occurrence, directory_match in enumerate(directory_matches):
        block_end = (
            directory_matches[occurrence + 1].start()
            if occurrence + 1 < len(directory_matches)
            else len(text)
        )
        block = text[directory_match.end() : block_end]
        title_match = TITLE_PATTERN.search(block)
        if title_match is None:
            continue

        directory = directory_match.group("value").strip()
        title = title_match.group("value").strip()
        original_text = block[title_match.end() :].strip()
        if original_text.startswith("属性："):
            original_text = original_text.removeprefix("属性：").lstrip()
        if not original_text or not title:
            continue

        hierarchy = [part.strip() for part in directory.split("\\") if part.strip()]
        volume = hierarchy[0] if hierarchy else ""
        chapter = hierarchy[-1] if len(hierarchy) > 1 else ""
        body_hash = _sha256(original_text.encode("utf-8"))
        duplicate_key = (book_id, directory, title, body_hash)
        duplicate_ordinal = duplicate_counts.get(duplicate_key, 0)
        duplicate_counts[duplicate_key] = duplicate_ordinal + 1
        section_id = _stable_id(
            book_id,
            directory,
            title,
            body_hash,
            duplicate_ordinal,
        )
        sections.append(
            SelectedSection(
                section_id=section_id,
                source_type="ancient_book",
                book_id=book_id,
                book_title=book_title,
                source_file=path.name,
                source_hash=source_hash,
                volume=volume,
                chapter=chapter,
                section=title,
                symptom_tags=[],
                original_text=original_text,
            )
        )

    return sections


def select_sections(
    sections: Iterable[SelectedSection],
    symptom_aliases: Mapping[str, Sequence[str]],
    method_sections: Iterable[str],
    fixed_sections: Iterable[str],
    symptom_scan: bool,
    exclude_title_patterns: Iterable[str],
) -> list[SelectedSection]:
    """
    从所有解析出来的章节中，筛选出真正需要进入知识库的章节。

    选择规则有两类：

    1. 精确指定章节：
        method_sections + fixed_sections

        例如：
        method_sections = ["十问篇（九）"]

        这些章节无论是否命中症状词，都会被选中。

    2. 症状扫描章节：
        如果 symptom_scan=True，
        程序会根据 symptom_aliases 去标题/目录中扫描症状词。

        例如标题层级中包含“头风”，就打上“头痛”标签。

    同时还会应用排除规则：
        exclude_title_patterns

        例如：
        产后、妊娠、小儿、妇人、痘、疹 等。
    """
    section_list = list(sections)
    exact_titles = {*method_sections, *fixed_sections}
    excluded_patterns = tuple(exclude_title_patterns)
    selected: list[SelectedSection] = []

    for title in exact_titles:
        matching_structures = {
            (section.source_file, section.volume, section.chapter)
            for section in section_list
            if section.section == title
        }
        if not matching_structures:
            raise ValueError(f"selected section title not found: {title}")
        if len(matching_structures) > 1:
            raise ValueError(
                "selected section title is ambiguous across structures: "
                f"{title}: {sorted(matching_structures)}"
            )

    for section in section_list:
        title = section.section
        is_exact_selection = title in exact_titles
        hierarchy = "\n".join((section.volume, section.chapter, section.section))
        symptom_tags = _title_symptom_tags(hierarchy, symptom_aliases)
        is_excluded_scan = any(
            pattern and pattern in hierarchy for pattern in excluded_patterns
        )
        is_symptom_selection = (
            symptom_scan and bool(symptom_tags) and not is_excluded_scan
        )
        if is_exact_selection or is_symptom_selection:
            selected.append(section.model_copy(update={"symptom_tags": symptom_tags}))

    return sorted(
        selected,
        key=lambda section: (
            section.source_file,
            section.volume,
            section.chapter,
            section.section,
            section.section_id,
        ),
    )
