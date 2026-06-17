import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from app.rag.documents import parse_markdown_sections

from .schema import SelectedSection


DIRECTORY_PATTERN = re.compile(r"(?m)^<目录>(?P<value>[^\r\n]*)\r?$")
TITLE_PATTERN = re.compile(r"(?m)^<篇名>(?P<value>[^\r\n]*)\r?$")


def _sha256(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest().upper()


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _title_symptom_tags(
    title: str,
    symptom_aliases: Mapping[str, Sequence[str]],
) -> list[str]:
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

        hierarchy = [
            part.strip() for part in directory.split("\\") if part.strip()
        ]
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
        symptom_tags = _title_symptom_tags(title, symptom_aliases)
        is_exact_selection = title in exact_titles
        is_excluded_scan = any(
            pattern and pattern in title for pattern in excluded_patterns
        )
        is_symptom_selection = (
            symptom_scan and bool(symptom_tags) and not is_excluded_scan
        )
        if is_exact_selection or is_symptom_selection:
            selected.append(
                section.model_copy(update={"symptom_tags": symptom_tags})
            )

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


def load_curated_sections(
    root: Path,
    symptom_aliases: Mapping[str, Sequence[str]],
) -> list[SelectedSection]:
    sections: list[SelectedSection] = []

    for path in sorted(root.glob("*.md"), key=lambda item: item.name):
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8", errors="strict")
        source_hash = _sha256(raw_bytes)
        documents = parse_markdown_sections(
            text=text,
            source=str(path),
            filename=path.name,
        )

        for occurrence, document in enumerate(documents):
            topic = str(document.metadata["topic"])
            section_title = str(document.metadata["section"])
            section_id = _stable_id(
                "curated",
                path.name,
                topic,
                section_title,
                occurrence,
                document.page_content,
            )
            sections.append(
                SelectedSection(
                    section_id=section_id,
                    source_type="curated_markdown",
                    book_id="curated_markdown",
                    book_title="人工整理知识",
                    source_file=path.name,
                    source_hash=source_hash,
                    volume="",
                    chapter=topic,
                    section=section_title,
                    symptom_tags=_title_symptom_tags(
                        f"{topic}\n{section_title}",
                        symptom_aliases,
                    ),
                    original_text=document.page_content,
                )
            )

    return sections
