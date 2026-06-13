import json
import re
from dataclasses import dataclass
from pathlib import Path

from experiments.rag_v1_5.corpus import CORPUS_VERSION
from experiments.rag_v1_5.schema import (
    EvidenceUnit,
    ParseAnomaly,
    ParseResult,
    ParseStatistics,
)


STRUCTURE_PATTERN = re.compile(
    r"(?m)^<(?P<kind>目录|篇名)>(?P<value>[^\r\n]*)[ \t]*\r?$"
)
CLAUSE_PATTERN = re.compile(
    r"(?m)^(?:属性：)?[ \t]*(?P<number>\d{1,4})[．.]"
)
FORMULA_PATTERN = re.compile(r"\\x(?P<name>[^\\\r\n]+)\\x")
IMPLICIT_FORMULA_MARKER_PATTERN = re.compile(
    r"方(?:[一二三四五六七八九十百\d]+[。．]?|[∶：])"
)
FORMULA_NAME_PATTERN = re.compile(
    r"[\u4e00-\u9fff]{2,16}(?:汤|丸|散|饮|煎|膏|丹|酒)$"
)
FORMULA_REFERENCE_PATTERN = re.compile(
    r"^[ \t\r\n]*(?:《[^》]+》)?"
    r"(?P<name>[\u4e00-\u9fff]{2,16}(?:汤|丸|散|饮|煎|膏|丹|酒))"
)
PREPARATION_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:上|右)(?=[一二三四五六七八九十百〇零两\d \t，、味药])"
)
PARENTHETICAL_PATTERN = re.compile(r"（(?P<fullwidth>[^（）]+)）|\((?P<ascii>[^()]+)\)")
NOTE_KEYWORDS = (
    "一云",
    "一本",
    "一作",
    "臣亿",
    "谨按",
    "按",
    "脉经",
    "玉函",
    "千金",
    "方见",
    "用前",
    "校",
    "注",
)
BOOK_PREFIXES = {
    "shang_han_lun": "shl",
    "jin_gui_yao_lue": "jgy",
}
FORMULA_SECTION_LABELS = {"附方"}
GENERIC_FORMULA_LABELS = {"又方", "治方"}


@dataclass(frozen=True)
class _FormulaMarker:
    start: int
    end: int
    formula_name: str | None
    original_start: int
    creates_formula: bool = True
    ingredients_without_preparation: bool = False


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = FORMULA_PATTERN.sub(lambda match: match.group("name").strip(), text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _book_prefix(book_id: str) -> str:
    if book_id in BOOK_PREFIXES:
        return BOOK_PREFIXES[book_id]

    parts = [part for part in re.split(r"[^a-zA-Z0-9]+", book_id) if part]
    prefix = "".join(part[0].lower() for part in parts)
    return prefix or "book"


def _is_editorial_note(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if any(keyword in compact for keyword in NOTE_KEYWORDS):
        return True

    dosage_units = ("两", "升", "合", "斤", "钱", "枚", "个", "分", "寸", "铢")
    if len(compact) <= 16 and any(unit in compact for unit in dosage_units):
        return False

    return len(compact) >= 20


def _extract_notes(text: str) -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    for match in PARENTHETICAL_PATTERN.finditer(text):
        note_text = (match.group("fullwidth") or match.group("ascii")).strip()
        if _is_editorial_note(note_text):
            notes.append((match.group(0), note_text))
    return notes


def _make_unit(
    *,
    evidence_id: str,
    book_id: str,
    book_title: str,
    volume: str,
    chapter_id: str,
    chapter_title: str,
    clause_id: str,
    clause_number: int,
    content_type: str,
    parent_id: str,
    original_text: str,
    normalized_text: str,
    notes: list[str],
    source_file: str,
    source_hash: str,
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=evidence_id,
        book_id=book_id,
        book_title=book_title,
        volume=volume,
        chapter_id=chapter_id,
        chapter_title=chapter_title,
        clause_id=clause_id,
        clause_number=clause_number,
        content_type=content_type,
        parent_id=parent_id,
        original_text=original_text,
        normalized_text=normalized_text,
        notes=notes,
        source_file=source_file,
        source_hash=source_hash,
        corpus_version=CORPUS_VERSION,
    )


def _clean_formula_label(text: str) -> str:
    return text.strip().rstrip("∶：。．")


def _formula_reference_name(formula_body: str) -> str | None:
    match = FORMULA_REFERENCE_PATTERN.search(formula_body)
    return match.group("name") if match else None


def _sentence_fragment_start(text: str, end: int) -> int:
    prefix = text[:end]
    return max(
        (prefix.rfind(separator) for separator in "。；;！？!?"),
        default=-1,
    ) + 1


def _is_inside_marker(
    position: int,
    explicit_markers: list[re.Match[str]],
) -> bool:
    return any(
        marker.start() <= position < marker.end()
        for marker in explicit_markers
    )


def _collect_formula_markers(clause_text: str) -> list[_FormulaMarker]:
    explicit_matches = list(FORMULA_PATTERN.finditer(clause_text))
    markers: list[_FormulaMarker] = []

    for match in explicit_matches:
        raw_label = match.group("name").strip()
        label = _clean_formula_label(raw_label)
        if label in FORMULA_SECTION_LABELS:
            markers.append(
                _FormulaMarker(
                    start=match.start(),
                    end=match.end(),
                    formula_name=None,
                    original_start=match.start(),
                    creates_formula=False,
                )
            )
            continue

        is_generic = label in GENERIC_FORMULA_LABELS
        markers.append(
            _FormulaMarker(
                start=match.start(),
                end=match.end(),
                formula_name=None if is_generic else raw_label,
                original_start=match.start(),
                ingredients_without_preparation=not is_generic,
            )
        )

    for match in IMPLICIT_FORMULA_MARKER_PATTERN.finditer(clause_text):
        if _is_inside_marker(match.start(), explicit_matches):
            continue

        formula_name = _implicit_formula_name(clause_text[:match.start()])
        original_start = match.start()
        if not formula_name:
            text_before_marker = clause_text[:match.start()].rstrip()
            if not re.search(r"(?:治|解)之$", text_before_marker):
                continue

            original_start = _sentence_fragment_start(
                clause_text,
                match.start(),
            )
            formula_name = _clean_formula_label(
                normalize_text(clause_text[original_start:match.end()])
            )

        markers.append(
            _FormulaMarker(
                start=match.start(),
                end=match.end(),
                formula_name=formula_name,
                original_start=original_start,
            )
        )

    return sorted(markers, key=lambda marker: (marker.start, marker.end))


def _formula_units(
    *,
    clause_text: str,
    clause_id: str,
    book_id: str,
    book_title: str,
    volume: str,
    chapter_id: str,
    chapter_title: str,
    clause_number: int,
    source_file: str,
    source_hash: str,
) -> list[EvidenceUnit]:
    markers = _collect_formula_markers(clause_text)
    units: list[EvidenceUnit] = []
    accepted_formula_count = 0

    for marker_index, marker in enumerate(markers):
        segment_end = (
            markers[marker_index + 1].start
            if marker_index + 1 < len(markers)
            else len(clause_text)
        )
        if not marker.creates_formula:
            continue

        formula_body = clause_text[marker.end:segment_end].strip()
        formula_name = marker.formula_name or _formula_reference_name(formula_body)
        formula_name = formula_name or "又方"
        accepted_formula_count += 1
        formula_id = f"{clause_id}-formula-{accepted_formula_count:02d}"
        normalized_body = normalize_text(formula_body)
        normalized_formula = formula_name
        if normalized_body:
            normalized_formula = f"{formula_name}\n{normalized_body}"

        units.append(
            _make_unit(
                evidence_id=formula_id,
                book_id=book_id,
                book_title=book_title,
                volume=volume,
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                clause_id=clause_id,
                clause_number=clause_number,
                content_type="formula",
                parent_id=clause_id,
                original_text=clause_text[
                    marker.original_start:segment_end
                ].strip(),
                normalized_text=normalized_formula,
                notes=[],
                source_file=source_file,
                source_hash=source_hash,
            )
        )

        preparation_match = PREPARATION_PATTERN.search(formula_body)
        ingredients_text = (
            formula_body[:preparation_match.start()].strip()
            if preparation_match
            else (
                formula_body.strip()
                if marker.ingredients_without_preparation
                else ""
            )
        )
        preparation_text = (
            formula_body[preparation_match.start():].strip()
            if preparation_match
            else ""
        )

        if ingredients_text:
            units.append(
                _make_unit(
                    evidence_id=f"{formula_id}-ingredients",
                    book_id=book_id,
                    book_title=book_title,
                    volume=volume,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    clause_id=clause_id,
                    clause_number=clause_number,
                    content_type="ingredients",
                    parent_id=formula_id,
                    original_text=ingredients_text,
                    normalized_text=normalize_text(ingredients_text),
                    notes=[],
                    source_file=source_file,
                    source_hash=source_hash,
                )
            )

        if preparation_text:
            units.append(
                _make_unit(
                    evidence_id=f"{formula_id}-preparation",
                    book_id=book_id,
                    book_title=book_title,
                    volume=volume,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    clause_id=clause_id,
                    clause_number=clause_number,
                    content_type="preparation",
                    parent_id=formula_id,
                    original_text=preparation_text,
                    normalized_text=normalize_text(preparation_text),
                    notes=[],
                    source_file=source_file,
                    source_hash=source_hash,
                )
            )

    return units


def _implicit_formula_name(text_before_marker: str) -> str | None:
    candidate = text_before_marker[-100:].rstrip(" \t\r\n。．；;：:")
    if candidate.endswith("主之"):
        candidate = candidate[:-2].rstrip(" \t\r\n。．；;：:")

    candidate = re.split(r"[，,。．；;：:\r\n]", candidate)[-1].strip()
    match = FORMULA_NAME_PATTERN.search(candidate)
    if not match:
        return None

    formula_name = match.group(0)
    formula_name = re.sub(r"^(?:则|宜|与|服|用|可)+", "", formula_name)
    return formula_name or None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_bytes(content.encode("utf-8"))


def parse_corpus_file(
    *,
    input_path: Path,
    book_id: str,
    book_title: str,
    source_hash: str,
    output_path: Path | None = None,
    anomalies_path: Path | None = None,
) -> ParseResult:
    text = input_path.read_bytes().decode("utf-8")
    structure_events = list(STRUCTURE_PATTERN.finditer(text))
    prefix = _book_prefix(book_id)
    evidence_units: list[EvidenceUnit] = []
    anomalies: list[ParseAnomaly] = []
    current_volume = ""
    chapter_count = 0

    for event_index, event in enumerate(structure_events):
        kind = event.group("kind")
        value = event.group("value").strip()

        if kind == "目录":
            current_volume = value
            continue

        if value == book_title:
            continue

        chapter_count += 1
        chapter_id = f"{prefix}-chapter-{chapter_count:02d}"
        chapter_title = value
        block_end = (
            structure_events[event_index + 1].start()
            if event_index + 1 < len(structure_events)
            else len(text)
        )
        chapter_text = text[event.end():block_end]
        clause_matches = list(CLAUSE_PATTERN.finditer(chapter_text))

        if not clause_matches and normalize_text(chapter_text):
            anomalies.append(
                ParseAnomaly(
                    anomaly_id=f"{chapter_id}-anomaly-unparsed-chapter-01",
                    book_id=book_id,
                    source_file=input_path.name,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    clause_id=None,
                    reason="unparsed_chapter_text",
                    original_text=chapter_text.strip(),
                )
            )
            continue

        for clause_index, clause_match in enumerate(clause_matches):
            clause_number = int(clause_match.group("number"))
            clause_end = (
                clause_matches[clause_index + 1].start()
                if clause_index + 1 < len(clause_matches)
                else len(chapter_text)
            )
            clause_text = chapter_text[clause_match.end():clause_end].strip()
            if not clause_text:
                anomalies.append(
                    ParseAnomaly(
                        anomaly_id=f"{chapter_id}-{clause_number:03d}-anomaly-empty-01",
                        book_id=book_id,
                        source_file=input_path.name,
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        clause_id=None,
                        reason="empty_clause",
                        original_text=clause_match.group(0),
                    )
                )
                continue

            clause_id = f"{chapter_id}-{clause_number:03d}"
            extracted_notes = _extract_notes(clause_text)
            note_texts = [note_text for _, note_text in extracted_notes]
            evidence_units.append(
                _make_unit(
                    evidence_id=clause_id,
                    book_id=book_id,
                    book_title=book_title,
                    volume=current_volume,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    clause_id=clause_id,
                    clause_number=clause_number,
                    content_type="clause",
                    parent_id=clause_id,
                    original_text=clause_text,
                    normalized_text=normalize_text(clause_text),
                    notes=note_texts,
                    source_file=input_path.name,
                    source_hash=source_hash,
                )
            )

            for note_index, (original_note, note_text) in enumerate(
                extracted_notes,
                start=1,
            ):
                evidence_units.append(
                    _make_unit(
                        evidence_id=f"{clause_id}-note-{note_index:02d}",
                        book_id=book_id,
                        book_title=book_title,
                        volume=current_volume,
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        clause_id=clause_id,
                        clause_number=clause_number,
                        content_type="note",
                        parent_id=clause_id,
                        original_text=original_note,
                        normalized_text=normalize_text(note_text),
                        notes=[],
                        source_file=input_path.name,
                        source_hash=source_hash,
                    )
                )

            evidence_units.extend(
                _formula_units(
                    clause_text=clause_text,
                    clause_id=clause_id,
                    book_id=book_id,
                    book_title=book_title,
                    volume=current_volume,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    clause_number=clause_number,
                    source_file=input_path.name,
                    source_hash=source_hash,
                )
            )

            if "KT" in clause_text:
                anomalies.append(
                    ParseAnomaly(
                        anomaly_id=(
                            f"{clause_id}-anomaly-missing-character-marker-01"
                        ),
                        book_id=book_id,
                        source_file=input_path.name,
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        clause_id=clause_id,
                        reason="missing_character_marker",
                        original_text=clause_text,
                    )
                )

    counts = {
        content_type: sum(
            unit.content_type == content_type for unit in evidence_units
        )
        for content_type in (
            "clause",
            "formula",
            "ingredients",
            "preparation",
            "note",
        )
    }
    result = ParseResult(
        evidence_units=evidence_units,
        anomalies=anomalies,
        statistics=ParseStatistics(
            chapter_count=chapter_count,
            clause_count=counts["clause"],
            formula_count=counts["formula"],
            ingredients_count=counts["ingredients"],
            preparation_count=counts["preparation"],
            note_count=counts["note"],
            anomaly_count=len(anomalies),
        ),
    )

    if output_path is not None:
        _write_jsonl(
            output_path,
            [unit.model_dump(mode="json") for unit in result.evidence_units],
        )
    if anomalies_path is not None:
        _write_jsonl(
            anomalies_path,
            [anomaly.model_dump(mode="json") for anomaly in result.anomalies],
        )

    return result
