import hashlib
import tempfile
import unittest
from pathlib import Path

from app.rag.ancient_books.corpus import parse_tagged_book, select_sections
from app.rag.ancient_books.schema import SelectedSection


SYMPTOM_ALIASES = {
    "头痛": ["头痛", "头风"],
    "咳嗽": ["咳嗽", "咳逆"],
    "喘促": ["喘", "喘逆", "上气", "短气"],
    "腹痛": ["腹痛", "腹满"],
}


def make_section(
    title: str,
    section_id: str,
    *,
    source_file: str = "sample.txt",
    volume: str = "卷一",
    chapter: str = "杂证",
) -> SelectedSection:
    return SelectedSection(
        section_id=section_id,
        source_type="ancient_book",
        book_id="sample_book",
        book_title="测试医书",
        source_file=source_file,
        source_hash="A" * 64,
        volume=volume,
        chapter=chapter,
        section=title,
        symptom_tags=[],
        original_text=f"{title}正文",
    )


class TaggedBookParserTests(unittest.TestCase):
    def write_cp936_book(self, root: Path, text: str) -> tuple[Path, bytes]:
        path = root / "637-景岳全书.txt"
        raw_bytes = text.encode("cp936")
        path.write_bytes(raw_bytes)
        return path, raw_bytes

    def test_parses_cp936_hierarchy_text_and_source_hash(self):
        text = (
            "<目录>卷之一\\杂证谟\\头痛\n"
            "<篇名>头痛论治\n"
            "属性：1．凡诊头痛者，当先辨表里虚实。\n"
            "次察寒热。\n"
            "<目录>卷之二\n"
            "<篇名>空篇\n"
            "属性：\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path, raw_bytes = self.write_cp936_book(Path(temp_dir), text)

            sections = parse_tagged_book(
                path,
                book_id="jing_yue_quan_shu",
                book_title="景岳全书",
                encoding="cp936",
            )

        self.assertEqual(len(sections), 1)
        section = sections[0]
        self.assertEqual(section.source_type, "ancient_book")
        self.assertEqual(section.book_id, "jing_yue_quan_shu")
        self.assertEqual(section.book_title, "景岳全书")
        self.assertEqual(section.source_file, "637-景岳全书.txt")
        self.assertEqual(section.volume, "卷之一")
        self.assertEqual(section.chapter, "头痛")
        self.assertEqual(section.section, "头痛论治")
        self.assertEqual(
            section.original_text,
            "1．凡诊头痛者，当先辨表里虚实。\n次察寒热。",
        )
        self.assertEqual(
            section.source_hash,
            hashlib.sha256(raw_bytes).hexdigest().upper(),
        )

    def test_duplicate_directory_and_title_get_unique_stable_ids(self):
        text = (
            "<目录>卷一\\头痛\n"
            "<篇名>头痛论治\n"
            "属性：第一段。\n"
            "<目录>卷一\\头痛\n"
            "<篇名>头痛论治\n"
            "属性：第二段。\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _ = self.write_cp936_book(Path(temp_dir), text)

            first_parse = parse_tagged_book(
                path,
                book_id="jing_yue_quan_shu",
                book_title="景岳全书",
                encoding="cp936",
            )
            second_parse = parse_tagged_book(
                path,
                book_id="jing_yue_quan_shu",
                book_title="景岳全书",
                encoding="cp936",
            )

        first_ids = [section.section_id for section in first_parse]
        self.assertEqual(len(first_ids), 2)
        self.assertEqual(len(set(first_ids)), 2)
        self.assertEqual(
            first_ids,
            [section.section_id for section in second_parse],
        )

    def test_unrelated_leading_section_does_not_change_existing_ids(self):
        original_text = (
            "<目录>卷一\\头痛\n"
            "<篇名>头痛论治\n"
            "属性：第一段。\n"
            "<目录>卷二\\咳嗽\n"
            "<篇名>咳嗽论治\n"
            "属性：第二段。\n"
        )
        leading_text = (
            "<目录>卷首\\总论\n"
            "<篇名>诊法总论\n"
            "属性：前置无关正文。\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _ = self.write_cp936_book(Path(temp_dir), original_text)
            original_sections = parse_tagged_book(
                path,
                book_id="jing_yue_quan_shu",
                book_title="景岳全书",
                encoding="cp936",
            )
            path.write_bytes((leading_text + original_text).encode("cp936"))
            sections_with_leading = parse_tagged_book(
                path,
                book_id="jing_yue_quan_shu",
                book_title="景岳全书",
                encoding="cp936",
            )

        original_ids = {
            section.section: section.section_id for section in original_sections
        }
        ids_with_leading = {
            section.section: section.section_id
            for section in sections_with_leading
            if section.section in original_ids
        }
        self.assertEqual(ids_with_leading, original_ids)

    def test_identical_duplicate_sections_get_unique_stable_ids(self):
        repeated_section = (
            "<目录>卷一\\头痛\n"
            "<篇名>头痛论治\n"
            "属性：完全相同正文。\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _ = self.write_cp936_book(Path(temp_dir), repeated_section * 2)
            first_parse = parse_tagged_book(
                path,
                book_id="jing_yue_quan_shu",
                book_title="景岳全书",
                encoding="cp936",
            )
            second_parse = parse_tagged_book(
                path,
                book_id="jing_yue_quan_shu",
                book_title="景岳全书",
                encoding="cp936",
            )

        first_ids = [section.section_id for section in first_parse]
        self.assertEqual(len(first_ids), 2)
        self.assertEqual(len(set(first_ids)), 2)
        self.assertEqual(
            first_ids,
            [section.section_id for section in second_parse],
        )

    def test_single_directory_segment_leaves_chapter_empty(self):
        text = "<目录>卷一\n<篇名>总论\n属性：正文。\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _ = self.write_cp936_book(Path(temp_dir), text)

            section = parse_tagged_book(
                path,
                book_id="sample",
                book_title="测试医书",
                encoding="cp936",
            )[0]

        self.assertEqual(section.volume, "卷一")
        self.assertEqual(section.chapter, "")

    def test_parses_windows_line_endings(self):
        text = "<目录>卷一\\头痛\r\n<篇名>头痛论治\r\n属性：正文。\r\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _ = self.write_cp936_book(Path(temp_dir), text)

            sections = parse_tagged_book(
                path,
                book_id="sample",
                book_title="测试医书",
                encoding="cp936",
            )

        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].original_text, "正文。")

    def test_invalid_cp936_bytes_raise_unicode_decode_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.txt"
            path.write_bytes(b"<\x81\x30>")

            with self.assertRaises(UnicodeDecodeError):
                parse_tagged_book(
                    path,
                    book_id="sample",
                    book_title="测试医书",
                    encoding="cp936",
                )


class SectionSelectionTests(unittest.TestCase):
    def test_symptom_scan_excludes_formula_and_obstetric_titles(self):
        sections = [
            make_section("产后腹痛", "s3"),
            make_section("十问篇（九）", "s4"),
            make_section("头痛论列方", "s2"),
            make_section("头痛论治", "s1"),
        ]

        selected = select_sections(
            sections,
            symptom_aliases=SYMPTOM_ALIASES,
            method_sections=["十问篇（九）"],
            fixed_sections=[],
            symptom_scan=True,
            exclude_title_patterns=["产后", "妇人", "附方", "论列方"],
        )

        self.assertEqual(
            [section.section for section in selected],
            ["十问篇（九）", "头痛论治"],
        )
        self.assertEqual(selected[0].symptom_tags, [])
        self.assertEqual(selected[1].symptom_tags, ["头痛"])

    def test_symptom_scan_excludes_specialty_markers_in_full_hierarchy(self):
        sections = [
            make_section(
                "泄泻",
                "pediatrics",
                volume="卷之四十五烈集·痘疹诠",
                chapter="痘疮（下）",
            ),
            make_section(
                "泄泻",
                "internal",
                volume="卷之二十四心集·杂证谟",
                chapter="泄泻",
            ),
        ]

        selected = select_sections(
            sections,
            symptom_aliases={"泄泻": ["泄泻"]},
            method_sections=[],
            fixed_sections=[],
            symptom_scan=True,
            exclude_title_patterns=["痘", "小儿", "妇人", "产后"],
        )

        self.assertEqual([section.section_id for section in selected], ["internal"])

    def test_symptom_scan_tags_section_from_chapter_hierarchy(self):
        section = make_section(
            "论证",
            "headache-evidence",
            volume="卷之十七理集·杂证谟",
            chapter="头痛",
        )

        selected = select_sections(
            [section],
            symptom_aliases={"头痛": ["头痛", "头风"]},
            method_sections=[],
            fixed_sections=[],
            symptom_scan=True,
            exclude_title_patterns=[],
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].symptom_tags, ["头痛"])

    def test_fixed_sections_are_tagged_when_symptom_scan_is_disabled(self):
        title = "肺痿肺痈咳嗽上气病脉证治第七"

        selected = select_sections(
            [make_section(title, "fixed-1")],
            symptom_aliases=SYMPTOM_ALIASES,
            method_sections=[],
            fixed_sections=[title],
            symptom_scan=False,
            exclude_title_patterns=[],
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].symptom_tags, ["咳嗽", "喘促"])

    def test_untagged_method_section_is_valid(self):
        selected = select_sections(
            [make_section("十问篇（九）", "method-1")],
            symptom_aliases=SYMPTOM_ALIASES,
            method_sections=["十问篇（九）"],
            fixed_sections=[],
            symptom_scan=False,
            exclude_title_patterns=[],
        )

        self.assertEqual(selected[0].symptom_tags, [])

    def test_missing_method_section_raises_with_title(self):
        missing_title = "十问篇（九）"

        with self.assertRaisesRegex(ValueError, missing_title):
            select_sections(
                [make_section("头痛论治", "s1")],
                symptom_aliases=SYMPTOM_ALIASES,
                method_sections=[missing_title],
                fixed_sections=[],
                symptom_scan=True,
                exclude_title_patterns=[],
            )

    def test_missing_fixed_section_raises_with_title(self):
        missing_title = "肺痿肺痈咳嗽上气病脉证治第七"

        with self.assertRaisesRegex(ValueError, missing_title):
            select_sections(
                [make_section("头痛论治", "s1")],
                symptom_aliases=SYMPTOM_ALIASES,
                method_sections=[],
                fixed_sections=[missing_title],
                symptom_scan=True,
                exclude_title_patterns=[],
            )

    def test_whitelisted_title_in_multiple_structures_is_ambiguous(self):
        title = "十问篇（九）"
        sections = [
            make_section(title, "method-1", volume="卷一", chapter="问诊"),
            make_section(title, "method-2", volume="卷二", chapter="问诊"),
        ]

        with self.assertRaisesRegex(ValueError, title):
            select_sections(
                sections,
                symptom_aliases=SYMPTOM_ALIASES,
                method_sections=[title],
                fixed_sections=[],
                symptom_scan=False,
                exclude_title_patterns=[],
            )

    def test_whitelisted_duplicate_fragments_in_same_structure_are_valid(self):
        title = "十问篇（九）"
        sections = [
            make_section(title, "method-1", volume="卷一", chapter="问诊"),
            make_section(title, "method-2", volume="卷一", chapter="问诊"),
        ]

        selected = select_sections(
            sections,
            symptom_aliases=SYMPTOM_ALIASES,
            method_sections=[title],
            fixed_sections=[],
            symptom_scan=False,
            exclude_title_patterns=[],
        )

        self.assertEqual(
            [section.section_id for section in selected],
            ["method-1", "method-2"],
        )


if __name__ == "__main__":
    unittest.main()
