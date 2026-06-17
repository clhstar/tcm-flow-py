import unittest

from app.rag.ancient_books.chunking import build_parent_child
from app.rag.ancient_books.filters import filter_retrievable_text
from app.rag.ancient_books.schema import SelectedSection


def make_section(
    text: str,
    *,
    section: str = "头痛证治",
    section_id: str = "section-001",
) -> SelectedSection:
    return SelectedSection(
        section_id=section_id,
        source_type="ancient_book",
        book_id="sample_book",
        book_title="测试医书",
        source_file="sample.txt",
        source_hash="A" * 64,
        volume="卷一",
        chapter="杂证",
        section=section,
        symptom_tags=["头痛"],
        original_text=text,
    )


class RetrievableTextFilterTests(unittest.TestCase):
    def test_mixed_diagnostic_and_formula_text_keeps_only_symptom_features(self):
        text = (
            "因风者恶风，川芎茶调散。因火者齿痛，连翘、丹皮、桑叶。"
            "\\x用药\\x 黄芩（二钱）、栀子（一钱），水煎。"
        )

        filtered = filter_retrievable_text(text)

        self.assertIn("因风者恶风", filtered)
        self.assertIn("因火者齿痛", filtered)
        for unsafe in ("川芎茶调散", "连翘", "丹皮", "桑叶", "黄芩", "二钱", "水煎"):
            self.assertNotIn(unsafe, filtered)

    def test_pure_formula_text_filters_to_empty(self):
        text = "川芎茶调散。川芎、白芷、羌活各二钱，为末，每服三钱。"

        self.assertEqual(filter_retrievable_text(text), "")

    def test_treatment_instructions_are_removed_but_diagnostic_location_remains(self):
        text = "头痛连及胁下。治头痛。每日服。空心下。慢火煎。"

        self.assertEqual(filter_retrievable_text(text), "头痛连及胁下。")

    def test_new_safe_subtitle_resumes_retention_after_excluded_subtitle(self):
        text = (
            "\\x方剂\\x 白术、茯苓各二钱，水煎。"
            "\\x辨证\\x 因寒者喜热饮，因热者喜冷饮。"
        )

        filtered = filter_retrievable_text(text)

        self.assertEqual(filtered, "因寒者喜热饮，因热者喜冷饮。")

    def test_empty_and_whitespace_inputs_return_empty(self):
        self.assertEqual(filter_retrievable_text(""), "")
        self.assertEqual(filter_retrievable_text(" \n\t "), "")


class ParentChildChunkingTests(unittest.TestCase):
    def test_empty_or_fully_filtered_section_produces_no_chunks(self):
        for text in (" ", "川芎茶调散。白芷、羌活各二钱，为末。"):
            with self.subTest(text=text):
                self.assertEqual(build_parent_child(make_section(text)), ([], []))

    def test_parent_child_lengths_relationship_and_inherited_metadata(self):
        text = "".join(f"症候{i}表现明显，遇劳则甚。" for i in range(180))

        parents, children = build_parent_child(make_section(text))

        self.assertGreater(len(parents), 1)
        self.assertGreater(len(children), len(parents))
        self.assertTrue(all(len(parent.original_text) <= 1000 for parent in parents))
        self.assertTrue(all(len(child.text) <= 300 for child in children))
        parent_ids = {parent.parent_id for parent in parents}
        self.assertEqual({child.parent_id for child in children} - parent_ids, set())
        for parent in parents:
            self.assertEqual(parent.source_file, "sample.txt")
            self.assertEqual(parent.source_hash, "A" * 64)
            self.assertEqual(parent.symptom_tags, ["头痛"])
            self.assertEqual(parent.normalized_text, " ".join(parent.original_text.split()))
        for child in children:
            self.assertEqual(child.source_type, "ancient_book")
            self.assertEqual(child.symptom_tags, ["头痛"])

    def test_long_sentence_is_safely_split_with_no_content_loss(self):
        text = "因风头痛" + "痛" * 1200 + "。"

        parents, children = build_parent_child(make_section(text))

        self.assertGreater(len(parents), 1)
        self.assertEqual("".join(parent.original_text for parent in parents), text)
        for parent in parents:
            parent_children = [child for child in children if child.parent_id == parent.parent_id]
            self.assertEqual("".join(child.text for child in parent_children), parent.original_text)

    def test_unrelated_leading_parent_does_not_shift_existing_parent_ids(self):
        first = "甲" * 999 + "。"
        second = "乙" * 999 + "。"
        original_parents, _ = build_parent_child(make_section(first + second))
        leading_parents, _ = build_parent_child(
            make_section("无" * 999 + "。" + first + second)
        )

        original_ids = {parent.original_text: parent.parent_id for parent in original_parents}
        leading_ids = {
            parent.original_text: parent.parent_id
            for parent in leading_parents
            if parent.original_text in original_ids
        }
        self.assertEqual(leading_ids, original_ids)

    def test_repeated_identical_bodies_get_unique_stable_ids(self):
        repeated_parent = "同" * 999 + "。"
        section = make_section(repeated_parent * 2)

        first_parents, first_children = build_parent_child(section)
        second_parents, second_children = build_parent_child(section)

        first_parent_ids = [parent.parent_id for parent in first_parents]
        first_child_ids = [child.chunk_id for child in first_children]
        self.assertEqual(len(first_parent_ids), len(set(first_parent_ids)))
        self.assertEqual(len(first_child_ids), len(set(first_child_ids)))
        self.assertEqual(first_parent_ids, [parent.parent_id for parent in second_parents])
        self.assertEqual(first_child_ids, [child.chunk_id for child in second_children])

    def test_section_titles_map_to_expected_evidence_roles(self):
        cases = {
            "十问篇": "diagnostic_method",
            "问病论": "diagnostic_method",
            "望色诀": "diagnostic_method",
            "闻声篇": "diagnostic_method",
            "辨息论": "diagnostic_method",
            "切脉法": "diagnostic_method",
            "合色脉法": "diagnostic_method",
            "问诊要诀": "diagnostic_method",
            "头痛脉案": "case",
            "头痛脉候": "differential",
            "危险证候": "differential",
            "头痛病机": "pathogenesis",
            "头痛证治": "syndrome_pattern",
        }

        for title, expected_role in cases.items():
            with self.subTest(title=title):
                parents, children = build_parent_child(
                    make_section("因风者恶风。", section=title)
                )
                self.assertEqual([parent.evidence_role for parent in parents], [expected_role])
                self.assertEqual([child.evidence_role for child in children], [expected_role])

    def test_parent_original_text_contains_no_formula_drug_dose_or_preparation(self):
        text = (
            "因风者恶风，川芎茶调散。因火者齿痛，连翘、丹皮、桑叶。"
            "白芷（二钱）、羌活三钱，为末，每服一钱。"
        )

        parents, _ = build_parent_child(make_section(text))
        parent_text = "".join(parent.original_text for parent in parents)

        self.assertIn("因风者恶风", parent_text)
        self.assertIn("因火者齿痛", parent_text)
        for unsafe in ("川芎茶调散", "连翘", "丹皮", "桑叶", "白芷", "二钱", "为末", "每服"):
            self.assertNotIn(unsafe, parent_text)

    def test_safe_original_text_formatting_is_preserved(self):
        text = "因风者恶风,  因火者齿痛。\n脉浮则病在表。"

        parents, _ = build_parent_child(make_section(text))

        self.assertEqual("".join(parent.original_text for parent in parents), text)
        self.assertEqual(
            " ".join(parent.normalized_text for parent in parents),
            "因风者恶风, 因火者齿痛。 脉浮则病在表。",
        )


if __name__ == "__main__":
    unittest.main()
