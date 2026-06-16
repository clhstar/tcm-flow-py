import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from experiments.rag_v1_5.corpus import CorpusFileSpec, prepare_corpus
from experiments.rag_v1_5.parser import parse_corpus_file
from experiments.rag_v1_5.pipeline import (
    parse_prepared_corpus,
    validate_evidence_graph,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def parse_single_clause(
    clause_text: str,
    *,
    book_id: str = "jin_gui_yao_lue",
):
    book_title = (
        "金匮要略方论"
        if book_id == "jin_gui_yao_lue"
        else "伤寒论"
    )
    text = (
        f"<篇名>{book_title}\n"
        "<目录>测试卷\n"
        "<篇名>测试篇\n"
        f"属性：1．{clause_text}\n"
    )
    temporary_directory = tempfile.TemporaryDirectory()
    input_path = Path(temporary_directory.name) / "sample.txt"
    input_path.write_text(text, encoding="utf-8")
    result = parse_corpus_file(
        input_path=input_path,
        book_id=book_id,
        book_title=book_title,
        source_hash="A" * 64,
    )
    temporary_directory.cleanup()
    return result


def units_by_type(result) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for unit in result.evidence_units:
        grouped.setdefault(unit.content_type, []).append(unit)
    return grouped


class AncientBookParserTests(unittest.TestCase):
    def test_parses_chapters_clauses_and_stable_ids(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "jin_gui_sample.txt",
            book_id="jin_gui_yao_lue",
            book_title="金匮要略方论",
            source_hash="A" * 64,
        )

        clauses = [
            unit for unit in result.evidence_units
            if unit.content_type == "clause"
        ]

        self.assertEqual(len(clauses), 3)
        self.assertEqual(clauses[0].chapter_id, "jgy-chapter-01")
        self.assertEqual(clauses[0].clause_id, "jgy-chapter-01-001")
        self.assertEqual(clauses[2].chapter_id, "jgy-chapter-02")
        self.assertEqual(clauses[2].clause_id, "jgy-chapter-02-001")
        self.assertEqual(clauses[0].volume, "卷上")
        self.assertEqual(clauses[2].volume, "卷中")
        self.assertIn("\n", clauses[0].original_text)
        self.assertNotIn("\n", clauses[0].normalized_text)

    def test_extracts_formula_ingredients_preparation_and_note_children(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "jin_gui_sample.txt",
            book_id="jin_gui_yao_lue",
            book_title="金匮要略方论",
            source_hash="A" * 64,
        )
        by_type = {}
        for unit in result.evidence_units:
            by_type.setdefault(unit.content_type, []).append(unit)

        self.assertEqual(len(by_type["formula"]), 1)
        self.assertEqual(len(by_type["ingredients"]), 1)
        self.assertEqual(len(by_type["preparation"]), 1)
        self.assertEqual(len(by_type["note"]), 1)

        formula = by_type["formula"][0]
        ingredients = by_type["ingredients"][0]
        preparation = by_type["preparation"][0]
        note = by_type["note"][0]

        self.assertEqual(formula.normalized_text.splitlines()[0], "栝蒌桂枝汤方")
        self.assertIn("栝蒌根（二两）", ingredients.normalized_text)
        self.assertIn("上三味", preparation.normalized_text)
        self.assertEqual(formula.parent_id, "jgy-chapter-01-002")
        self.assertEqual(ingredients.parent_id, formula.evidence_id)
        self.assertEqual(preparation.parent_id, formula.evidence_id)
        self.assertEqual(note.parent_id, "jgy-chapter-01-001")
        self.assertEqual(note.normalized_text, "一云痉病。")

    def test_records_missing_character_markers_without_repairing_them(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "jin_gui_sample.txt",
            book_id="jin_gui_yao_lue",
            book_title="金匮要略方论",
            source_hash="A" * 64,
        )

        self.assertEqual(len(result.anomalies), 1)
        self.assertEqual(result.anomalies[0].reason, "missing_character_marker")
        self.assertIn("KT KT", result.anomalies[0].original_text)

        clause = next(
            unit for unit in result.evidence_units
            if unit.clause_id == "jgy-chapter-01-002"
            and unit.content_type == "clause"
        )
        self.assertIn("KT KT", clause.normalized_text)

    def test_writes_evidence_and_anomaly_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "evidence.jsonl"
            anomalies_path = Path(temp_dir) / "anomalies.jsonl"

            result = parse_corpus_file(
                input_path=FIXTURES_DIR / "jin_gui_sample.txt",
                book_id="jin_gui_yao_lue",
                book_title="金匮要略方论",
                source_hash="A" * 64,
                output_path=output_path,
                anomalies_path=anomalies_path,
            )

            evidence_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            anomaly_rows = [
                json.loads(line)
                for line in anomalies_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(len(evidence_rows), len(result.evidence_units))
            self.assertEqual(len(anomaly_rows), len(result.anomalies))
            self.assertEqual(
                evidence_rows[0]["source_file"],
                "jin_gui_sample.txt",
            )

    def test_parses_shang_han_numbered_clauses(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "shang_han_sample.txt",
            book_id="shang_han_lun",
            book_title="伤寒论",
            source_hash="B" * 64,
        )

        self.assertEqual(result.statistics.chapter_count, 1)
        self.assertEqual(result.statistics.clause_count, 3)
        self.assertEqual(
            result.evidence_units[0].normalized_text,
            "太阳之为病，脉浮、 头项强痛而恶寒。",
        )

    def test_extracts_implicit_shang_han_formula_structure(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "shang_han_sample.txt",
            book_id="shang_han_lun",
            book_title="伤寒论",
            source_hash="B" * 64,
        )
        formula = next(
            unit for unit in result.evidence_units
            if unit.content_type == "formula"
        )
        ingredients = next(
            unit for unit in result.evidence_units
            if unit.content_type == "ingredients"
        )
        preparation = next(
            unit for unit in result.evidence_units
            if unit.content_type == "preparation"
        )

        self.assertEqual(formula.normalized_text.splitlines()[0], "桂枝汤")
        self.assertEqual(formula.parent_id, "shl-chapter-01-003")
        self.assertIn("桂枝（去皮，三两）", ingredients.normalized_text)
        self.assertIn("上三味", preparation.normalized_text)

    def test_parses_inline_treatment_formula_before_generic_alternative_marker(self):
        text = """<篇名>金匮要略方论
<目录>卷下
<篇名>果实菜谷禁忌并治第二十五
属性：40．饮食中毒，烦满，治之方∶
苦参（三两） 苦酒（一升半）
上二味，煮三沸，三上三下，之服，吐食出即瘥。
或以水煮亦得。
\\x又方∶\\x
犀角汤亦佳。
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "499-sample.txt"
            input_path.write_text(text, encoding="utf-8")

            result = parse_corpus_file(
                input_path=input_path,
                book_id="jin_gui_yao_lue",
                book_title="金匮要略方论",
                source_hash="A" * 64,
            )

        formulas = [
            unit for unit in result.evidence_units
            if unit.content_type == "formula"
        ]
        ingredients = [
            unit for unit in result.evidence_units
            if unit.content_type == "ingredients"
        ]
        preparations = [
            unit for unit in result.evidence_units
            if unit.content_type == "preparation"
        ]

        self.assertEqual(
            [unit.evidence_id for unit in formulas],
            [
                "jgy-chapter-01-040-formula-01",
                "jgy-chapter-01-040-formula-02",
            ],
        )
        self.assertEqual(
            formulas[0].normalized_text.splitlines()[0],
            "饮食中毒，烦满，治之方",
        )
        self.assertEqual(
            formulas[1].normalized_text.splitlines()[0],
            "犀角汤",
        )
        self.assertEqual(len(ingredients), 1)
        self.assertEqual(
            ingredients[0].normalized_text,
            "苦参（三两） 苦酒（一升半）",
        )
        self.assertEqual(len(preparations), 1)
        self.assertTrue(preparations[0].normalized_text.startswith("上二味"))
        self.assertFalse(
            any("犀角汤亦佳" in unit.normalized_text for unit in ingredients)
        )

    def test_treats_attached_formula_label_as_a_section_boundary(self):
        text = """<篇名>金匮要略方论
<目录>卷上
<篇名>痉湿病脉证第二
属性：1．测试条文。
\\x甲汤方\\x
甲药（一两）
上一味，水煎服。
\\x附方\\x
\\x乙汤\\x
乙药（二两）
上一味，水煎服。
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "499-sample.txt"
            input_path.write_text(text, encoding="utf-8")

            result = parse_corpus_file(
                input_path=input_path,
                book_id="jin_gui_yao_lue",
                book_title="金匮要略方论",
                source_hash="A" * 64,
            )

        formulas = [
            unit for unit in result.evidence_units
            if unit.content_type == "formula"
        ]
        self.assertEqual(
            [unit.normalized_text.splitlines()[0] for unit in formulas],
            ["甲汤方", "乙汤"],
        )
        self.assertEqual(
            [unit.evidence_id for unit in formulas],
            [
                "jgy-chapter-01-001-formula-01",
                "jgy-chapter-01-001-formula-02",
            ],
        )

    def test_parses_unmarked_formula_headings_and_count_marker_formula(self):
        cases = (
            (
                "退五脏虚热，四时加减柴胡饮子方。\n"
                "冬三月加∶柴胡（八分） 白术（八分）\n"
                "上各 咀，分为三帖，一帖以水三升，煮取二升。",
                "jin_gui_yao_lue",
                "四时加减柴胡饮子方",
            ),
            (
                "长服诃黎勒丸方（疑非仲景方。）\n"
                "诃黎勒 陈皮 浓朴（各三两）\n"
                "上三味，末之，炼蜜丸，如梧子大。",
                "jin_gui_yao_lue",
                "长服诃黎勒丸方",
            ),
            (
                "伤寒瘥以后更发热，小柴胡汤主之；"
                "脉浮者，以汗解之。方三。\n"
                "柴胡（八两） 人参（二两） 黄芩（二两）\n"
                "上七味，以水一斗二升，煮取六升。",
                "shang_han_lun",
                "小柴胡汤",
            ),
        )

        for clause_text, book_id, formula_name in cases:
            with self.subTest(formula_name=formula_name):
                grouped = units_by_type(
                    parse_single_clause(clause_text, book_id=book_id)
                )
                self.assertEqual(len(grouped.get("formula", [])), 1)
                self.assertEqual(len(grouped.get("ingredients", [])), 1)
                self.assertEqual(len(grouped.get("preparation", [])), 1)
                self.assertEqual(
                    grouped["formula"][0].normalized_text.splitlines()[0],
                    formula_name,
                )

    def test_formula_count_and_colon_markers_do_not_cross_boundaries(self):
        cases = (
            (
                "霍乱，头痛、发热、身疼痛、热多欲饮水者，"
                "五苓散主之；寒多不用水者，理中丸主之。方二。\n"
                "五苓散方∶\n"
                "猪苓（去皮） 白术 茯苓（各十八铢）\n"
                "上五味，为散，白饮和服方寸匕。\n"
                "理中丸方∶\n"
                "人参 干姜 甘草（炙） 白术（各三两）\n"
                "上四味，捣筛，蜜和为丸。",
                ("五苓散", "理中丸"),
            ),
            (
                "本太阳病，医反下之，因尔腹满时痛者，"
                "桂枝加芍药汤主之；大实痛者，桂枝加大黄汤主之。"
                "方三。\n"
                "桂枝加芍药汤方∶\n"
                "桂枝（三两） 芍药（六两）\n"
                "上二味，以水七升，煮取三升。\n"
                "桂枝加大黄汤方∶\n"
                "桂枝（三两） 大黄（二两）\n"
                "上二味，以水七升，煮取三升。",
                ("桂枝加芍药汤", "桂枝加大黄汤"),
            ),
        )

        for clause_text, expected_names in cases:
            with self.subTest(expected_names=expected_names):
                grouped = units_by_type(
                    parse_single_clause(
                        clause_text,
                        book_id="shang_han_lun",
                    )
                )
                formulas = grouped.get("formula", [])
                ingredients = grouped.get("ingredients", [])
                preparations = grouped.get("preparation", [])
                self.assertEqual(
                    tuple(
                        formula.normalized_text.splitlines()[0]
                        for formula in formulas
                    ),
                    expected_names,
                )
                self.assertEqual(len(ingredients), 2)
                self.assertEqual(len(preparations), 2)
                for index in range(len(expected_names) - 1):
                    next_name = expected_names[index + 1]
                    self.assertNotIn(next_name, formulas[index].original_text)
                    self.assertNotIn(next_name, ingredients[index].original_text)
                    self.assertNotIn(
                        next_name,
                        preparations[index].original_text,
                    )
                self.assertFalse(
                    any(
                        formula.normalized_text.splitlines()[0]
                        in {"方二", "方三"}
                        for formula in formulas
                    )
                )

    def test_formula_count_marker_can_start_direct_ingredients(self):
        clause_text = (
            "阳明病，大承气汤主之；若腹大满不通者，"
            "可与小承气汤。大承气汤。方二。\n"
            "大黄（酒洗，四两） 浓朴（炙，半斤）\n"
            "上二味，以水一斗，煮取二升。\n"
            "小承气汤方∶\n"
            "大黄（酒洗，四两） 浓朴（炙，二两）\n"
            "上二味，以水四升，煮取一升。"
        )

        grouped = units_by_type(
            parse_single_clause(clause_text, book_id="shang_han_lun")
        )
        formulas = grouped.get("formula", [])

        self.assertEqual(
            [
                formula.normalized_text.splitlines()[0]
                for formula in formulas
            ],
            ["大承气汤", "小承气汤"],
        )
        self.assertEqual(len(grouped.get("ingredients", [])), 2)
        self.assertEqual(len(grouped.get("preparation", [])), 2)
        self.assertNotIn("小承气汤", formulas[0].original_text)

    def test_formula_count_restores_prefixed_formula_name(self):
        clause_text = (
            "太阳病，外证未解，脉浮弱者，当以汗解，"
            "宜桂枝汤。方十二。\n"
            "桂枝（去皮） 芍药 生姜（切，各三两）\n"
            "上三味，以水七升，煮取三升。"
        )

        grouped = units_by_type(
            parse_single_clause(clause_text, book_id="shang_han_lun")
        )

        self.assertEqual(
            [
                formula.normalized_text.splitlines()[0]
                for formula in grouped.get("formula", [])
            ],
            ["桂枝汤"],
        )
        self.assertEqual(len(grouped.get("ingredients", [])), 1)
        self.assertEqual(len(grouped.get("preparation", [])), 1)

    def test_formula_count_normalizes_short_prefixed_reference(self):
        grouped = units_by_type(
            parse_single_clause(
                "宜桂枝汤。方十九。（用前第十二方。）",
                book_id="shang_han_lun",
            )
        )

        self.assertEqual(
            [
                formula.normalized_text.splitlines()[0]
                for formula in grouped.get("formula", [])
            ],
            ["桂枝汤"],
        )

    def test_formula_count_reference_without_body_does_not_create_formula(self):
        clause_text = (
            "血弱、气尽，邪气因入，小柴胡汤主之。"
            "服柴胡汤已，渴者属阳明，以法治之。"
            "方四十九。（用前方。）"
        )

        grouped = units_by_type(
            parse_single_clause(clause_text, book_id="shang_han_lun")
        )

        self.assertEqual(grouped.get("formula", []), [])

    def test_formula_count_before_noted_heading_is_metadata_only(self):
        clause_text = (
            "伤寒，先与小建中汤；不瘥者，小柴胡汤主之。"
            "方五十一。（用前方。）\n"
            "小建中汤方。\n"
            "桂枝（三两） 甘草（二两） 芍药（六两）\n"
            "上三味，以水七升，煮取三升。"
        )

        grouped = units_by_type(
            parse_single_clause(clause_text, book_id="shang_han_lun")
        )

        self.assertEqual(
            [
                formula.normalized_text.splitlines()[0]
                for formula in grouped.get("formula", [])
            ],
            ["小建中汤方"],
        )
        self.assertEqual(len(grouped.get("ingredients", [])), 1)
        self.assertEqual(len(grouped.get("preparation", [])), 1)

    def test_reference_heading_does_not_consume_next_formula_body(self):
        clause_text = (
            "里水，越婢加术汤主之，甘草麻黄汤亦主之。\n"
            "越婢加术汤方（见上。）\n"
            "\\x甘草麻黄汤方\\x\n"
            "甘草（二两） 麻黄（四两）\n"
            "上二味，以水五升，先煮麻黄。"
        )

        grouped = units_by_type(parse_single_clause(clause_text))

        self.assertEqual(
            [
                formula.normalized_text.splitlines()[0]
                for formula in grouped.get("formula", [])
            ],
            ["甘草麻黄汤方"],
        )
        self.assertEqual(len(grouped.get("ingredients", [])), 1)
        self.assertEqual(len(grouped.get("preparation", [])), 1)

    def test_recognizes_additional_preparation_starts(self):
        cases = (
            (
                "\\x百合知母汤方\\x\n"
                "百合（七枚，擘） 知母（三两，切）\n"
                "上先以水洗百合，渍一宿。",
                "上先",
            ),
            (
                "\\x桂枝救逆汤方\\x\n"
                "桂枝（三两） 甘草（二两）\n"
                "上为末，以水一斗二升，先煮蜀漆。",
                "上为",
            ),
            (
                "退五脏虚热，四时加减柴胡饮子方。\n"
                "柴胡（八分） 白术（八分）\n"
                "上各 咀，分为三帖。",
                "上各",
            ),
        )

        for clause_text, preparation_start in cases:
            with self.subTest(preparation_start=preparation_start):
                grouped = units_by_type(parse_single_clause(clause_text))
                ingredients = grouped.get("ingredients", [])
                preparations = grouped.get("preparation", [])
                self.assertEqual(len(ingredients), 1)
                self.assertEqual(len(preparations), 1)
                self.assertNotIn(
                    preparation_start,
                    ingredients[0].original_text,
                )
                self.assertTrue(
                    preparations[0].original_text.startswith(
                        preparation_start
                    )
                )

    def test_formula_not_seen_is_note_only_formula_body(self):
        grouped = units_by_type(
            parse_single_clause(
                "病患常以手指臂肿动，藜芦甘草汤主之。\n"
                "\\x藜芦甘草汤\\x（方未见）"
            )
        )

        self.assertEqual(len(grouped.get("formula", [])), 1)
        self.assertEqual(grouped.get("ingredients", []), [])
        self.assertEqual(grouped.get("preparation", []), [])
        self.assertEqual(len(grouped.get("note", [])), 1)
        self.assertEqual(grouped["note"][0].normalized_text, "方未见")
        self.assertEqual(
            grouped["note"][0].parent_id,
            grouped["formula"][0].clause_id,
        )

    def test_parses_prepared_manifest_into_combined_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            manifest_path = root / "corpus-v1.5.0.json"
            source_dir.mkdir()

            source_text = (
                FIXTURES_DIR / "jin_gui_sample.txt"
            ).read_text(encoding="utf-8")
            source_bytes = source_text.encode("cp936")
            source_file = source_dir / "499-金匮要略方论.txt"
            source_file.write_bytes(source_bytes)
            source_hash = hashlib.sha256(source_bytes).hexdigest().upper()
            prepare_corpus(
                source_dir=source_dir,
                output_dir=raw_dir,
                manifest_path=manifest_path,
                specs=[
                    CorpusFileSpec(
                        book_id="jin_gui_yao_lue",
                        book_title="金匮要略方论",
                        source_filename=source_file.name,
                        expected_sha256=source_hash,
                    )
                ],
                generated_at=datetime(
                    2026,
                    6,
                    12,
                    8,
                    0,
                    tzinfo=timezone.utc,
                ),
            )

            statistics = parse_prepared_corpus(
                raw_dir=raw_dir,
                manifest_path=manifest_path,
                processed_dir=processed_dir,
            )

            self.assertEqual(
                statistics["books"]["jin_gui_yao_lue"]["clause_count"],
                3,
            )
            self.assertTrue((processed_dir / "evidence.jsonl").is_file())
            self.assertTrue((processed_dir / "anomalies.jsonl").is_file())
            self.assertTrue((processed_dir / "statistics.json").is_file())

            rows = [
                json.loads(line)
                for line in (
                    processed_dir / "evidence.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(
                all(row["source_hash"] == source_hash for row in rows)
            )

    def test_rejects_duplicate_and_orphan_evidence_ids(self):
        result = parse_corpus_file(
            input_path=FIXTURES_DIR / "jin_gui_sample.txt",
            book_id="jin_gui_yao_lue",
            book_title="金匮要略方论",
            source_hash="A" * 64,
        )

        with self.assertRaisesRegex(ValueError, "重复 evidence_id"):
            validate_evidence_graph(
                [*result.evidence_units, result.evidence_units[0]]
            )

        orphan = result.evidence_units[1].model_copy(
            update={"parent_id": "missing-parent"}
        )
        with self.assertRaisesRegex(ValueError, "找不到 parent_id"):
            validate_evidence_graph([result.evidence_units[0], orphan])


if __name__ == "__main__":
    unittest.main()
