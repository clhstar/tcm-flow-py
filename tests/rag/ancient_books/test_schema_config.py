import copy
import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.rag.ancient_books.config import EXPECTED_BOOK_IDS, load_production_config
from app.rag.ancient_books.schema import (
    EvidenceParent,
    RetrievalChunk,
    RetrievalHit,
    SelectedSection,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPOSITORY_ROOT / "app" / "rag" / "config" / "ancient_books.yaml"


def selected_section_data() -> dict[str, object]:
    return {
        "section_id": "jing-yue-headache-001",
        "source_type": "ancient_book",
        "book_id": "jing_yue_quan_shu",
        "book_title": "景岳全书",
        "source_file": "637-景岳全书.txt",
        "source_hash": "a" * 64,
        "volume": "卷之十",
        "chapter": "头痛",
        "section": "论证",
        "symptom_tags": ["头痛"],
        "original_text": "凡诊头痛者，当先辨其表里虚实。",
    }


def retrieval_hit_data() -> dict[str, object]:
    return {
        "citation_id": "E5",
        "chunk_id": "chunk-001",
        "parent_id": "parent-001",
        "matched_child": "先辨其表里虚实",
        "content": "凡诊头痛者，当先辨其表里虚实。",
        "source_type": "ancient_book",
        "book_title": "景岳全书",
        "source_file": "637-景岳全书.txt",
        "volume": "卷之十",
        "chapter": "头痛",
        "section": "论证",
        "symptom_tags": ["头痛"],
        "evidence_role": "diagnostic_method",
        "retrieval_sources": ["bm25", "dense"],
        "bm25_rank": 1,
        "dense_rank": 2,
        "rrf_score": 0.0325,
        "reranker_score": 0.91,
    }


class AncientBookSchemaTests(unittest.TestCase):
    def test_selected_section_allows_empty_hierarchy_and_symptom_tags(self):
        data = selected_section_data()
        data.update(volume="", chapter="", symptom_tags=[])

        section = SelectedSection.model_validate(data)

        self.assertEqual(section.volume, "")
        self.assertEqual(section.chapter, "")
        self.assertEqual(section.symptom_tags, [])

    def test_selected_section_normalizes_source_hash_and_forbids_extra_fields(self):
        section = SelectedSection.model_validate(selected_section_data())
        self.assertEqual(section.source_hash, "A" * 64)

        invalid = selected_section_data()
        invalid["unexpected"] = True
        with self.assertRaises(ValidationError):
            SelectedSection.model_validate(invalid)

    def test_evidence_parent_rejects_formula_role(self):
        data = {
            "parent_id": "parent-001",
            **{
                key: value
                for key, value in selected_section_data().items()
                if key != "section_id"
            },
            "evidence_role": "formula",
            "normalized_text": "凡诊头痛者，当先辨其表里虚实。",
        }

        with self.assertRaises(ValidationError):
            EvidenceParent.model_validate(data)

    def test_evidence_parent_allows_empty_hierarchy_and_symptom_tags(self):
        data = {
            "parent_id": "parent-001",
            **{
                key: value
                for key, value in selected_section_data().items()
                if key != "section_id"
            },
            "evidence_role": "diagnostic_method",
            "normalized_text": "凡诊头痛者，当先辨其表里虚实。",
        }
        data.update(volume="", chapter="", symptom_tags=[])

        parent = EvidenceParent.model_validate(data)

        self.assertEqual(parent.volume, "")
        self.assertEqual(parent.chapter, "")
        self.assertEqual(parent.symptom_tags, [])

    def test_retrieval_chunk_rejects_text_longer_than_300_characters(self):
        with self.assertRaises(ValidationError):
            RetrievalChunk(
                chunk_id="chunk-001",
                parent_id="parent-001",
                text="医" * 301,
                source_type="curated_markdown",
                symptom_tags=["头痛"],
                evidence_role="symptom_feature",
            )

    def test_retrieval_chunk_allows_empty_symptom_tags(self):
        chunk = RetrievalChunk(
            chunk_id="chunk-001",
            parent_id="parent-001",
            text="无已标注症状的检索片段。",
            source_type="curated_markdown",
            symptom_tags=[],
            evidence_role="symptom_feature",
        )

        self.assertEqual(chunk.symptom_tags, [])

    def test_retrieval_hit_rejects_e6_citation(self):
        data = retrieval_hit_data()
        data["citation_id"] = "E6"

        with self.assertRaises(ValidationError):
            RetrievalHit.model_validate(data)

    def test_retrieval_hit_allows_omitted_ranking_scores(self):
        data = retrieval_hit_data()
        for field in (
            "bm25_rank",
            "dense_rank",
            "rrf_score",
            "reranker_score",
        ):
            del data[field]

        hit = RetrievalHit.model_validate(data)

        self.assertIsNone(hit.bm25_rank)
        self.assertIsNone(hit.dense_rank)
        self.assertIsNone(hit.rrf_score)
        self.assertIsNone(hit.reranker_score)

    def test_retrieval_hit_allows_empty_hierarchy_and_symptom_tags(self):
        data = retrieval_hit_data()
        data.update(volume="", chapter="", symptom_tags=[])

        hit = RetrievalHit.model_validate(data)

        self.assertEqual(hit.volume, "")
        self.assertEqual(hit.chapter, "")
        self.assertEqual(hit.symptom_tags, [])

    def test_schema_models_reject_empty_symptom_tag_values(self):
        selected_section = selected_section_data()
        selected_section["symptom_tags"] = [""]

        evidence_parent = {
            "parent_id": "parent-001",
            **{
                key: value
                for key, value in selected_section_data().items()
                if key != "section_id"
            },
            "symptom_tags": [""],
            "evidence_role": "diagnostic_method",
            "normalized_text": "normalized evidence",
        }

        retrieval_chunk = {
            "chunk_id": "chunk-001",
            "parent_id": "parent-001",
            "text": "retrieval text",
            "source_type": "curated_markdown",
            "symptom_tags": [""],
            "evidence_role": "symptom_feature",
        }

        retrieval_hit = retrieval_hit_data()
        retrieval_hit["symptom_tags"] = [""]

        invalid_models = (
            (SelectedSection, selected_section),
            (EvidenceParent, evidence_parent),
            (RetrievalChunk, retrieval_chunk),
            (RetrievalHit, retrieval_hit),
        )
        for model, data in invalid_models:
            with self.subTest(model=model.__name__):
                with self.assertRaises(ValidationError):
                    model.model_validate(data)


class ProductionConfigTests(unittest.TestCase):
    def load_yaml_data(self) -> dict[str, object]:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def write_temporary_config(self, data: dict[str, object]) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "ancient_books.yaml"
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def test_loads_repository_production_yaml_as_plain_dict(self):
        config = load_production_config(CONFIG_PATH)

        self.assertIsInstance(config, dict)
        self.assertEqual(config["version"], "v1.0.0")
        self.assertEqual(config["source_encoding"], "cp936")
        self.assertEqual(
            {book["book_id"] for book in config["books"]},
            {"jing_yue_quan_shu"},
        )
        self.assertEqual(EXPECTED_BOOK_IDS, frozenset({"jing_yue_quan_shu"}))
        self.assertEqual(
            config["symptoms"],
            {
                "头痛": ["头痛", "头风"],
                "眩晕": ["眩晕", "眩运", "头眩"],
                "咳嗽": ["咳嗽", "咳逆"],
                "喘促": ["喘", "喘逆", "喘症", "上气", "短气"],
                "心悸": ["心悸", "惊悸", "怔忡", "怔仲"],
                "不寐": ["不寐", "不得卧", "失眠"],
                "胃脘痛": ["胃脘痛", "胃脘", "心痛"],
                "腹痛": ["腹痛", "腹满", "心腹痛"],
                "泄泻": ["泄泻", "下利"],
                "便秘": ["便秘", "秘结", "大便不通"],
            },
        )
        self.assertEqual(
            config["exclude_title_patterns"],
            [
                "产后",
                "妊娠",
                "经期",
                "小儿",
                "妇人",
                "女科",
                "幼科",
                "外科",
                "疹",
                "痘",
                "附方",
                "论列方",
                "选方",
            ],
        )
        books = {book["book_id"]: book for book in config["books"]}
        self.assertEqual(len(books), 1)
        self.assertEqual(books["jing_yue_quan_shu"]["title"], "景岳全书")
        self.assertNotIn("book_title", books["jing_yue_quan_shu"])
        self.assertEqual(books["jing_yue_quan_shu"]["method_sections"], ["十问篇（九）"])
        self.assertEqual(
            [
                (book["source_file"], book["symptom_scan"])
                for book in config["books"]
            ],
            [
                ("637-景岳全书.txt", True),
            ],
        )
        self.assertEqual(config["models"]["embedding"]["device"], "cuda")
        self.assertTrue(config["models"]["embedding"]["use_fp16"])
        self.assertEqual(config["models"]["embedding"]["batch_size"], 4)
        self.assertEqual(
            config["models"]["embedding"]["revision"],
            "5617a9f61b028005a4858fdac845db406aefb181",
        )
        self.assertEqual(config["models"]["reranker"]["batch_size"], 2)
        self.assertTrue(config["models"]["reranker"]["normalize_score"])
        self.assertEqual(
            config["models"]["reranker"]["revision"],
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        )
        self.assertEqual(
            config["retrieval"],
            {
                "bm25_top_k": 20,
                "dense_top_k": 20,
                "rrf_k": 60,
                "reranker_candidate_k": 40,
                "final_top_k": 5,
            },
        )

    def test_rejects_config_with_any_book_other_than_jing_yue_quan_shu(self):
        data = copy.deepcopy(self.load_yaml_data())
        extra_book = copy.deepcopy(data["books"][0])
        extra_book.update(
            book_id="yi_men_fa_lv",
            title="医门法律",
            source_file="207-医门法律.txt",
        )
        data["books"].append(extra_book)

        with self.assertRaisesRegex(ValueError, "exactly 1 book|恰好 1 本"):
            load_production_config(self.write_temporary_config(data))

    def test_rejects_duplicate_book_ids(self):
        data = copy.deepcopy(self.load_yaml_data())
        data["books"].append(copy.deepcopy(data["books"][0]))

        with self.assertRaisesRegex(ValueError, "duplicate book IDs|重复"):
            load_production_config(self.write_temporary_config(data))

    def test_rejects_non_commit_model_revision(self):
        data = copy.deepcopy(self.load_yaml_data())
        data["models"]["embedding"]["revision"] = "main"

        with self.assertRaisesRegex(ValueError, "40-character hexadecimal commit hash"):
            load_production_config(self.write_temporary_config(data))

    def test_rejects_unhashable_book_ids_with_clear_validation_error(self):
        for invalid_book_id in (["jing_yue_quan_shu"], {"id": "jing_yue_quan_shu"}):
            with self.subTest(book_id=invalid_book_id):
                data = copy.deepcopy(self.load_yaml_data())
                data["books"][0]["book_id"] = invalid_book_id

                with self.assertRaisesRegex(
                    (ValueError, ValidationError),
                    "book_id",
                ):
                    load_production_config(self.write_temporary_config(data))


if __name__ == "__main__":
    unittest.main()
