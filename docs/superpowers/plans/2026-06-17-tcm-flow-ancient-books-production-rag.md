# TCM-Flow Ancient Books Production RAG Implementation Plan

> **Scope amendment (2026-06-17):** The production corpus is restricted to
> `637-景岳全书.txt`. Earlier seven-book steps below are retained as historical planning
> context but are superseded by the single-book production configuration and operator guide.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Integrate a production ancient-books RAG pipeline for ten adult internal-medicine symptoms into the existing retrieve_tcm_knowledge tool without adding a new formal experiment.

**Architecture:** Add an isolated app/rag/ancient_books package for corpus selection, CP936 parsing, treatment-content filtering, Parent-Child chunking, persistent BGE-M3 indexes, and runtime hybrid retrieval. Keep retrieve_tcm_docs() and retrieve_tcm_knowledge as compatibility adapters, and keep the existing curated Markdown source in the same production index with explicit source_type metadata.

**Tech Stack:** Python 3, Pydantic 2, PyYAML, jieba, rank-bm25, NumPy, FlagEmbedding BGE-M3, BGE reranker, unittest

---

## Scope Guard

This plan implements project functionality only. It must not create:

- a B4/P comparison matrix;
- a 120-case research dataset;
- bootstrap statistics or confidence intervals;
- a new experiments/rag_* package;
- a paper results document.

The only verification in this plan is unit, integration, index-doctor, and fixed smoke testing.

## File Map

### New production package

- app/rag/ancient_books/__init__.py: package exports.
- app/rag/ancient_books/schema.py: strict production corpus, chunk, index, and hit models.
- app/rag/ancient_books/config.py: YAML loading and validated paths/settings.
- app/rag/ancient_books/corpus.py: CP936 import, tagged-section parsing, whitelist selection, and curated Markdown conversion.
- app/rag/ancient_books/filters.py: formula, dosage, preparation, and specialty exclusion.
- app/rag/ancient_books/chunking.py: stable Parent-Child construction.
- app/rag/ancient_books/models.py: fixed-revision BGE embedding and reranker adapters.
- app/rag/ancient_books/indexing.py: deterministic rows, BM25 token, dense-vector, and manifest writer.
- app/rag/ancient_books/runtime.py: verified index loader, BM25/dense/RRF/rerank, Parent recovery, and explicit degradation.
- app/rag/ancient_books/query.py: ten-symptom routing and conservative alias expansion.
- app/rag/ancient_books/pipeline.py: corpus artifact builder and build orchestration.
- app/rag/ancient_books/cli.py: prepare-models, build-corpus, build-index, build-all, doctor, and smoke commands.
- app/rag/config/ancient_books.yaml: seven-book whitelist and fixed production retrieval settings.

### Existing production integration

- app/rag/documents.py: retain Markdown section parsing and expose conversion metadata.
- app/rag/terms.py: expand the supported symptom vocabulary without injecting unobserved syndromes.
- app/rag/vector_store.py: compatibility wrapper for the verified production index.
- app/rag/bm25_retriever.py: compatibility wrapper for production BM25 retrieval.
- app/rag/retriever.py: preserve the public function and response shape while delegating to the new runtime.
- app/rag/build_index.py: delegate to the production build command.
- app/tools/builtins/retrieval_tool.py: log and format ancient-book citation metadata.
- app/agents/lead_agent/prompt.py: explain E1-E5 evidence use and formula prohibition.
- requirements.txt: move local BGE runtime dependencies into production requirements.

### Tests and operations

- tests/rag/__init__.py
- tests/rag/ancient_books/__init__.py
- tests/rag/ancient_books/test_schema_config.py
- tests/rag/ancient_books/test_corpus.py
- tests/rag/ancient_books/test_filters_chunking.py
- tests/rag/ancient_books/test_indexing.py
- tests/rag/ancient_books/test_runtime.py
- tests/rag/ancient_books/test_query.py
- tests/rag/ancient_books/test_cli.py
- tests/test_rag_tool.py
- docs/rag/ancient-books-production-rag.md

## Task 1: Add strict production configuration and schemas

**Files:**
- Create: app/rag/ancient_books/__init__.py
- Create: app/rag/ancient_books/schema.py
- Create: app/rag/ancient_books/config.py
- Create: app/rag/config/ancient_books.yaml
- Create: tests/rag/__init__.py
- Create: tests/rag/ancient_books/__init__.py
- Create: tests/rag/ancient_books/test_schema_config.py

- [ ] **Step 1: Write failing schema and configuration tests**

~~~python
# tests/rag/ancient_books/test_schema_config.py
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from app.rag.ancient_books.config import load_production_config
from app.rag.ancient_books.schema import EvidenceParent


class SchemaConfigTests(unittest.TestCase):
    def test_parent_rejects_formula_role(self):
        with self.assertRaises(ValidationError):
            EvidenceParent(
                parent_id="p1",
                source_type="ancient_book",
                book_id="jing_yue_quan_shu",
                book_title="景岳全书",
                source_file="637-景岳全书.txt",
                source_hash="A" * 64,
                volume="卷之一入集",
                chapter="传忠录（上）",
                section="十问篇（九）",
                symptom_tags=["头痛"],
                evidence_role="formula",
                original_text="某方及剂量",
                normalized_text="某方及剂量",
            )

    def test_config_requires_seven_unique_books(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(
                "version: v1.0.0\nbooks: []\nretrieval: {}\nmodels: {}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "恰好包含 7 本书"):
                load_production_config(path)


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_schema_config -v
~~~

Expected: FAIL because app.rag.ancient_books does not exist.

- [ ] **Step 3: Implement strict models and configuration loading**

~~~python
# app/rag/ancient_books/schema.py
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SourceType = Literal["ancient_book", "curated_markdown"]
EvidenceRole = Literal[
    "diagnostic_method",
    "symptom_feature",
    "syndrome_pattern",
    "pathogenesis",
    "differential",
    "case",
]


class SelectedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    source_type: SourceType
    book_id: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    volume: str
    chapter: str
    section: str = Field(min_length=1)
    symptom_tags: list[str]
    original_text: str = Field(min_length=1)

    @field_validator("source_hash")
    @classmethod
    def normalize_hash(cls, value: str) -> str:
        return value.upper()


class EvidenceParent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_id: str = Field(min_length=1)
    source_type: SourceType
    book_id: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    volume: str
    chapter: str
    section: str = Field(min_length=1)
    symptom_tags: list[str]
    evidence_role: EvidenceRole
    original_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)


class RetrievalChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    parent_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=300)
    source_type: SourceType
    symptom_tags: list[str]
    evidence_role: EvidenceRole


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(pattern=r"^E[1-5]$")
    chunk_id: str
    parent_id: str
    matched_child: str
    content: str
    source_type: SourceType
    book_title: str
    source_file: str
    volume: str
    chapter: str
    section: str
    symptom_tags: list[str]
    evidence_role: EvidenceRole
    retrieval_sources: list[str]
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
~~~

~~~python
# app/rag/ancient_books/config.py
from pathlib import Path

import yaml


EXPECTED_BOOK_IDS = {
    "jing_yue_quan_shu",
    "yi_men_fa_lv",
    "zheng_yin_mai_zhi",
    "lei_zheng_zhi_cai",
    "zheng_zhi_hui_bu",
    "jin_gui_yao_lue",
    "huang_di_nei_jing_su_wen",
}


def load_production_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    books = config.get("books", [])
    book_ids = [book.get("book_id") for book in books]
    if len(books) != 7 or set(book_ids) != EXPECTED_BOOK_IDS:
        raise ValueError("生产古籍配置必须恰好包含 7 本书")
    if len(book_ids) != len(set(book_ids)):
        raise ValueError("生产古籍配置存在重复 book_id")
    for key in ("embedding", "reranker"):
        revision = config["models"][key]["revision"]
        if len(revision) != 40:
            raise ValueError(f"{key} revision 必须是 40 位 commit hash")
    return config
~~~

Create app/rag/config/ancient_books.yaml with all seven file names, method-section names, fixed auxiliary sections, the ten alias groups from the design, exclusion title patterns, and these exact model settings:

~~~yaml
version: v1.0.0
source_encoding: cp936
symptoms:
  头痛: [头痛, 头风]
  眩晕: [眩晕, 眩运, 头眩]
  咳嗽: [咳嗽, 咳逆]
  喘促: [喘, 喘逆, 喘症, 上气, 短气]
  心悸: [心悸, 惊悸, 怔忡, 怔仲]
  不寐: [不寐, 不得卧, 失眠]
  胃脘痛: [胃脘痛, 胃脘, 心痛]
  腹痛: [腹痛, 腹满, 心腹痛]
  泄泻: [泄泻, 下利]
  便秘: [便秘, 秘结, 大便不通]
exclude_title_patterns: [产后, 妊娠, 经期, 小儿, 妇人, 女科, 幼科, 外科, 疹, 痘, 附方, 论列方, 选方]
books:
  - {book_id: jing_yue_quan_shu, title: 景岳全书, source_file: 637-景岳全书.txt, symptom_scan: true, method_sections: [十问篇（九）], fixed_sections: []}
  - {book_id: yi_men_fa_lv, title: 医门法律, source_file: 207-医门法律.txt, symptom_scan: false, method_sections: [望色论（附律一条）, 闻声论（附律二条）, 辨息论（附律一条）, 问病论（附律一条）, 切脉论（附律一条）, 合色脉论（附律一条）], fixed_sections: []}
  - {book_id: zheng_yin_mai_zhi, title: 症因脉治, source_file: 257-症因脉治.txt, symptom_scan: true, method_sections: [], fixed_sections: []}
  - {book_id: lei_zheng_zhi_cai, title: 类证治裁, source_file: 602-类证治裁.txt, symptom_scan: true, method_sections: [], fixed_sections: []}
  - {book_id: zheng_zhi_hui_bu, title: 证治汇补, source_file: 289-证治汇补.txt, symptom_scan: true, method_sections: [], fixed_sections: []}
  - {book_id: jin_gui_yao_lue, title: 金匮要略方论, source_file: 499-金匮要略方论.txt, symptom_scan: false, method_sections: [], fixed_sections: [肺痿肺痈咳嗽上气病脉证治第七, 胸痹心痛短气病脉证治第九, 腹满寒疝宿食病脉证治第十, 痰饮咳嗽病脉证并治第十二, 惊悸吐衄下血胸满瘀血病脉证治第十六, 呕吐哕下利病脉证治第十七]}
  - {book_id: huang_di_nei_jing_su_wen, title: 黄帝内经素问, source_file: 437-黄帝内经素问.txt, symptom_scan: false, method_sections: [], fixed_sections: [移精变气论篇第十三, 诊要经终论篇第十六, 脉要精微论篇第十七, 平人气象论篇第十八, 咳论篇第三十八, 举痛论篇第三十九]}
models:
  embedding: {model: BAAI/bge-m3, revision: 5617a9f61b028005a4858fdac845db406aefb181, device: cuda, use_fp16: true, batch_size: 4, max_length: 1024}
  reranker: {model: BAAI/bge-reranker-v2-m3, revision: 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e, device: cuda, use_fp16: true, batch_size: 2, max_length: 1024, normalize_score: true}
retrieval: {bm25_top_k: 20, dense_top_k: 20, rrf_k: 60, reranker_candidate_k: 40, final_top_k: 5}
~~~

- [ ] **Step 4: Run schema/config tests**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_schema_config -v
~~~

Expected: 2 tests pass.

- [ ] **Step 5: Commit the schema and configuration**

~~~powershell
git add app/rag/ancient_books app/rag/config/ancient_books.yaml tests/rag
git commit -m "feat: define production ancient-book corpus"
~~~

## Task 2: Parse CP936 tagged books and apply the chapter whitelist

**Files:**
- Create: app/rag/ancient_books/corpus.py
- Create: tests/rag/ancient_books/test_corpus.py
- Modify: app/rag/documents.py

- [ ] **Step 1: Write failing CP936 and whitelist tests**

~~~python
# tests/rag/ancient_books/test_corpus.py
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.rag.ancient_books.corpus import parse_tagged_book, select_sections


class CorpusTests(unittest.TestCase):
    def test_cp936_parser_preserves_directory_title_and_text(self):
        text = (
            "<篇名>景岳全书\n书名：景岳全书\n"
            "<目录>卷之一入集\\\\传忠录（上）\n"
            "<篇名>十问篇（九）\n"
            "属性：一问寒热二问汗。\n"
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "637-景岳全书.txt"
            path.write_bytes(text.encode("cp936"))
            sections = parse_tagged_book(
                path=path,
                book_id="jing_yue_quan_shu",
                book_title="景岳全书",
                encoding="cp936",
            )
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].volume, "卷之一入集")
        self.assertEqual(sections[0].chapter, "传忠录（上）")
        self.assertEqual(sections[0].section, "十问篇（九）")
        self.assertIn("一问寒热", sections[0].original_text)
        self.assertEqual(
            sections[0].source_hash,
            hashlib.sha256(text.encode("cp936")).hexdigest().upper(),
        )

    def test_selection_excludes_formula_and_specialty_titles(self):
        sections = self.make_sections(
            ["头痛论治", "头痛论列方", "产后腹痛", "十问篇（九）"]
        )
        selected = select_sections(
            sections,
            symptom_aliases={"头痛": ["头痛"], "腹痛": ["腹痛"]},
            method_sections=["十问篇（九）"],
            fixed_sections=[],
            symptom_scan=True,
            exclude_title_patterns=["产后", "论列方"],
        )
        self.assertEqual(
            [item.section for item in selected],
            ["头痛论治", "十问篇（九）"],
        )

    @staticmethod
    def make_sections(titles):
        from app.rag.ancient_books.schema import SelectedSection
        return [
            SelectedSection(
                section_id=f"s-{index}",
                source_type="ancient_book",
                book_id="book",
                book_title="书",
                source_file="book.txt",
                source_hash="A" * 64,
                volume="卷一",
                chapter="章",
                section=title,
                symptom_tags=[],
                original_text="正文",
            )
            for index, title in enumerate(titles)
        ]


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Verify the tests fail before implementation**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_corpus -v
~~~

Expected: FAIL because parse_tagged_book and select_sections are missing.

- [ ] **Step 3: Implement tagged-section parsing and deterministic selection**

~~~python
# app/rag/ancient_books/corpus.py
import hashlib
import re
from pathlib import Path

from app.rag.ancient_books.schema import SelectedSection


BLOCK_PATTERN = re.compile(
    r"<目录>(?P<directory>[^\r\n]*)\s*"
    r"<篇名>(?P<title>[^\r\n]+)\s*"
    r"(?P<body>.*?)(?=(?:\r?\n){2,}<目录>|\Z)",
    re.S,
)


def _stable_id(*parts: str) -> str:
    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def parse_tagged_book(
    *,
    path: Path,
    book_id: str,
    book_title: str,
    encoding: str,
) -> list[SelectedSection]:
    raw = path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest().upper()
    text = raw.decode(encoding, errors="strict")
    sections = []
    for match in BLOCK_PATTERN.finditer(text):
        directory = match.group("directory").strip()
        directory_parts = [part for part in directory.split("\\") if part]
        volume = directory_parts[0] if directory_parts else ""
        chapter = directory_parts[-1] if len(directory_parts) > 1 else ""
        title = match.group("title").strip()
        body = re.sub(r"^\s*属性：", "", match.group("body")).strip()
        if not body:
            continue
        sections.append(
            SelectedSection(
                section_id=_stable_id(book_id, directory, title),
                source_type="ancient_book",
                book_id=book_id,
                book_title=book_title,
                source_file=path.name,
                source_hash=source_hash,
                volume=volume,
                chapter=chapter,
                section=title,
                symptom_tags=[],
                original_text=body,
            )
        )
    return sections


def select_sections(
    sections: list[SelectedSection],
    *,
    symptom_aliases: dict[str, list[str]],
    method_sections: list[str],
    fixed_sections: list[str],
    symptom_scan: bool,
    exclude_title_patterns: list[str],
) -> list[SelectedSection]:
    selected = []
    for section in sections:
        title = section.section
        fixed = title in fixed_sections
        method = title in method_sections
        tags = [
            symptom
            for symptom, aliases in symptom_aliases.items()
            if symptom_scan and any(alias in title for alias in aliases)
        ]
        excluded = any(pattern in title for pattern in exclude_title_patterns)
        if fixed or method or (tags and not excluded):
            role_tags = tags or section.symptom_tags
            selected.append(section.model_copy(update={"symptom_tags": role_tags}))
    return sorted(selected, key=lambda item: item.section_id)


def load_curated_sections(
    root: Path,
    symptom_aliases: dict[str, list[str]],
) -> list[SelectedSection]:
    from app.rag.documents import parse_markdown_sections

    selected = []
    for path in sorted(root.glob("*.md")):
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest().upper()
        documents = parse_markdown_sections(
            raw.decode("utf-8"),
            source=str(path),
            filename=path.name,
        )
        for index, document in enumerate(documents):
            topic = str(document.metadata.get("topic", "通用"))
            section = str(document.metadata.get("section", "正文"))
            tags = [
                symptom
                for symptom, aliases in symptom_aliases.items()
                if symptom == topic
                or any(alias in topic or alias in section for alias in aliases)
            ]
            selected.append(
                SelectedSection(
                    section_id=_stable_id(path.name, topic, section, str(index)),
                    source_type="curated_markdown",
                    book_id=f"curated_{path.stem}",
                    book_title="人工整理知识",
                    source_file=path.name,
                    source_hash=source_hash,
                    volume="",
                    chapter=topic,
                    section=section,
                    symptom_tags=tags,
                    original_text=document.page_content,
                )
            )
    return selected
~~~

Keep parse_markdown_sections() in app/rag/documents.py otherwise unchanged. The helper assigns source_type=curated_markdown and ensures existing data/raw/tcm_knowledge.md enters the same build.

- [ ] **Step 4: Run parser tests and existing clarification tests**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_corpus tests.test_clarification_flow -v
~~~

Expected: all tests pass.

- [ ] **Step 5: Commit corpus selection**

~~~powershell
git add app/rag/ancient_books/corpus.py app/rag/documents.py tests/rag/ancient_books/test_corpus.py
git commit -m "feat: select whitelisted ancient-book sections"
~~~

## Task 3: Remove treatment content and build Parent-Child chunks

**Files:**
- Create: app/rag/ancient_books/filters.py
- Create: app/rag/ancient_books/chunking.py
- Create: tests/rag/ancient_books/test_filters_chunking.py

- [ ] **Step 1: Write regression tests for mixed diagnosis and formula text**

~~~python
# tests/rag/ancient_books/test_filters_chunking.py
import unittest

from app.rag.ancient_books.chunking import build_parent_child
from app.rag.ancient_books.filters import filter_retrievable_text
from app.rag.ancient_books.schema import SelectedSection


class FilterChunkTests(unittest.TestCase):
    def test_filter_keeps_features_and_removes_formula_clauses(self):
        text = (
            "\\x外候\\x 因风者恶风，川芎茶调散。"
            "因火者齿痛，连翘、丹皮、桑叶。"
            "\\x用药\\x 黄芩（二钱）白芷（一钱半），水煎。"
        )
        filtered = filter_retrievable_text(text)
        self.assertIn("因风者恶风", filtered)
        self.assertNotIn("川芎茶调散", filtered)
        self.assertNotIn("连翘", filtered)
        self.assertNotIn("黄芩", filtered)
        self.assertNotIn("水煎", filtered)

    def test_chunks_never_exceed_300_and_restore_clean_parent(self):
        section = SelectedSection(
            section_id="s1",
            source_type="ancient_book",
            book_id="book",
            book_title="书",
            source_file="book.txt",
            source_hash="A" * 64,
            volume="卷一",
            chapter="章",
            section="头痛",
            symptom_tags=["头痛"],
            original_text="因风者恶风。因热者烦心恶热。" * 30,
        )
        parents, chunks = build_parent_child(section)
        self.assertTrue(parents)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.text) <= 300 for chunk in chunks))
        self.assertTrue(
            all(chunk.parent_id in {parent.parent_id for parent in parents}
                for chunk in chunks)
        )


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run tests and confirm the missing-filter failure**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_filters_chunking -v
~~~

Expected: FAIL because filters.py and chunking.py do not exist.

- [ ] **Step 3: Implement conservative content filtering**

~~~python
# app/rag/ancient_books/filters.py
import re


EXCLUDED_SUBHEADINGS = {
    "用药", "方剂", "选方", "附方", "快捷方式法", "制法", "服法", "煎服法"
}
FORMULA_CLAUSE = re.compile(
    r"[\u4e00-\u9fff]{1,12}(?:汤|散|丸|饮|膏|丹)(?:。|；|$)|"
    r"(?:汤|散|丸|饮|膏|丹|方)(?:主之|治|加|减|服|下|煎)|"
    r"(?:黄芩|黄连|白芷|川芎|人参|甘草|附子|大黄)[^。；]{0,40}"
    r"(?:钱|两|分|枚|水煎|为末|为丸)"
)
DOSAGE_OR_PREPARATION = re.compile(
    r"[（(][一二三四五六七八九十\d]+(?:钱|两|分|枚)[）)]|"
    r"水煎|为末|为丸|姜汤下|每服|煎服"
)
HERB_NAME = re.compile(
    r"黄芩|黄连|白芷|川芎|人参|甘草|附子|大黄|"
    r"连翘|丹皮|桑叶|羚羊角|山栀|薄荷|菊花|麦冬|柴胡|白芍"
)


def is_excluded_clause(clause: str) -> bool:
    return (
        FORMULA_CLAUSE.search(clause) is not None
        or DOSAGE_OR_PREPARATION.search(clause) is not None
        or len(HERB_NAME.findall(clause)) >= 2
    )


def filter_retrievable_text(text: str) -> str:
    current_heading = ""
    kept = []
    pieces = re.split(r"(\\x[^\\\r\n]+\\x)", text)
    for piece in pieces:
        marker = re.fullmatch(r"\\x([^\\\r\n]+)\\x", piece)
        if marker:
            current_heading = marker.group(1).strip()
            continue
        if any(name in current_heading for name in EXCLUDED_SUBHEADINGS):
            continue
        for sentence in re.split(r"(?<=[。；])", piece):
            clauses = re.split(r"[，,]", sentence)
            safe_clauses = [
                clause.strip()
                for clause in clauses
                if clause.strip()
                and not is_excluded_clause(clause)
            ]
            if safe_clauses:
                kept.append("，".join(safe_clauses).rstrip("。；") + "。")
    return "".join(kept).strip()
~~~

The production filter is intentionally conservative. Add every observed false-negative formula pattern as a named regression test before expanding the regex.

- [ ] **Step 4: Implement deterministic Parent-Child construction**

~~~python
# app/rag/ancient_books/chunking.py
import hashlib
import re

from app.rag.ancient_books.filters import filter_retrievable_text
from app.rag.ancient_books.schema import (
    EvidenceParent,
    RetrievalChunk,
    SelectedSection,
)


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _role(section: SelectedSection) -> str:
    if any(
        marker in section.section
        for marker in ("十问", "问病", "望色", "闻声", "辨息", "切脉", "合色脉")
    ):
        return "diagnostic_method"
    if "脉案" in section.section:
        return "case"
    if "脉候" in section.section:
        return "differential"
    if "病机" in section.section:
        return "pathogenesis"
    return "syndrome_pattern"


def _split_units(text: str, limit: int) -> list[str]:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？；])", text)
        if item.strip()
    ]
    units = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > limit:
            units.append(current)
            current = ""
        if len(sentence) <= limit:
            current += sentence
        else:
            for offset in range(0, len(sentence), limit):
                piece = sentence[offset:offset + limit]
                if piece:
                    units.append(piece)
    if current:
        units.append(current)
    return units


def build_parent_child(
    section: SelectedSection,
) -> tuple[list[EvidenceParent], list[RetrievalChunk]]:
    clean = filter_retrievable_text(section.original_text)
    if not clean:
        return [], []
    parents = []
    chunks = []
    for parent_index, parent_text in enumerate(_split_units(clean, 1000)):
        parent_id = _id("parent", section.section_id, str(parent_index), parent_text)
        parent = EvidenceParent(
            parent_id=parent_id,
            source_type=section.source_type,
            book_id=section.book_id,
            book_title=section.book_title,
            source_file=section.source_file,
            source_hash=section.source_hash,
            volume=section.volume,
            chapter=section.chapter,
            section=section.section,
            symptom_tags=section.symptom_tags,
            evidence_role=_role(section),
            original_text=parent_text,
            normalized_text=" ".join(parent_text.split()),
        )
        parents.append(parent)
        for child_index, child_text in enumerate(_split_units(parent_text, 300)):
            chunks.append(
                RetrievalChunk(
                    chunk_id=_id(
                        "child", parent_id, str(child_index), child_text
                    ),
                    parent_id=parent_id,
                    text=child_text,
                    source_type=section.source_type,
                    symptom_tags=section.symptom_tags,
                    evidence_role=parent.evidence_role,
                )
            )
    return parents, chunks
~~~

- [ ] **Step 5: Run filter/chunk tests**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_filters_chunking -v
~~~

Expected: both tests pass and no formula text appears in generated parents.

- [ ] **Step 6: Commit filtering and chunking**

~~~powershell
git add app/rag/ancient_books/filters.py app/rag/ancient_books/chunking.py tests/rag/ancient_books/test_filters_chunking.py
git commit -m "feat: build safe ancient-book parent child chunks"
~~~

## Task 4: Write deterministic corpus artifacts and a corpus doctor

**Files:**
- Create: app/rag/ancient_books/pipeline.py
- Create: app/rag/ancient_books/cli.py
- Create: tests/rag/ancient_books/test_cli.py

- [ ] **Step 1: Write a failing end-to-end corpus-build test**

~~~python
# tests/rag/ancient_books/test_cli.py
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.rag.ancient_books.pipeline import build_corpus


class PipelineTests(unittest.TestCase):
    def test_build_corpus_writes_deterministic_artifacts(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            body = (
                "<篇名>测试书\n"
                "<目录>卷一\\\\头痛\n"
                "<篇名>头痛\n属性：因风者恶风。\n"
            )
            (source / "book.txt").write_bytes(body.encode("cp936"))
            config = {
                "version": "v1.0.0",
                "source_encoding": "cp936",
                "symptoms": {"头痛": ["头痛"]},
                "exclude_title_patterns": [],
                "books": [{
                    "book_id": "book",
                    "title": "测试书",
                    "source_file": "book.txt",
                    "symptom_scan": True,
                    "method_sections": [],
                    "fixed_sections": [],
                }],
            }
            manifest = build_corpus(
                config=config,
                source_root=source,
                curated_root=None,
                output_dir=root / "out",
            )
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["book_count"], 1)
            self.assertTrue((root / "out" / "parents.jsonl").is_file())
            self.assertTrue((root / "out" / "chunks.jsonl").is_file())
            rows = [
                json.loads(line)
                for line in (root / "out" / "chunks.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["symptom_tags"], ["头痛"])


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the test and verify it fails**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_cli -v
~~~

Expected: FAIL because build_corpus is missing.

- [ ] **Step 3: Implement deterministic JSONL and manifest writing**

In app/rag/ancient_books/pipeline.py implement:

~~~python
def build_corpus(*, config, source_root, curated_root, output_dir):
    sections = []
    source_records = []
    for book in config["books"]:
        path = source_root / book["source_file"]
        parsed = parse_tagged_book(
            path=path,
            book_id=book["book_id"],
            book_title=book["title"],
            encoding=config["source_encoding"],
        )
        selected = select_sections(
            parsed,
            symptom_aliases=config["symptoms"],
            method_sections=book["method_sections"],
            fixed_sections=book["fixed_sections"],
            symptom_scan=book["symptom_scan"],
            exclude_title_patterns=config["exclude_title_patterns"],
        )
        if not selected:
            raise ValueError(f"{book['source_file']} 未选择到任何章节")
        sections.extend(selected)
        source_records.append({
            "book_id": book["book_id"],
            "source_file": book["source_file"],
            "source_sha256": selected[0].source_hash,
            "selected_section_count": len(selected),
            "selected_sections": [
                {
                    "volume": item.volume,
                    "chapter": item.chapter,
                    "section": item.section,
                    "symptom_tags": item.symptom_tags,
                }
                for item in selected
            ],
        })
    if curated_root is not None:
        sections.extend(load_curated_sections(curated_root, config["symptoms"]))
    parents, chunks = [], []
    for section in sections:
        section_parents, section_chunks = build_parent_child(section)
        parents.extend(section_parents)
        chunks.extend(section_chunks)
    if not parents or not chunks:
        raise ValueError("生产语料没有生成可检索 Parent/Child")
    write_jsonl(output_dir / "sections.jsonl", sections)
    write_jsonl(output_dir / "parents.jsonl", parents)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    manifest = build_corpus_manifest(
        version=config["version"],
        sources=source_records,
        sections=sections,
        parents=parents,
        chunks=chunks,
        output_dir=output_dir,
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest
~~~

Add these concrete helpers in the same module:

~~~python
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        records,
        key=lambda item: (
            getattr(item, "section_id", "")
            or getattr(item, "parent_id", "")
            or getattr(item, "chunk_id", "")
        ),
    )
    payload = "".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n"
        for item in ordered
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_corpus_manifest(
    *,
    version,
    sources,
    sections,
    parents,
    chunks,
    output_dir,
):
    parent_ids = [item.parent_id for item in parents]
    chunk_ids = [item.chunk_id for item in chunks]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("生产语料存在重复 parent_id")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("生产语料存在重复 chunk_id")
    known_parents = set(parent_ids)
    orphans = [
        item.chunk_id for item in chunks
        if item.parent_id not in known_parents
    ]
    if orphans:
        raise ValueError(f"生产语料存在孤儿 Child: {orphans[:3]}")
    artifact_paths = {
        name: output_dir / f"{name}.jsonl"
        for name in ("sections", "parents", "chunks")
    }
    return {
        "version": version,
        "status": "ready",
        "book_count": len(sources),
        "section_count": len(sections),
        "parent_count": len(parents),
        "chunk_count": len(chunks),
        "sources": sources,
        "by_source_type": dict(Counter(item.source_type for item in parents)),
        "by_evidence_role": dict(
            Counter(item.evidence_role for item in parents)
        ),
        "by_symptom": dict(
            Counter(tag for item in parents for tag in item.symptom_tags)
        ),
        "files": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        },
    }
~~~

doctor_corpus() must reload and validate all three JSONL files with their Pydantic models, verify the manifest file hashes and counts, reject duplicates/orphans, scan for the exclusion regexes from filters.py, and return the exact count fields asserted by Task 9.

- [ ] **Step 4: Add CLI corpus and doctor commands**

~~~python
# app/rag/ancient_books/cli.py
import argparse
from pathlib import Path

from app.rag.ancient_books.config import load_production_config
from app.rag.ancient_books.pipeline import build_corpus, doctor_corpus


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    corpus = subparsers.add_parser("build-corpus")
    corpus.add_argument("--source-root", type=Path, required=True)
    corpus.add_argument(
        "--config",
        type=Path,
        default=Path("app/rag/config/ancient_books.yaml"),
    )
    corpus.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rag/ancient_books/corpus"),
    )
    subparsers.add_parser("doctor")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "build-corpus":
        config = load_production_config(args.config)
        manifest = build_corpus(
            config=config,
            source_root=args.source_root,
            curated_root=Path("data/raw"),
            output_dir=args.output_dir,
        )
        print(
            f"status={manifest['status']} "
            f"parents={manifest['parent_count']} "
            f"chunks={manifest['chunk_count']}"
        )
    elif args.command == "doctor":
        result = doctor_corpus(Path("data/rag/ancient_books/corpus"))
        print(f"status={result['status']}")


if __name__ == "__main__":
    main()
~~~

- [ ] **Step 5: Run the pipeline tests**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_cli -v
~~~

Expected: the deterministic artifact test passes.

- [ ] **Step 6: Commit the corpus pipeline**

~~~powershell
git add app/rag/ancient_books/pipeline.py app/rag/ancient_books/cli.py tests/rag/ancient_books/test_cli.py
git commit -m "feat: build production ancient-book corpus artifacts"
~~~

## Task 5: Build fixed-revision BGE indexes

**Files:**
- Create: app/rag/ancient_books/models.py
- Create: app/rag/ancient_books/indexing.py
- Create: tests/rag/ancient_books/test_indexing.py
- Modify: requirements.txt
- Modify: app/rag/ancient_books/cli.py

- [ ] **Step 1: Write an index test with a fake encoder**

~~~python
# tests/rag/ancient_books/test_indexing.py
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.rag.ancient_books.indexing import build_index


class FakeEncoder:
    def encode(self, texts):
        return np.asarray(
            [[index + 1.0, 1.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )


class IndexingTests(unittest.TestCase):
    def test_index_writes_rows_tokens_vectors_and_hash_manifest(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks = root / "chunks.jsonl"
            chunks.write_text(
                json.dumps({
                    "chunk_id": "c1",
                    "parent_id": "p1",
                    "text": "头痛恶风",
                    "source_type": "ancient_book",
                    "symptom_tags": ["头痛"],
                    "evidence_role": "syndrome_pattern",
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            manifest = build_index(
                chunks_path=chunks,
                corpus_manifest_sha256="A" * 64,
                output_dir=root / "index",
                encoder=FakeEncoder(),
                model_record={"model": "fake", "revision": "1" * 40},
            )
            self.assertEqual(manifest["row_count"], 1)
            self.assertTrue((root / "index" / "rows.jsonl").is_file())
            self.assertTrue((root / "index" / "bm25_tokens.jsonl").is_file())
            self.assertTrue((root / "index" / "dense.npy").is_file())
            self.assertIn("sha256", manifest["files"]["dense"])


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Verify the test fails before index implementation**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_indexing -v
~~~

Expected: FAIL because indexing.py is missing.

- [ ] **Step 3: Implement model adapters without importing experiments/**

app/rag/ancient_books/models.py must define:

~~~python
class BgeM3Encoder:
    def __init__(self, model_path, settings):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(
            str(model_path),
            use_fp16=bool(settings["use_fp16"]),
            device=settings["device"],
        )
        self.batch_size = int(settings["batch_size"])
        self.max_length = int(settings["max_length"])

    def encode(self, texts):
        result = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        return result["dense_vecs"]


class BgeReranker:
    def __init__(self, model_path, settings):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(
            str(model_path),
            use_fp16=bool(settings["use_fp16"]),
            device=settings["device"],
        )
        self.settings = settings

    def score(self, pairs):
        return self.model.compute_score(
            pairs,
            batch_size=int(self.settings["batch_size"]),
            max_length=int(self.settings["max_length"]),
            normalize=bool(self.settings["normalize_score"]),
        )
~~~

Add the model snapshot helpers and prepare_models() in the same file:

~~~python
import hashlib
import json
from pathlib import Path


def snapshot_files(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or ".cache" in relative.parts:
            continue
        rows.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        })
    if not rows:
        raise ValueError(f"模型快照为空: {root}")
    return rows


def snapshot_tree_sha256(files: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in files
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def prepare_models(*, config, output_dir, manifest_path, downloader=None):
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    existing = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {"version": config["version"]}
    for role in ("embedding", "reranker"):
        settings = config["models"][role]
        model_name = settings["model"].rsplit("/", 1)[-1]
        target = output_dir / model_name / settings["revision"]
        record = existing.get(role, {})
        valid_existing = False
        if target.is_dir() and record.get("revision") == settings["revision"]:
            files = snapshot_files(target)
            valid_existing = (
                snapshot_tree_sha256(files)
                == record.get("snapshot_tree_sha256")
            )
        if not valid_existing:
            target.mkdir(parents=True, exist_ok=True)
            downloader(
                repo_id=settings["model"],
                revision=settings["revision"],
                local_dir=target,
            )
            files = snapshot_files(target)
        manifest[role] = {
            "model": settings["model"],
            "revision": settings["revision"],
            "local_path": target.as_posix(),
            "snapshot_tree_sha256": snapshot_tree_sha256(files),
            "files": files,
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest
~~~

A snapshot is reusable only when its revision and tree hash match the manifest.

- [ ] **Step 4: Implement deterministic index writing**

app/rag/ancient_books/indexing.py must:

1. validate every RetrievalChunk row;
2. sort by chunk_id;
3. tokenize normalized child text with jieba.lcut(text, HMM=False);
4. write rows.jsonl and bm25_tokens.jsonl;
5. encode and L2-normalize float32 dense vectors;
6. reject empty, zero, NaN, or Inf vectors;
7. write dense.npy with allow_pickle=False;
8. hash all three files into manifest.json.

Use the complete normalization contract:

~~~python
def normalize_vectors(vectors, expected_count):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != expected_count:
        raise ValueError("Dense 向量形状不符合索引输入")
    if not np.isfinite(array).all():
        raise ValueError("Dense 向量包含 NaN 或 Inf")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Dense 向量不能为零向量")
    return (array / norms).astype(np.float32, copy=False)
~~~

- [ ] **Step 5: Add production dependencies**

Append only missing production dependencies to requirements.txt:

~~~text
numpy==2.2.6
PyYAML==6.0.3
FlagEmbedding==1.4.0
huggingface_hub==0.36.2
transformers==4.57.6
~~~

Do not add a second torch source; use the existing environment-specific torch installation.

- [ ] **Step 6: Add prepare-models and build-index CLI commands**

The commands must be:

~~~powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli prepare-models
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli build-index
~~~

build-index must verify corpus/manifest.json and chunks.jsonl hashes before loading the embedding model.

- [ ] **Step 7: Run index tests**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_indexing -v
~~~

Expected: the fake-encoder test passes without loading CUDA or network models.

- [ ] **Step 8: Commit index construction**

~~~powershell
git add app/rag/ancient_books/models.py app/rag/ancient_books/indexing.py app/rag/ancient_books/cli.py tests/rag/ancient_books/test_indexing.py requirements.txt
git commit -m "feat: build persistent ancient-book BGE indexes"
~~~

## Task 6: Load verified indexes and implement runtime hybrid retrieval

**Files:**
- Create: app/rag/ancient_books/runtime.py
- Create: tests/rag/ancient_books/test_runtime.py

- [ ] **Step 1: Write failing hybrid, Parent recovery, and integrity tests**

~~~python
# tests/rag/ancient_books/test_runtime.py
import unittest

from app.rag.ancient_books.runtime import (
    reciprocal_rank_fusion,
    recover_parents,
)


class RuntimeTests(unittest.TestCase):
    def test_rrf_merges_bm25_and_dense_with_stable_ties(self):
        merged = reciprocal_rank_fusion(
            {"bm25": ["c2", "c1"], "dense": ["c1", "c3"]},
            rrf_k=60,
        )
        self.assertEqual(merged[0][0], "c1")
        self.assertEqual(
            [chunk_id for chunk_id, _ in merged[1:]],
            sorted(chunk_id for chunk_id, _ in merged[1:]),
        )

    def test_parent_recovery_deduplicates_multiple_children(self):
        hits = [
            {"chunk_id": "c1", "parent_id": "p1", "score": 0.9},
            {"chunk_id": "c2", "parent_id": "p1", "score": 0.8},
            {"chunk_id": "c3", "parent_id": "p2", "score": 0.7},
        ]
        recovered = recover_parents(hits, {"p1": {"parent_id": "p1"},
                                           "p2": {"parent_id": "p2"}})
        self.assertEqual([row["parent_id"] for row in recovered], ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run runtime tests and verify failure**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_runtime -v
~~~

Expected: FAIL because runtime.py is missing.

- [ ] **Step 3: Implement verified loading and retrieval primitives**

app/rag/ancient_books/runtime.py must define a LoadedProductionIndex containing rows, row_by_id, parents, BM25 object, dense vectors, and manifest. load_index() must verify manifest hashes and row counts before constructing BM25Okapi.

Implement deterministic RRF exactly as:

~~~python
def reciprocal_rank_fusion(rankings, *, rrf_k):
    scores = {}
    for source, chunk_ids in sorted(rankings.items()):
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
~~~

Define Parent recovery and the engine contract in the same module:

~~~python
from dataclasses import dataclass

import jieba
import numpy as np


@dataclass
class LoadedProductionIndex:
    rows: list
    row_by_id: dict
    parents: dict
    bm25: object
    dense: np.ndarray
    manifest: dict


def recover_parents(hits, parents):
    recovered = []
    seen = set()
    for hit in hits:
        parent_id = hit["parent_id"]
        if parent_id in seen:
            continue
        parent = parents.get(parent_id)
        if parent is None:
            raise ValueError(f"找不到 Parent: {parent_id}")
        recovered.append({**hit, **parent})
        seen.add(parent_id)
    return recovered


class ProductionRetrievalEngine:
    def __init__(self, *, index, encoder, reranker, settings):
        self.index = index
        self.encoder = encoder
        self.reranker = reranker
        self.settings = settings

    def _eligible(self, chief_symptom):
        return [
            index for index, row in enumerate(self.index.rows)
            if not chief_symptom or chief_symptom in row.symptom_tags
        ]

    def _bm25(self, query, eligible):
        tokens = [
            token.strip()
            for token in jieba.lcut(query, HMM=False)
            if token.strip()
        ]
        scores = self.index.bm25.get_scores(tokens)
        return [
            self.index.rows[index].chunk_id
            for index in sorted(
                eligible,
                key=lambda item: (
                    -float(scores[item]),
                    self.index.rows[item].chunk_id,
                ),
            )[: int(self.settings["bm25_top_k"])]
        ]

    def _dense(self, query, eligible):
        vector = np.asarray(self.encoder.encode([query]), dtype=np.float32)[0]
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(vector).all():
            raise ValueError("查询 Dense 向量无效")
        scores = self.index.dense @ (vector / norm)
        return [
            self.index.rows[index].chunk_id
            for index in sorted(
                eligible,
                key=lambda item: (
                    -float(scores[item]),
                    self.index.rows[item].chunk_id,
                ),
            )[: int(self.settings["dense_top_k"])]
        ]

    def retrieve(self, query, *, chief_symptom, mode="hybrid", top_k=5):
        eligible = self._eligible(chief_symptom)
        if not eligible:
            return {
                "status": "insufficient_evidence",
                "retrieval_mode": mode,
                "degraded": False,
                "degraded_reason": None,
                "results": [],
            }
        bm25_ids = [] if mode == "vector" else self._bm25(query, eligible)
        degraded = False
        degraded_reason = None
        try:
            dense_ids = [] if mode == "keyword" else self._dense(query, eligible)
            rankings = {}
            if bm25_ids:
                rankings["bm25"] = bm25_ids
            if dense_ids:
                rankings["dense"] = dense_ids
            fused = reciprocal_rank_fusion(
                rankings,
                rrf_k=int(self.settings["rrf_k"]),
            )
            candidate_ids = [
                chunk_id for chunk_id, _ in
                fused[: int(self.settings["reranker_candidate_k"])]
            ]
            if mode == "hybrid":
                scores = self.reranker.score([
                    [query, self.index.row_by_id[chunk_id].text]
                    for chunk_id in candidate_ids
                ])
                ranked = sorted(
                    zip(candidate_ids, [float(score) for score in scores]),
                    key=lambda item: (-item[1], item[0]),
                )
            else:
                ranked = [
                    (chunk_id, score)
                    for chunk_id, score in fused
                    if chunk_id in candidate_ids
                ]
        except Exception as error:
            degraded = True
            degraded_reason = str(error)
            mode = "keyword"
            ranked = [(chunk_id, 0.0) for chunk_id in bm25_ids]
        child_hits = [
            {
                "chunk_id": chunk_id,
                "parent_id": self.index.row_by_id[chunk_id].parent_id,
                "score": score,
            }
            for chunk_id, score in ranked
        ]
        parents = recover_parents(child_hits, self.index.parents)[:top_k]
        return {
            "status": "ok" if parents else "insufficient_evidence",
            "retrieval_mode": mode,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "results": parents,
        }
~~~

Runtime retrieval must:

- filter rows by chief_symptom when routing succeeds;
- filter to the six allowed EvidenceRole values enforced by Pydantic;
- retrieve BM25 Top-20 and normalized dense Top-20;
- fuse with RRF k=60;
- score at most 40 child pairs with the reranker;
- sort by descending reranker score then chunk_id;
- recover and deduplicate Parent rows;
- return at most five results with E1-E5 IDs;
- set degraded=true and degraded_reason when dense or reranker explicitly fails;
- never silently drop to keyword-only retrieval.

- [ ] **Step 4: Add explicit degraded-mode tests**

Add a FakeEncoder that raises RuntimeError("model unavailable"). Assert:

~~~python
result = engine.retrieve("头痛恶风", chief_symptom="头痛")
self.assertTrue(result["degraded"])
self.assertEqual(result["retrieval_mode"], "keyword")
self.assertIn("model unavailable", result["degraded_reason"])
~~~

- [ ] **Step 5: Run runtime tests**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_runtime -v
~~~

Expected: RRF, Parent deduplication, hash validation, and explicit degradation tests pass.

- [ ] **Step 6: Commit runtime retrieval**

~~~powershell
git add app/rag/ancient_books/runtime.py tests/rag/ancient_books/test_runtime.py
git commit -m "feat: retrieve verified ancient-book evidence"
~~~

## Task 7: Add ten-symptom routing and conservative query rewriting

**Files:**
- Create: app/rag/ancient_books/query.py
- Create: tests/rag/ancient_books/test_query.py
- Modify: app/rag/terms.py

- [ ] **Step 1: Write routing and no-invented-symptom tests**

~~~python
# tests/rag/ancient_books/test_query.py
import unittest

from app.rag.ancient_books.query import detect_chief_symptom, rewrite_query


class QueryTests(unittest.TestCase):
    def test_routes_all_supported_user_phrases(self):
        cases = {
            "头疼两天": "头痛",
            "感觉天旋地转": "眩晕",
            "晚上一直咳": "咳嗽",
            "活动后气喘": "喘促",
            "心里怦怦跳": "心悸",
            "睡不着而且容易醒": "不寐",
            "胃疼饭后明显": "胃脘痛",
            "肚子痛": "腹痛",
            "最近拉肚子": "泄泻",
            "大便干结难解": "便秘",
        }
        for text, expected in cases.items():
            self.assertEqual(detect_chief_symptom(text), expected)

    def test_rewrite_adds_aliases_but_not_unreported_syndromes(self):
        rewritten = rewrite_query("饭后胃疼，喜按")
        self.assertIn("胃脘痛", rewritten)
        self.assertIn("心痛", rewritten)
        self.assertNotIn("脾胃虚弱", rewritten)
        self.assertNotIn("肝郁气滞", rewritten)


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Verify the tests fail**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_query -v
~~~

Expected: FAIL because query.py is missing.

- [ ] **Step 3: Implement routing with longest-alias priority**

~~~python
# app/rag/ancient_books/query.py
ALIASES = {
    "头痛": ["突发剧烈头痛", "头疼", "头痛"],
    "眩晕": ["天旋地转", "头晕", "眩晕"],
    "咳嗽": ["咳嗽", "咳"],
    "喘促": ["呼吸困难", "气喘", "喘促", "喘"],
    "心悸": ["怦怦跳", "心慌", "心悸"],
    "不寐": ["睡不着", "易醒", "失眠", "不寐"],
    "胃脘痛": ["胃脘痛", "胃疼", "心口痛"],
    "腹痛": ["肚子痛", "腹痛"],
    "泄泻": ["拉肚子", "腹泻", "泄泻"],
    "便秘": ["大便干结", "排便困难", "便秘"],
}
SEARCH_ALIASES = {
    "头痛": ["头痛", "头风"],
    "眩晕": ["眩晕", "眩运", "头眩"],
    "咳嗽": ["咳嗽", "咳逆"],
    "喘促": ["喘促", "喘逆", "上气", "短气"],
    "心悸": ["心悸", "惊悸", "怔忡", "怔仲"],
    "不寐": ["不寐", "不得卧"],
    "胃脘痛": ["胃脘痛", "胃脘", "心痛"],
    "腹痛": ["腹痛", "腹满", "心腹痛"],
    "泄泻": ["泄泻", "下利"],
    "便秘": ["便秘", "秘结", "大便不通"],
}


def detect_chief_symptom(query):
    candidates = [
        (len(alias), symptom)
        for symptom, aliases in ALIASES.items()
        for alias in aliases
        if alias in query
    ]
    return max(candidates, default=(0, None))[1]


def rewrite_query(query):
    symptom = detect_chief_symptom(query)
    terms = [query]
    if symptom:
        terms.extend(SEARCH_ALIASES[symptom])
    return " ".join(dict.fromkeys(term for term in terms if term))
~~~

Update app/rag/terms.py so TCM_TERMS contains the ten standard symptoms and their search aliases. Remove automatic additions of syndrome names such as 脾胃虚弱, 心脾两虚, 肝郁化火, 心肾不交, and 痰热扰心 from QUERY_EXPANSIONS; those terms may be returned only when retrieved evidence contains them.

- [ ] **Step 4: Run query tests**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_query -v
~~~

Expected: all ten phrases route correctly and no unreported syndrome is injected.

- [ ] **Step 5: Commit query routing**

~~~powershell
git add app/rag/ancient_books/query.py app/rag/terms.py tests/rag/ancient_books/test_query.py
git commit -m "feat: route production RAG across ten symptoms"
~~~

## Task 8: Preserve the existing retrieval API and tool contract

**Files:**
- Modify: app/rag/vector_store.py
- Modify: app/rag/bm25_retriever.py
- Modify: app/rag/retriever.py
- Modify: app/rag/build_index.py
- Modify: app/tools/builtins/retrieval_tool.py
- Modify: app/agents/lead_agent/prompt.py
- Create: tests/test_rag_tool.py

- [ ] **Step 1: Write failing compatibility tests**

~~~python
# tests/test_rag_tool.py
import unittest
from unittest.mock import patch

from app.rag.retriever import format_retrieval_results
from app.tools.builtins.retrieval_tool import retrieve_tcm_knowledge


PAYLOAD = {
    "status": "ok",
    "retrieval_mode": "hybrid_parent",
    "degraded": False,
    "original_query": "头痛恶风",
    "rewritten_query": "头痛恶风 头痛 头风",
    "chief_symptom": "头痛",
    "allowed_terms": ["头痛", "恶风"],
    "results": [{
        "citation_id": "E1",
        "content": "因风者恶风。",
        "matched_child": "因风者恶风。",
        "book_title": "证治汇补",
        "source_file": "289-证治汇补.txt",
        "volume": "卷之四",
        "chapter": "上窍门",
        "section": "头痛",
        "evidence_role": "syndrome_pattern",
        "chunk_id": "c1",
        "parent_id": "p1",
        "source_type": "ancient_book",
        "symptom_tags": ["头痛"],
        "retrieval_sources": ["bm25", "dense"],
    }],
}


class RagToolTests(unittest.TestCase):
    def test_formatter_emits_citation_and_source(self):
        text = format_retrieval_results(PAYLOAD)
        self.assertIn("[E1]", text)
        self.assertIn("《证治汇补》", text)
        self.assertIn("头痛", text)
        self.assertNotIn("处方", text)

    @patch("app.tools.builtins.retrieval_tool.retrieve_tcm_docs")
    @patch("app.tools.builtins.retrieval_tool.write_retrieval_log")
    def test_tool_keeps_name_and_logs_stable_evidence_ids(self, log, retrieve):
        retrieve.return_value = PAYLOAD
        result = retrieve_tcm_knowledge.invoke(
            {"query": "头痛恶风", "mode": "hybrid"}
        )
        self.assertIn("[E1]", result)
        self.assertEqual(
            log.call_args.args[0]["final_results"][0]["parent_id"],
            "p1",
        )


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run compatibility tests and verify failure**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.test_rag_tool -v
~~~

Expected: FAIL because the old formatter does not emit E1 book citations and the log omits parent_id.

- [ ] **Step 3: Replace internals while preserving retrieve_tcm_docs()**

app/rag/retriever.py must keep:

~~~python
def retrieve_tcm_docs(
    query: str,
    k: int = 5,
    candidate_k: int = 20,
    mode: str = "hybrid",
) -> dict:
~~~

The implementation must:

1. reject or normalize unknown modes to hybrid;
2. detect chief_symptom and rewrite only with aliases;
3. obtain a cached ProductionRetrievalEngine from vector_store.py;
4. map mode=vector to dense, keyword to BM25, and hybrid to hybrid+reranker+Parent;
5. pass k capped at five;
6. calculate allowed_terms only from returned Parent content;
7. return status, degradation fields, query fields, chief_symptom, results, and allowed_terms.

format_retrieval_results() must render each result as:

~~~text
[E1]
主症：头痛
证据角色：syndrome_pattern
原文：因风者恶风。
来源：《证治汇补》 卷之四 / 上窍门 / 头痛
证据ID：parent_id=p1，chunk_id=c1
~~~

It must end with the existing boundaries: evidence is for health consultation, is not a diagnosis, and cannot support prescriptions or doses.

- [ ] **Step 4: Convert legacy modules into compatibility wrappers**

- vector_store.py: expose get_production_engine() cached by index manifest path and remove MiniMax as the production default.
- bm25_retriever.py: delegate bm25_search() to the loaded production engine.
- build_index.py: call app.rag.ancient_books.cli build-all instead of Chroma.add_documents().
- retrieval_tool.py: request Top-5, log citation_id, source_type, book_title, volume, chapter, section, evidence_role, parent_id, chunk_id, and degraded state.

Do not remove the public function names used by existing imports.

- [ ] **Step 5: Update the lead-agent evidence rules**

Add these exact constraints to app/agents/lead_agent/prompt.py:

~~~text
- 古籍检索结果使用 E1-E5 引用编号；涉及古籍依据时必须在相关表述后标注编号。
- 只能使用本次 retrieve_tcm_knowledge 返回的证候和病机术语。
- 即使古籍原文提及方药，也不得向用户推荐方剂、药物、剂量或煎服法。
- retrieval status 为 insufficient_evidence 时，应继续追问或说明依据不足。
- degraded=true 时，应明确说明当前仅完成降级检索。
~~~

- [ ] **Step 6: Run tool and clarification regression tests**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.test_rag_tool tests.test_clarification_flow tests.test_subagent_clarification -v
~~~

Expected: all tests pass and the existing clarification workflow remains unchanged.

- [ ] **Step 7: Commit API integration**

~~~powershell
git add app/rag app/tools/builtins/retrieval_tool.py app/agents/lead_agent/prompt.py tests/test_rag_tool.py
git commit -m "feat: integrate ancient-book RAG with retrieval tool"
~~~

## Task 9: Build the real local corpus and index

**Files:**
- Local only: data/rag/ancient_books/**
- Commit: app/rag/ancient_books/manifests/corpus-selection-v1.0.0.json
- Commit: app/rag/ancient_books/manifests/index-v1.0.0.json

- [ ] **Step 1: Prepare fixed model snapshots**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli prepare-models
~~~

Expected output includes:

~~~text
embedding_model=BAAI/bge-m3
embedding_revision=5617a9f61b028005a4858fdac845db406aefb181
reranker_model=BAAI/bge-reranker-v2-m3
reranker_revision=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
status=ready
~~~

- [ ] **Step 2: Build the selected local corpus**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli build-corpus --source-root "G:\work\TCM-Ancient-Books-master"
~~~

Expected: status=ready, book_count=7, parent_count greater than zero, chunk_count greater than or equal to parent_count.

- [ ] **Step 3: Inspect the committed selection manifest before indexing**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli doctor
~~~

Expected:

~~~text
status=ready
source_hash_mismatch_count=0
duplicate_parent_count=0
orphan_chunk_count=0
excluded_content_match_count=0
~~~

Open the generated selection summary and verify that all seven books are present, only ten symptom tags occur, no excluded specialty title occurs, and no formula/dosage/preparation role is present.

- [ ] **Step 4: Build the production index**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli build-index
~~~

Expected: status=ready with rows.jsonl, bm25_tokens.jsonl, dense.npy, and verified manifest hashes.

- [ ] **Step 5: Export content-free commit manifests**

First add export-manifests to cli.py and this recursive privacy check to pipeline.py:

~~~python
FORBIDDEN_COMMIT_KEYS = {
    "original_text",
    "normalized_text",
    "content",
    "matched_child",
    "question",
    "answer",
}


def assert_content_free(value):
    if isinstance(value, dict):
        forbidden = FORBIDDEN_COMMIT_KEYS.intersection(value)
        if forbidden:
            raise ValueError(
                f"可提交 Manifest 包含正文键: {sorted(forbidden)}"
            )
        for child in value.values():
            assert_content_free(child)
    elif isinstance(value, list):
        for child in value:
            assert_content_free(child)


def export_manifests(*, corpus_manifest, index_manifest, output_dir):
    payloads = {
        "corpus-selection-v1.0.0.json": corpus_manifest,
        "index-v1.0.0.json": index_manifest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in payloads.items():
        assert_content_free(payload)
        write_json(output_dir / filename, payload)
~~~

The export command copies only source hashes, selected chapter titles, counts, model revisions, artifact hashes, and build status into app/rag/ancient_books/manifests. It rejects all forbidden keys before writing.

Run:

~~~powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli export-manifests
rg -n '"(original_text|normalized_text|content|matched_child|question|answer)"' app/rag/ancient_books/manifests
~~~

Expected: rg exits 1 because none of those raw-content keys are present.

- [ ] **Step 6: Commit content-free manifests**

~~~powershell
git add app/rag/ancient_books/manifests
git commit -m "data: freeze production ancient-book RAG manifests"
~~~

## Task 10: Add fixed smoke checks and operator documentation

**Files:**
- Modify: app/rag/ancient_books/cli.py
- Create: docs/rag/ancient-books-production-rag.md
- Modify: tests/rag/ancient_books/test_cli.py

- [ ] **Step 1: Add a smoke command test without real models**

Use a fake engine and assert the smoke command submits these exact ten queries:

~~~python
SMOKE_QUERIES = {
    "头痛": "头痛恶风，遇冷加重",
    "眩晕": "眩晕伴耳鸣和乏力",
    "咳嗽": "咳嗽有痰，夜间较重",
    "喘促": "活动后喘促并有胸闷",
    "心悸": "心悸反复，劳累后明显",
    "不寐": "入睡困难并且多梦易醒",
    "胃脘痛": "胃脘痛，饭后加重",
    "腹痛": "腹痛，排便后稍缓解",
    "泄泻": "泄泻清稀，受凉后明显",
    "便秘": "大便干结，排出困难",
}
~~~

The command passes when every query returns either status=ok with at least one correctly tagged result, or status=insufficient_evidence with no fabricated result. degraded=true must make the smoke command fail.

- [ ] **Step 2: Run the smoke-command unit test**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.ancient_books.test_cli -v
~~~

Expected: the smoke query set and failure rules pass with the fake engine.

- [ ] **Step 3: Run the real smoke command**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli smoke
~~~

Expected:

~~~text
query_count=10
ok_count=10
degraded_count=0
status=ready
~~~

- [ ] **Step 4: Write the operator document**

docs/rag/ancient-books-production-rag.md must document:

- the seven selected books and ten supported symptoms;
- source privacy and the local-only data directories;
- exact prepare-models, build-corpus, doctor, build-index, and smoke commands;
- startup prerequisites and expected index paths;
- how to interpret ok, insufficient_evidence, and degraded;
- how to rebuild after a source or config change;
- how E1-E5 map to book, volume, chapter, section, parent_id, and chunk_id;
- the prohibition on prescription, dosage, and preparation content;
- troubleshooting for CP936 errors, hash mismatch, missing CUDA/model snapshots, and stale indexes.

- [ ] **Step 5: Commit smoke checks and documentation**

~~~powershell
git add app/rag/ancient_books/cli.py tests/rag/ancient_books/test_cli.py docs/rag/ancient-books-production-rag.md
git commit -m "docs: add ancient-book RAG operations guide"
~~~

## Task 11: Run final verification

**Files:**
- Verify only; do not create experiment artifacts.

- [ ] **Step 1: Run the focused production RAG suite**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/rag/ancient_books -p "test_*.py" -v
.\.venv\Scripts\python.exe -m unittest tests.test_rag_tool -v
~~~

Expected: all production RAG tests pass.

- [ ] **Step 2: Run existing application regression tests**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest tests.test_clarification_flow tests.test_subagent_clarification -v
~~~

Expected: all existing application tests pass.

- [ ] **Step 3: Run existing experiment unit tests as a compatibility check**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/rag_v1_5 -p "test_*.py"
.\.venv\Scripts\python.exe -m unittest discover -s tests/rag_v1_6 -p "test_*.py"
~~~

Expected: all tests pass; production integration has not changed frozen experiment modules.

- [ ] **Step 4: Re-run index doctor and real smoke checks**

Run:

~~~powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli doctor
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli smoke
~~~

Expected: both commands report status=ready and smoke reports degraded_count=0.

- [ ] **Step 5: Verify privacy and scope boundaries**

Run:

~~~powershell
git status --short
git check-ignore data/rag/ancient_books/corpus/parents.jsonl
git check-ignore data/rag/ancient_books/index/dense.npy
rg -n "bootstrap|confidence interval|B4/P 对照|120-case" app/rag/ancient_books docs/rag
~~~

Expected:

- local corpus and dense index are ignored;
- only intended code, tests, docs, and content-free manifests are tracked;
- the final rg command exits 1 because production files contain no experiment workflow.

- [ ] **Step 6: Commit any final verification-only documentation corrections**

If verification required a documentation correction, commit only that correction:

~~~powershell
git add docs/rag/ancient-books-production-rag.md
git commit -m "docs: clarify ancient-book RAG verification"
~~~

If no correction was required, do not create an empty commit.
