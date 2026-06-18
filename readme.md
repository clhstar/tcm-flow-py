每次打开项目终端后，先执行：

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# 启动项目
uvicorn app.main:app --reload --port 2026

# RAG 索引流程
这本《景岳全书》是代码按原始 txt 里的固定标记解析出来的。核心入口在 [corpus.py](G:/work/tcm-flow/app/rag/ancient_books/corpus.py:35)。

整体流程是：

```text
637-景岳全书.txt
  -> parse_tagged_book()
  -> SelectedSection
  -> select_sections()
  -> build_parent_child()
  -> EvidenceParent + RetrievalChunk
  -> rows / bm25_tokens / dense index
```

**1. 原始书的结构怎么读出来**
源文件按 `cp936` 读取，配置在 [ancient_books.yaml](G:/work/tcm-flow/app/rag/config/ancient_books.yaml:1)。

代码主要识别两个标记：

```text
<目录>...
<篇名>...
```

见 [corpus.py](G:/work/tcm-flow/app/rag/ancient_books/corpus.py:9)。

比如原文里类似：

```text
<目录>卷之十七理集·杂证谟\眩运
<篇名>论证（共四条）
属性：眩运一证，虚者居其八九……
```

代码会把它拆成：

```text
volume  = 卷之十七理集·杂证谟
chapter = 眩运
section = 论证（共四条）
text    = 眩运一证，虚者居其八九……
```

这里的 `volume/chapter/section` 就是第一层结构。

**2. 第一层结构：SelectedSection**
解析出来的每个章节片段会变成 `SelectedSection`，字段定义在 [schema.py](G:/work/tcm-flow/app/rag/ancient_books/schema.py:32)。

它包含：

```text
section_id
source_type = ancient_book
book_id = jing_yue_quan_shu
book_title = 景岳全书
source_file = 637-景岳全书.txt
source_hash
volume
chapter
section
symptom_tags
original_text
```

注意：现在已经只允许 `source_type="ancient_book"`，不会再混 `data/raw` 或人工 markdown。

**3. 不是全书都进 RAG，会做章节筛选**
筛选逻辑在 [corpus.py](G:/work/tcm-flow/app/rag/ancient_books/corpus.py:102)。

配置里定义了 10 个症状及别名，比如：

```yaml
眩晕: [眩晕, 眩运, 头眩]
咳嗽: [咳嗽, 咳逆]
腹痛: [腹痛, 腹满, 心腹痛]
```

见 [ancient_books.yaml](G:/work/tcm-flow/app/rag/config/ancient_books.yaml:4)。

筛选规则大致是：

```text
如果 volume/chapter/section 中命中症状词 -> 选入
如果是 method_sections，比如 十问篇（九） -> 选入
如果命中排除词，比如 产后/妇人/小儿/外科/附方/论列方 -> 排除
```

所以它不是整本书全文入库，而是“按症状章节 + 少量问诊章节”选入。

**4. 第二层结构：EvidenceParent**
选中的 `SelectedSection` 会经过 `build_parent_child()`，见 [chunking.py](G:/work/tcm-flow/app/rag/ancient_books/chunking.py:93)。

先过滤掉方药、剂量、煎服等不适合直接检索给用户的内容。然后按句子边界切成较大的 parent 证据块。

`EvidenceParent` 字段在 [schema.py](G:/work/tcm-flow/app/rag/ancient_books/schema.py:45)，主要是：

```text
parent_id
source_type
book_id / book_title / source_file / source_hash
volume / chapter / section
symptom_tags
evidence_role
original_text
normalized_text
```

这个 parent 是最终展示给用户看的“较完整证据上下文”。

**5. 第三层结构：RetrievalChunk**
每个 parent 下面还会切出更小的 child chunk，字段在 [schema.py](G:/work/tcm-flow/app/rag/ancient_books/schema.py:60)。

```text
chunk_id
parent_id
text        # 最长 300 字
source_type
symptom_tags
evidence_role
```

检索时主要先命中这个小 chunk，因为短文本更适合 BM25/向量召回；命中后再恢复到 parent，把完整上下文返回给回答模型。

**6. evidence_role 怎么分**
`evidence_role` 在 [chunking.py](G:/work/tcm-flow/app/rag/ancient_books/chunking.py:30) 判定：

```text
标题含 十问/问病/望色/闻声/辨息/切脉/问诊 -> diagnostic_method
标题含 脉案 -> case
标题含 脉候 或 危险 -> differential
标题含 病机 -> pathogenesis
其他 -> syndrome_pattern
```

所以像“眩运 / 论证（共四条）”通常就是 `syndrome_pattern`。

**7. 最终落地文件**
构建后会生成三类语料文件，逻辑在 [pipeline.py](G:/work/tcm-flow/app/rag/ancient_books/pipeline.py:196)：

```text
sections.jsonl  # SelectedSection
parents.jsonl   # EvidenceParent
chunks.jsonl    # RetrievalChunk
manifest.json   # 数量、hash、来源、症状分布、证据角色分布
```

然后再生成检索索引，见 [indexing.py](G:/work/tcm-flow/app/rag/ancient_books/indexing.py:49)：

```text
rows.jsonl          # chunk 行
bm25_tokens.jsonl   # jieba 分词结果
dense.npy           # BGE-M3 向量
index manifest.json
```

一句话概括：**《景岳全书》先按 `<目录>/<篇名>` 解析成章节结构，再按症状白名单筛选成 `SelectedSection`，然后切成“Parent 证据上下文 + Child 检索片段”，最后生成 BM25 和向量索引。**