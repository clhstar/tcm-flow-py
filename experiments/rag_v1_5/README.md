# TCM-Flow V1.5 Corpus Pipeline

该目录保存论文实验专用代码，与在线服务的 `app/rag` 隔离。

实验设计、执行过程、问题修正和阶段统计持续记录在
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)。

## 运行

从仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-experiment.txt
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli prepare-corpus
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli parse-corpus
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli build-chunks
```

默认输入：

```text
G:\work\TCM-Ancient-Books-master\457-伤寒论.txt
G:\work\TCM-Ancient-Books-master\499-金匮要略方论.txt
```

默认输出：

```text
experiments/rag_v1_5/manifests/corpus-v1.5.0.json
data/rag_v1_5/raw/
data/rag_v1_5/processed/evidence.jsonl
data/rag_v1_5/processed/anomalies.jsonl
data/rag_v1_5/processed/statistics.json
```
raw/：统一编码后的古籍原文。
evidence.jsonl：机器可以理解和检索的结构化古籍。
anomalies.jsonl：机器不敢擅自处理的问题。
statistics.json：本次处理结果的统计报告。

`data/` 已被 Git 忽略，不会提交古籍全文或完整结构化语料。

## C0-C4 Chunk 实验

五种策略读取同一份 `data/rag_v1_5/processed/evidence.jsonl`：

| 策略 | 输入范围 | 规则 |
| --- | --- | --- |
| C0 | 篇章 | 通用字符切分，`500/80` |
| C1 | 篇章 | 通用字符切分，`250/40` |
| C2 | 条文 | 一条 clause 一个 Chunk，超过 1000 字才按 `500/80` 切分 |
| C3 | EvidenceUnit | 结构感知切分，最大 500 字，每段保留书名、篇名和类型 |
| C4 | EvidenceUnit | Child 最大 300 字，检索后恢复完整 clause Parent |

默认输出：

```text
data/rag_v1_5/chunks/c0.jsonl
data/rag_v1_5/chunks/c1.jsonl
data/rag_v1_5/chunks/c2.jsonl
data/rag_v1_5/chunks/c3.jsonl
data/rag_v1_5/chunks/c4.jsonl
data/rag_v1_5/chunks/statistics.json
experiments/rag_v1_5/manifests/chunks-v1.5.0.json
```

`build-chunks` 在写入后校验 Chunk ID、来源 Evidence、书/篇/条文边界、
C4 clause Parent 和字符数。五个 JSONL 使用稳定排序和固定序列化格式，
相同 Evidence 与配置重复运行应得到完全相同的 SHA256。

## 解析规则

- 原始 CP936 文件必须先通过固定 SHA256 校验。
- UTF-8 导入只转换编码，不修订原文。
- `<目录>` 和 `<篇名>` 用于识别卷次与篇章。
- 行首 `属性：N．` 或 `N．` 用于识别编号条文。
- 条文是 Parent；方剂、组成、煎服法和校注是 Child。
- `\x方名\x` 是显式方剂标记。
- 显式方名、“方一/方二”和“治之方”按原文位置统一识别。
- `附方` 是分节边界，不作为方名；`又方` 优先从正文提取真实方名。
- 无可靠方名的替代治法保留为“又方”，不进行推测命名。
- `KT` 缺字进入异常清单，不进行无依据修补。
- 方剂统计表示方剂出现次数；同一方在多个条文中出现会重复计数。

## 数据来源边界

Manifest 记录设计文档声明的源提交，但当前本地快照没有 `.git`，
因此提交状态标记为 `declared_not_locally_verified`。本地快照也没有许可证
文件，全文和完整派生数据仅用于本地研究。
