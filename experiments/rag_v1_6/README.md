# TCM-Flow V1.6 公开 TCM-QG 实验说明

本目录是 V1.6 公开 TCM-QG 实验的独立命名空间。它和
`experiments/rag_v1_5` 分开维护，代码、配置、可提交 manifest 和测试放在
仓库中；原始语料、检索明细、模型回答、人工审核 CSV 等大文件或含文本内容的产物
统一写入 Git 忽略的 `data/rag_v1_6`。

全仓库实验文件的中文总索引见
`docs/experiments/experiment-file-index.md`；本 README 聚焦 V1.6 公开 TCM-QG
实验自身。

本实验使用本地 `train.json` 作为公开 Tianchi TCM-QG 数据源。可提交的
manifest 只保存数量、哈希、聚合指标和隐私标记，不保存题目全文、证据全文或模型
答案全文。

## 实验阶段

V1.6 包含两个阶段：

| 阶段 | 中文名称 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| Offline proxy | 离线代理实验 | 使用 `jieba_bm25_plus_overlap_rerank` 和 `extractive_oracle_proxy` 快速验证公开 TCM-QG 流程与 Parent context 消融方向 | 已冻结 |
| Formal BGE/Reranker/LLM | 正式 BGE/Reranker/LLM 实验 | 使用 BGE-M3、BGE reranker 和真实 `deepseek-chat` 回答矩阵，完成自动指标、人工盲审和最终冻结 | 已冻结 |

离线代理实验只能解释“检索证据是否覆盖参考答案片段”，不能作为真实 LLM 生成效果。
正式实验才是本轮 V1.6 的主要结果来源。

## 执行命令

从仓库根目录执行。

### 离线代理实验

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-source
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli prepare-public-tcm-qg-dataset
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli build-public-tcm-qg-chunks
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli build-public-tcm-qg-indexes
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-retrieval-dev
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-retrieval-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-answer-dev
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-answer-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli summarize-public-tcm-qg-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-runs
```

### 正式实验

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-prereg
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli build-public-tcm-qg-formal-indexes
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-formal-retrieval-dev
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-formal-retrieval-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli summarize-public-tcm-qg-formal-retrieval-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-retrieval-runs
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli estimate-public-tcm-qg-formal-answer-cost
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-answer-prereg
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-formal-answer-dev
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-answer-dev
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-formal-answer-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli summarize-public-tcm-qg-formal-answer-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli prepare-public-tcm-qg-formal-answer-review
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli import-public-tcm-qg-formal-answer-review
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-answer-runs
```

## 实验文件中文索引

### 入口与日志

| 文件 | 中文名称 | 作用 | 是否提交 |
| --- | --- | --- | --- |
| `experiments/rag_v1_6/README.md` | V1.6 实验总说明 | 说明实验阶段、命令、文件结构和各文件作用 | 是 |
| `experiments/rag_v1_6/EXPERIMENT_LOG.md` | V1.6 实验执行日志 | 记录实际执行过的命令、结果哈希、异常修正、审核导入和验证结果 | 是 |
| `experiments/rag_v1_6/__init__.py` | Python 包标记 | 让 `experiments.rag_v1_6` 可作为 Python 模块运行 | 是 |
| `experiments/rag_v1_6/cli.py` | 命令行入口 | 汇总所有 V1.6 子命令，负责调用数据、索引、检索、回答、评估和冻结流程 | 是 |
| `experiments/rag_v1_6/common.py` | 公共工具 | 提供 JSON/JSONL/CSV 读写、哈希、时间戳、文本归一化等通用函数 | 是 |
| `experiments/rag_v1_6/schema.py` | 数据结构定义 | 用 Pydantic 校验公开 TCM-QG 文档、问答、chunk、检索命中和回答记录 | 是 |

### 配置文件

| 文件 | 中文名称 | 作用 | 是否提交 |
| --- | --- | --- | --- |
| `experiments/rag_v1_6/configs/public-tcm-qg.yaml` | 离线代理实验配置 | 固定公开数据源哈希、过滤规则、split、chunk 参数、检索后端、代理回答和 bootstrap 设置 | 是 |
| `experiments/rag_v1_6/configs/public-tcm-qg-formal.yaml` | 正式实验配置 | 固定 BGE-M3、BGE reranker、正式检索参数、LLM 回答矩阵、人工审核抽样和统计设置 | 是 |

### 离线代理实验代码

| 文件 | 中文名称 | 作用 | 是否提交 |
| --- | --- | --- | --- |
| `experiments/rag_v1_6/public_tcm_qg.py` | 公开数据准备 | 校验 `train.json` 哈希，清洗公开问答，生成 doc-level dev/test split 和数据 manifest | 是 |
| `experiments/rag_v1_6/public_tcm_qg_index.py` | 离线 chunk 与索引 | 构建 B4 普通 chunk、P child chunk 和本地 BM25/overlap 索引 | 是 |
| `experiments/rag_v1_6/public_tcm_qg_runner.py` | 离线检索运行器 | 运行 B4、P、P-no-parent 的离线检索矩阵并写出检索摘要 | 是 |
| `experiments/rag_v1_6/public_tcm_qg_answer.py` | 离线代理回答 | 在证据命中参考答案片段时输出代理答案，否则弃答，用于上界/流程验证 | 是 |
| `experiments/rag_v1_6/public_tcm_qg_metrics.py` | 离线指标与冻结 | 计算离线代理指标、paired bootstrap、success gate，并生成可提交 runs manifest | 是 |

### 正式实验代码

| 文件 | 中文名称 | 作用 | 是否提交 |
| --- | --- | --- | --- |
| `experiments/rag_v1_6/public_tcm_qg_formal_index.py` | 正式 BGE 索引 | 构建 BGE-M3 向量索引和正式索引 manifest | 是 |
| `experiments/rag_v1_6/public_tcm_qg_formal_runner.py` | 正式检索运行器 | 运行 7 配置检索矩阵，包含 BM25、Dense、Hybrid、Reranker、Parent ablation 等设置 | 是 |
| `experiments/rag_v1_6/public_tcm_qg_formal_answer.py` | 正式 LLM 回答 | 读取正式检索结果，调用 OpenAI-compatible `deepseek-chat`，生成 B0/B4/P/P-no-parent 回答矩阵 | 是 |
| `experiments/rag_v1_6/public_tcm_qg_formal_metrics.py` | 正式自动指标 | 计算 EM、Char F1、ROUGE-L、citation recall、unsupported rate、paired bootstrap 和最终 answer-run manifest | 是 |
| `experiments/rag_v1_6/public_tcm_qg_formal_review.py` | 人工盲审工具 | 生成盲审 CSV，导入审核标签，校验一致性，并输出人工审核聚合摘要 | 是 |

### 可提交 manifest

| 文件 | 中文名称 | 作用 | 是否提交 |
| --- | --- | --- | --- |
| `experiments/rag_v1_6/manifests/public-tcm-qg-source-v1.6.0.json` | 公开源文件冻结 manifest | 保存 `train.json` 的来源、数量和 SHA256，不保存原文 | 是 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-dataset-v1.6.0.json` | 数据集冻结 manifest | 保存清洗后问答数量、split 数量、数据哈希和隐私标记 | 是 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-runs-v1.6.0.json` | 离线代理结果 manifest | 保存离线代理实验的聚合指标、bootstrap 和 success gate | 是 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-prereg-v1.6.0.json` | 正式检索预注册 manifest | 冻结正式检索模型、参数、环境和输入哈希 | 是 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-indexes-v1.6.0.json` | 正式索引 manifest | 保存正式 BGE 索引构建参数、索引数量和索引哈希 | 是 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-retrieval-runs-v1.6.0.json` | 正式检索结果 manifest | 保存 dev/test 正式检索矩阵摘要、输入哈希和聚合指标 | 是 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-answer-prereg-v1.6.0.json` | 正式回答预注册 manifest | 冻结回答模型、方法矩阵、temperature、repeat、token 设置和检索输入 | 是 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-answer-runs-v1.6.0.json` | 正式回答最终 manifest | 保存真实 LLM 回答自动指标、人工审核摘要、success gate 和最终冻结哈希 | 是 |

最终正式回答 manifest SHA256：

```text
53BA6D46C8A2C3B69E8DE50E2AF63DCD8C6209E93A0BE294A02D31B1B21D7B32
```

### 结果文档与计划文档

| 文件 | 中文名称 | 作用 | 是否提交 |
| --- | --- | --- | --- |
| `docs/experiments/v1.6-public-tcm-qg-results.md` | V1.6 离线代理实验结果 | 说明离线代理实验的输入、方法、检索指标、代理答案指标、门控和局限 | 是 |
| `docs/experiments/v1.6-public-tcm-qg-formal-results.md` | V1.6 正式实验结果 | 说明正式检索、真实 LLM 回答、人工盲审、最终门控和论文可写结论 | 是 |
| `docs/superpowers/plans/2026-06-16-tcm-flow-v1.5-public-tcm-qg-eval.md` | 公开 TCM-QG 初始执行计划 | 记录从 V1.5 路径设想到 V1.6 落地前的公开 TCM-QG 实验设计意图 | 是 |
| `docs/superpowers/plans/2026-06-16-tcm-flow-v1.6-public-tcm-qg-formal-experiment.md` | V1.6 正式实验执行计划 | 记录正式 BGE/Reranker/LLM 实验的假设、门控、步骤和人工审核要求 | 是 |

### 测试文件

| 文件 | 中文名称 | 作用 | 是否提交 |
| --- | --- | --- | --- |
| `tests/rag_v1_6/__init__.py` | 测试包标记 | 让 V1.6 测试目录可作为 Python 测试包发现 | 是 |
| `tests/rag_v1_6/test_public_tcm_qg.py` | 公开数据准备测试 | 覆盖源文件冻结、清洗、split 和隐私 manifest 行为 | 是 |
| `tests/rag_v1_6/test_public_tcm_qg_index.py` | 离线索引测试 | 覆盖 chunk、parent/child 结构和本地索引产物 | 是 |
| `tests/rag_v1_6/test_public_tcm_qg_runner.py` | 离线检索测试 | 覆盖离线检索矩阵、恢复运行和指标摘要 | 是 |
| `tests/rag_v1_6/test_public_tcm_qg_metrics.py` | 离线指标测试 | 覆盖代理回答指标、bootstrap、success gate 和隐私 manifest | 是 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_index.py` | 正式索引测试 | 覆盖 BGE 索引构建和正式索引 manifest | 是 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_runner.py` | 正式检索测试 | 覆盖正式检索矩阵、预注册校验和冻结 manifest | 是 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_answer.py` | 正式回答测试 | 覆盖 LLM 回答矩阵、成本估算、恢复运行和回答预注册 | 是 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_metrics.py` | 正式指标测试 | 覆盖自动指标、bootstrap、success gate、最终 manifest 和隐私扫描回归 | 是 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_review.py` | 人工审核测试 | 覆盖盲审包生成、GB18030/CP936 CSV 导入、数值标签归一化和一致性检查 | 是 |

### 本地生成产物

以下文件或目录不提交到 Git。它们可能包含公开题目全文、证据全文、模型回答或人工审核
内容，只作为本地复现实验使用。

| 路径 | 中文名称 | 作用 | 是否提交 |
| --- | --- | --- | --- |
| `train.json` | 本地公开数据源 | Tianchi TCM-QG 原始公开数据文件，作为实验输入 | 否 |
| `data/rag_v1_6/public_tcm_qg/processed/` | 清洗后数据 | 保存规范化 JSONL、split 文件和中间统计 | 否 |
| `data/rag_v1_6/public_tcm_qg/chunks/` | chunk 产物 | 保存 B4 chunk、child chunk 和 chunk manifest | 否 |
| `data/rag_v1_6/public_tcm_qg/indexes/` | 离线索引 | 保存离线 BM25/overlap 索引 | 否 |
| `data/rag_v1_6/public_tcm_qg/runs/` | 离线检索明细 | 保存离线 dev/test 检索矩阵和逐题结果 | 否 |
| `data/rag_v1_6/public_tcm_qg/answer/` | 离线代理答案 | 保存离线代理回答和自动指标明细 | 否 |
| `data/rag_v1_6/public_tcm_qg/formal/indexes/` | 正式 BGE 索引 | 保存 BGE-M3 embedding、BM25 数据和正式索引文件 | 否 |
| `data/rag_v1_6/public_tcm_qg/formal/runs/` | 正式检索明细 | 保存正式 dev/test 检索矩阵和逐题证据 | 否 |
| `data/rag_v1_6/public_tcm_qg/formal/answer/` | 正式回答与审核 | 保存真实 LLM 回答、自动指标明细、盲审 CSV 和人工审核导入结果 | 否 |

## 最终结论

正式实验最终门控为：

```text
success_gate=parent_ablation_only
strong_success=false
parent_ablation_only=true
failed=false
```

论文中可以稳妥表述：Parent context 相比 child-only 能改善证据扩展和回答可追溯性。
但不能写成 Parent-Child 方法在答案级整体显著优于 B4 hybrid-rerank baseline。
