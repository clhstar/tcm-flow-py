# TCM-Flow 实验文件总索引

本索引用中文说明当前仓库中的实验相关文件。文件路径继续保留英文原名，原因是这些
路径已经被命令行入口、Python import、manifest 哈希和实验日志引用；中文说明放在
本文档和各实验 README 中，避免破坏可复现性。

## 结果报告

| 文件 | 中文名称 | 作用 | 提交策略 |
| --- | --- | --- | --- |
| `docs/experiments/v1.5-retrieval-pilot-summary.md` | V1.5 检索 pilot 总结 | 汇总 V1.5 检索 pilot 的执行状态、阶段结论和进入正式实验前的依据 | 提交 |
| `docs/experiments/v1.5-formal-retrieval-results.md` | V1.5 Formal-400 检索结果 | 记录古籍 Formal-400 检索阶段的正式结果、指标和论文边界 | 提交 |
| `docs/experiments/v1.6-public-tcm-qg-results.md` | V1.6 公开 TCM-QG 离线代理结果 | 记录公开 TCM-QG 离线代理实验的输入、方法、指标、门控和局限 | 提交 |
| `docs/experiments/v1.6-public-tcm-qg-formal-results.md` | V1.6 公开 TCM-QG 正式实验结果 | 记录正式 BGE/Reranker/LLM 实验、人工盲审、最终门控和论文可写结论 | 提交 |
| `docs/experiments/experiment-file-index.md` | 实验文件总索引 | 用中文解释实验文件路径、中文名称、作用和提交策略 | 提交 |

## 执行计划

| 文件 | 中文名称 | 作用 | 提交策略 |
| --- | --- | --- | --- |
| `docs/superpowers/plans/2026-06-12-tcm-flow-v1.5-rag-experiment.md` | V1.5 RAG 实验总计划 | 定义论文实验主线、baseline/ablation 框架和总体执行范围 | 提交 |
| `docs/superpowers/plans/2026-06-13-tcm-flow-v1.5-chunking.md` | V1.5 chunking 子实验计划 | 规划 C0-C4 分块策略、质量检查和实验日志记录方式 | 提交 |
| `docs/superpowers/plans/2026-06-13-tcm-flow-v1.5-retrieval-pilot.md` | V1.5 检索 pilot 计划 | 规划 audit、quality gate、smoke、Pilot-40 到正式检索的阶段流程 | 提交 |
| `docs/superpowers/plans/2026-06-14-tcm-flow-v1.5-parser-quality-gate-repair.md` | 解析器质量门修复计划 | 记录古籍解析器质量门问题的修复、回归测试和验证路径 | 提交 |
| `docs/superpowers/plans/2026-06-14-tcm-flow-v1.5-pilot-40.md` | V1.5 Pilot-40 计划 | 规划 40 题检索 pilot 的冻结输入、指标和报告输出 | 提交 |
| `docs/superpowers/plans/2026-06-14-tcm-flow-v1.5-formal-400-retrieval.md` | V1.5 Formal-400 检索计划 | 规划古籍 Formal-400 检索矩阵、冻结 manifest 和 bootstrap 统计 | 提交 |
| `docs/superpowers/plans/2026-06-16-tcm-flow-v1.5-formal-400-answer-level.md` | V1.5 Formal-400 答案级交接计划 | 说明正式检索冻结后如何进入 B0/B4/P/P-no-parent 答案级评估 | 提交 |
| `docs/superpowers/plans/2026-06-16-tcm-flow-v1.5-public-tcm-qg-eval.md` | 公开 TCM-QG 初始计划 | 记录公开 TCM-QG 主实验从 V1.5 路径设想到 V1.6 实现前的设计意图 | 提交 |
| `docs/superpowers/plans/2026-06-16-tcm-flow-v1.6-public-tcm-qg-formal-experiment.md` | V1.6 公开 TCM-QG 正式实验计划 | 规划 BGE/Reranker/LLM 正式实验、人工盲审和最终成功门控 | 提交 |

## V1.6 实验代码

| 文件 | 中文名称 | 作用 | 提交策略 |
| --- | --- | --- | --- |
| `experiments/rag_v1_6/__init__.py` | V1.6 实验包标记 | 让 V1.6 实验目录可作为 Python 包导入和运行 | 提交 |
| `experiments/rag_v1_6/README.md` | V1.6 实验总说明 | 说明 V1.6 阶段、命令、文件中文索引、最终门控和本地数据边界 | 提交 |
| `experiments/rag_v1_6/EXPERIMENT_LOG.md` | V1.6 实验执行日志 | 记录真实执行命令、修复、审核导入、冻结哈希和验证结果 | 提交 |
| `experiments/rag_v1_6/cli.py` | V1.6 命令行入口 | 暴露公开 TCM-QG 数据、索引、检索、回答、审核、指标和冻结命令 | 提交 |
| `experiments/rag_v1_6/common.py` | 公共工具模块 | 提供 JSON/JSONL/CSV 读写、SHA256、时间戳、文本指标等复用工具 | 提交 |
| `experiments/rag_v1_6/schema.py` | 数据结构模块 | 定义公开 TCM-QG 文档、问答、chunk、检索命中和回答记录的校验 schema | 提交 |
| `experiments/rag_v1_6/public_tcm_qg.py` | 公开数据准备模块 | 校验源文件，清洗问答，生成 split 和数据 manifest | 提交 |
| `experiments/rag_v1_6/public_tcm_qg_index.py` | 离线索引模块 | 构建离线 B4 chunk、child chunk 和 BM25/overlap 索引 | 提交 |
| `experiments/rag_v1_6/public_tcm_qg_runner.py` | 离线检索模块 | 运行离线代理检索矩阵和检索摘要 | 提交 |
| `experiments/rag_v1_6/public_tcm_qg_answer.py` | 离线代理回答模块 | 根据证据是否包含参考答案片段生成代理回答或弃答 | 提交 |
| `experiments/rag_v1_6/public_tcm_qg_metrics.py` | 离线指标模块 | 计算离线代理指标、bootstrap、success gate 和 runs manifest | 提交 |
| `experiments/rag_v1_6/public_tcm_qg_formal_index.py` | 正式 BGE 索引模块 | 构建正式 BGE-M3 向量索引和索引 manifest | 提交 |
| `experiments/rag_v1_6/public_tcm_qg_formal_runner.py` | 正式检索模块 | 运行正式 BGE/Reranker 检索矩阵并冻结检索结果 | 提交 |
| `experiments/rag_v1_6/public_tcm_qg_formal_answer.py` | 正式 LLM 回答模块 | 调用 `deepseek-chat` 生成 B0/B4/P/P-no-parent 回答矩阵 | 提交 |
| `experiments/rag_v1_6/public_tcm_qg_formal_metrics.py` | 正式指标模块 | 计算自动答案指标、paired bootstrap、success gate 和最终 answer-run manifest | 提交 |
| `experiments/rag_v1_6/public_tcm_qg_formal_review.py` | 人工盲审模块 | 生成盲审 CSV，导入审核标签，检查一致性并汇总人工审核指标 | 提交 |

## V1.6 配置和 manifest

| 文件 | 中文名称 | 作用 | 提交策略 |
| --- | --- | --- | --- |
| `experiments/rag_v1_6/configs/public-tcm-qg.yaml` | 离线代理配置 | 冻结公开数据源、过滤规则、split、chunk、代理回答和统计参数 | 提交 |
| `experiments/rag_v1_6/configs/public-tcm-qg-formal.yaml` | 正式实验配置 | 冻结 BGE/Reranker/LLM、人工审核和 bootstrap 设置 | 提交 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-source-v1.6.0.json` | 源文件 manifest | 保存公开源文件数量和哈希，不保存原文 | 提交 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-dataset-v1.6.0.json` | 数据集 manifest | 保存清洗后数据集和 split 的数量、哈希和隐私标记 | 提交 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-runs-v1.6.0.json` | 离线 runs manifest | 保存离线代理实验的聚合指标和门控 | 提交 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-prereg-v1.6.0.json` | 正式检索预注册 manifest | 冻结正式检索输入、模型和参数 | 提交 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-indexes-v1.6.0.json` | 正式索引 manifest | 保存正式索引构建信息和索引哈希 | 提交 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-retrieval-runs-v1.6.0.json` | 正式检索 runs manifest | 保存正式 dev/test 检索矩阵摘要和结果哈希 | 提交 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-answer-prereg-v1.6.0.json` | 正式回答预注册 manifest | 冻结回答模型、方法矩阵和检索输入 | 提交 |
| `experiments/rag_v1_6/manifests/public-tcm-qg-formal-answer-runs-v1.6.0.json` | 正式回答最终 manifest | 保存自动指标、人工审核摘要、最终 success gate 和冻结哈希 | 提交 |

## V1.6 测试

| 文件 | 中文名称 | 作用 | 提交策略 |
| --- | --- | --- | --- |
| `tests/rag_v1_6/__init__.py` | V1.6 测试包标记 | 让测试目录可被 unittest 发现 | 提交 |
| `tests/rag_v1_6/test_public_tcm_qg.py` | 公开数据准备测试 | 覆盖源文件冻结、清洗、split 和隐私 manifest 行为 | 提交 |
| `tests/rag_v1_6/test_public_tcm_qg_index.py` | 离线索引测试 | 覆盖 chunk、parent/child 结构和索引产物 | 提交 |
| `tests/rag_v1_6/test_public_tcm_qg_runner.py` | 离线检索测试 | 覆盖离线检索矩阵、恢复运行和指标摘要 | 提交 |
| `tests/rag_v1_6/test_public_tcm_qg_metrics.py` | 离线指标测试 | 覆盖代理指标、bootstrap、success gate 和隐私 manifest | 提交 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_index.py` | 正式索引测试 | 覆盖 BGE 索引构建和正式索引 manifest | 提交 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_runner.py` | 正式检索测试 | 覆盖正式检索矩阵、预注册校验和冻结 manifest | 提交 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_answer.py` | 正式回答测试 | 覆盖 LLM 回答矩阵、成本估算、恢复运行和回答预注册 | 提交 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_metrics.py` | 正式指标测试 | 覆盖自动指标、bootstrap、success gate、最终 manifest 和隐私扫描回归 | 提交 |
| `tests/rag_v1_6/test_public_tcm_qg_formal_review.py` | 人工审核测试 | 覆盖盲审包、CSV 编码导入、数值标签归一化和一致性检查 | 提交 |

## 不提交的本地文件

| 路径 | 中文名称 | 作用 | 提交策略 |
| --- | --- | --- | --- |
| `train.json` | 本地公开数据源 | Tianchi TCM-QG 原始输入文件，可能包含完整题文和答案 | 不提交 |
| `data/rag_v1_6/` | V1.6 本地实验产物 | 保存清洗数据、chunk、索引、检索明细、模型回答和人工审核 CSV | 不提交 |
| `2020版中国药典（1部）.pdf` | 本地参考 PDF | 本地资料文件，不属于本次可复现实验提交内容 | 不提交 |
