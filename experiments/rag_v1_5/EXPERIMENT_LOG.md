# TCM-Flow V1.5 实验过程记录

本文档按时间顺序记录论文实验的实际过程。设计目标和计划分别见：

- `docs/superpowers/specs/2026-06-12-tcm-flow-v1.5-rag-experiment-design.md`
- `docs/superpowers/plans/2026-06-12-tcm-flow-v1.5-rag-experiment.md`

## 记录规则

每次实验或数据处理调整都应追加以下信息：

1. 日期、目标和实验阶段；
2. 输入语料、Manifest、代码版本和关键参数；
3. 实际执行命令；
4. 观察到的问题和原始样例；
5. 处理决定及其理由；
6. 结果、统计数据和验证方式；
7. 已知局限与下一步。

完整古籍、完整评测集和运行产物只保存在本地 `data/rag_v1_5/`，
仓库仅记录规则、测试、统计摘要和可复现命令。

---

## 2026-06-12：冻结 V1.5 实验范围

### 目标

建立面向硕士论文的可控 RAG 对比实验，研究结构感知切分、
Parent-Child 上下文恢复、混合检索和重排对证据检索及回答忠实度的影响。

### 已冻结决定

- 主实验语料仅使用《伤寒论》和《金匮要略方论》。
- 不在 V1.5 主实验中加入知识图谱和多智能体对比。
- 采用强基线、完整方法和消融实验，不以产品功能数量作为实验结果。
- 无中医专家标注条件下，采用 evidence-first 评测，问题和答案必须能回指原文。
- 语料和完整标注集不公开，只报告来源、哈希、规则和统计。
- 开发集和测试集按证据组划分，固定随机种子 `20260612`。

### 语料快照

来源目录：

```text
G:\work\TCM-Ancient-Books-master
```

源文件 SHA256：

```text
457-伤寒论.txt
EF2FDFA298F1367B9E7501E7C868C6BCDFE8A3ACD7C4991C9262857C606BF462

499-金匮要略方论.txt
617250F7522DA17132A97D7FE6AFD9B128F442E3980163DFAE836C1C98663F7C
```

### 阶段产物

- 实验设计：`docs/superpowers/specs/2026-06-12-tcm-flow-v1.5-rag-experiment-design.md`
- 实施计划：`docs/superpowers/plans/2026-06-12-tcm-flow-v1.5-rag-experiment.md`
- 当前仓库对应设计提交：`813062c`

### 下一步

先实现可追溯的语料导入、Manifest、Evidence Schema 和古籍解析器，
再开展 Chunk、索引、检索和评测数据集实验。

---

## 2026-06-12：完成语料导入与第一版结构化解析

### 目标

将两部 CP936 古籍转换为可复现的 UTF-8 本地快照，并生成带稳定 ID、
父子关系和来源哈希的结构化证据。

### 实现内容

- 导入前校验原始文件 SHA256。
- 只转换编码，不修改原始古籍内容。
- 使用 `<目录>`、`<篇名>` 和编号条文构建篇章与 clause。
- 生成 `clause -> formula -> ingredients/preparation` 父子关系。
- 保留 `original_text`，另生成仅做技术清理的 `normalized_text`。
- 将缺字标记等无法可靠修复的问题写入 `anomalies.jsonl`。
- 生成全库和分书统计。

### 执行命令

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli prepare-corpus
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli parse-corpus
```

### 本地输出

```text
data/rag_v1_5/raw/
data/rag_v1_5/processed/evidence.jsonl
data/rag_v1_5/processed/anomalies.jsonl
data/rag_v1_5/processed/statistics.json
```

### 阶段提交

- `267c579 feat: 实现V1.5古籍语料导入与结构化解析`
- `edc2644 fix: 兼容Windows换行的古籍结构解析`

### 已知局限

第一版方剂解析主要依赖显式 `\x方名\x` 和《伤寒论》的“方一/方二”格式。
《金匮要略方论》后部存在“治之方”“又方”“附方”等混合写法，
需要通过人工抽检继续校正。

---

## 2026-06-13：《金匮要略方论》方剂边界纠错

### 实验阶段

语料结构化质量检查。

### 问题样例

抽检 `jgy-chapter-25-040` 时发现原文包含两个治疗方案：

```text
40．饮食中毒，烦满，治之方∶
苦参（三两） 苦酒（一升半）
上二味，煮三沸，三上三下，之服，吐食出即瘥。
或以水煮亦得。
\x又方∶\x
犀角汤亦佳。
```

第一版结构化结果存在两个问题：

1. “饮食中毒，烦满，治之方”没有生成 formula 子节点；
2. “犀角汤亦佳”被错误生成 ingredients 子节点。

### 原因分析

- 解析器只要发现显式 `\x...\x` 标记，就会跳过同一条文中的隐式方剂识别。
- “又方”和“附方”被统一当作真实方名，没有区分替代方和分节标题。
- 显式方剂不存在“上某味”煎服标记时，正文被无条件归类为 ingredients。

### 处理决定

- 将显式方名、“方一/方二”和“治之方”等标记统一收集，并按原文位置排序。
- `附方` 只作为分节边界，不生成 formula。
- `又方` 和 `治方` 作为通用标签；若正文以“某某汤/丸/散”等开头，
  使用正文中的真实方名。
- 只有存在明确煎服标记，或显式真实方名允许组成识别时，才生成 ingredients。
- 保留 clause 作为 Parent，方剂及其组成、煎服法继续作为 Child。
- 正文没有真实方名的替代治法保留为“又方”，不做无依据命名。

### 预期结构

```text
jgy-chapter-25-040
├── formula-01：饮食中毒，烦满，治之方
│   ├── ingredients：苦参（三两） 苦酒（一升半）
│   └── preparation：上二味……
└── formula-02：犀角汤
```

### 测试过程

先增加回归测试，并确认旧实现只能生成一个 formula，测试失败；随后修改解析器，
使测试通过。另增加 `附方` 分节测试，防止分节标题再次被计入方剂。

验证命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe -m compileall -q experiments\rag_v1_5 tests\rag_v1_5
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli parse-corpus
```

### 验证结果

- 完整单元测试：`37/37` 通过。
- Python 编译检查通过。
- Evidence Graph 校验通过。
- 第 40 条生成 5 个节点：
  - clause：1
  - formula：2
  - ingredients：1
  - preparation：1
- `附方` 作为 formula 标题的数量为 0。
- 全库仍有 6 个匿名“又方”，其正文没有可靠的真实方名，因此保留原标签。

本次重新解析后的本地统计：

| 典籍 | 篇章 | 条文 | 方剂 | 药物组成 | 煎服法 | 校注 | 异常 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 伤寒论 | 10 | 398 | 206 | 134 | 134 | 124 | 2 |
| 金匮要略方论 | 25 | 520 | 231 | 217 | 191 | 80 | 0 |
| 合计 | 35 | 918 | 437 | 351 | 325 | 204 | 2 |

### 当前结论

条文仍是完整上下文 Parent；方剂、组成和煎服法是可检索的结构化 Child。
本次修正提高了《金匮要略方论》混合方剂格式的覆盖率，但规则解析仍然需要
分层抽检，不能把“测试通过”等同于全库结构完全正确。

### 下一步

1. 按篇章、内容类型和特殊标记分层抽检结构化结果；
2. 建立抽检表，记录正确、边界错误、类型错误和父子关系错误；
3. 冻结解析规则后实现 C0-C4 Chunk 策略；
4. 在生成正式 400 条评测数据前，先完成语料结构质量验收。
