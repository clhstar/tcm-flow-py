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

---

## 2026-06-13：完成 C0-C4 Chunk 构建与确定性验证

### 目标

基于同一份 Evidence Tree 构建五种可比较的切分策略，验证稳定 Chunk ID、
来源 Evidence 映射、条文边界和 C4 Parent 恢复，并冻结输入、配置和输出哈希。

### 参数

| 策略 | 参数 | 说明 |
| --- | --- | --- |
| C0 | `500/80` | 按篇章拼接 clause 后做通用字符切分 |
| C1 | `250/40` | 更细粒度的篇章字符切分基线 |
| C2 | `max_length=1000`，溢出 `500/80` | 普通 clause 不拆分、不与相邻条文合并 |
| C3 | `max_length=500` | 每个 EvidenceUnit 独立，重复标题上下文 |
| C4 | `max_length=300` | Child 独立检索，恢复完整 clause Parent |

字符分隔符按 `段落 -> 换行 -> 句号 -> 分号 -> 逗号 -> 空格` 使用；
运行时追加字符级兜底，防止无标点长文本超过配置上限。

### 执行命令

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-experiment.txt
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli build-chunks
Get-FileHash data\rag_v1_5\chunks\c*.jsonl -Algorithm SHA256
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli build-chunks
Get-FileHash data\rag_v1_5\chunks\c*.jsonl -Algorithm SHA256
```

### 全库结果

| 策略 | Chunk 数 | mean | median | P95 | max | `<100` 比例 | `>500` 比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 198 | 397.88 | 426.0 | 498 | 499 | 0.51% | 0.00% |
| C1 | 414 | 186.63 | 203.0 | 247 | 250 | 8.45% | 0.00% |
| C2 | 918 | 80.35 | 56.5 | 218 | 820 | 71.02% | 0.22% |
| C3 | 2237 | 99.26 | 82.0 | 211 | 488 | 64.15% | 0.00% |
| C4 | 2281 | 98.03 | 82.0 | 211 | 300 | 63.74% | 0.00% |

C4 共恢复到 `918` 个唯一 clause Parent。按唯一 Parent 计算，
完整上下文平均长度为 `80.35` 字，P95 为 `218` 字。

### 输入与输出哈希

```text
corpus manifest
B9FB823BF736AD48F8F45722F971E398BBB755480EC5A125568E0147B342C82D

evidence.jsonl
D0A703699E2947C0FA9132436C1DBDB8C3EF1F1C0DDE831CF81485FC80083B7B

chunks.yaml
EAC1CF154BAC4E887468EE84A38CBCD78A3639300D9B25F54F05035997384548

c0.jsonl
2E6AD8225E5ECE5EA305EFEF3D1DFC8532E672178823637BECE2728BF8B3F5B4

c1.jsonl
92F73997A233BF910213AED5A9355EF91F6B719F3F5EFD94A489940636199411

c2.jsonl
4737A00E3AF7F50B680CDF10579FEB98B083A83D8B482A5D19B2629E7E837DC5

c3.jsonl
EA87719FE9B39F34B1322522282DD6560B20F8721FC1E3CBC66D75CD650CB239

c4.jsonl
C2E49F8FF37F6F93AC7CBADD298B63023C4DD2CA3C0C3C617BD7275B35564A75
```

连续两次构建中，五个 JSONL 的 SHA256 逐一一致。

### 验证结论与限制

- C0-C4 均在两部真实古籍上生成并通过 Chunk Graph 校验。
- Chunk ID 无重复，来源 Evidence 可追溯，C2-C4 不跨 clause。
- C4 每个 Child 唯一恢复到 clause Parent，Parent 上下文保持完整。
- 技术实现满足后续索引烟雾测试的输入条件。
- 140 组正式人工抽检仍未完成，因此暂不冻结正式索引，也不开始
  40 条 evidence-first 试标集的最终制作。

---

## 2026-06-14：完成人工审核并冻结阻断态 Quality Gate

### 目标

导入 140 组人工审核结果，校验不可变列和审核字段，生成问题清单、审核摘要
与 Quality Gate，并验证未通过门禁时真实索引不能构建。

### 执行过程

人工审核表由 Excel/WPS 保存为 CP936。正式导入前保留原始字节备份：

```text
data/rag_v1_5/audit/audit-140.cp936-backup-20260614-130330.csv
```

随后仅进行 CP936 到 UTF-8 BOM 的编码转换，未修改 CSV 字段值。执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli review-audit
.\.venv\Scripts\python.exe -m unittest tests.rag_v1_5.test_audit -v
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli build-indexes
```

### 审核结果

| 指标 | 结果 |
| --- | ---: |
| reviewed | 140 |
| pass | 128 |
| fail | 12 |
| pending | 0 |
| boundary_error | 5 |
| type_error | 7 |
| parent_error | 0 |
| text_error | 0 |

审核人为 `陈力恒`，审核日期为 `2026/6/14`。问题按典籍分布为：

- 《金匮要略方论》：8 条；
- 《伤寒论》：4 条。

### 冻结哈希

```text
audit-140.jsonl
91523494434A5EB58F5ED064DA18832D8EC2116C736FFEFC91FB04BB9D89B399

audit-140.csv
63769B766893A5747F974F93335B48196BC7E62665E9DF08FB49413E62EBBFE6

evidence.jsonl
D0A703699E2947C0FA9132436C1DBDB8C3EF1F1C0DDE831CF81485FC80083B7B

chunk manifest
636A70A86306546C5C761127E2E2752F4C6E22CA29D35F9A4BFD71756960237B
```

### 验证结果

- `review-audit` 成功生成 `audit-issues.jsonl`、`audit-summary.json`
  和 `quality-gate-v1.5.0.json`；
- Quality Gate 状态为 `blocked`；
- 审核模块测试 `8/8` 通过；
- `build-indexes` 以
  `ValueError: Quality Gate 未就绪: status=blocked` 拒绝构建真实索引。

### 当前结论与限制

本轮人工审核已完成，但语料质量门禁未通过，因此不能进入 Smoke-10 和
Pilot-40。12 条失败记录仍需逐条归因：审核表将同一 clause Parent 和其
全部 Child 放在一组，并允许同一 clause 出现在不同抽样层，单纯的展示重复
不等于解析重复。下一阶段必须先区分审核口径误判与真实 parser 边界/类型
缺陷，再针对确认缺陷增加回归测试、重新解析、重建 C0-C4，并复审受影响项。

### 二次复核

对 `audit-jin_gui_yao_lue-formula-001` 和
`audit-jin_gui_yao_lue-note-or-boundary-010` 再次人工复核后，两项仍保留为
`type_error`，批注改为更明确的预期结构说明。复核后的统计保持不变：

```text
pass=128
fail=12
boundary_error=5
type_error=7
```

复核后的审核 CSV SHA256 更新为：

```text
0A18BB8D79CC1F9DF4CDFDE6239627A5F42A0717FC8F515DF73B6AA84431EF5E
```

再次执行 `review-audit` 后，Quality Gate 仍为 `blocked`；审核模块测试
`8/8` 通过。12 条失败记录对应 10 个唯一 clause，初步归因集中在：

1. `方二/方三` 等方数标记与真实方名边界混淆；
2. 未带 `\x...\x` 标记的方名未生成 formula Child；
3. `上先`、`上为末` 未被识别为 preparation 起点；
4. `（方未见）` 被误建为 ingredients，而不是 note；
5. `jgy-chapter-25-040` 的“又方”是否独立建 formula 仍需冻结审核口径。
