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

### “又方”口径冻结

人工确认 `jgy-chapter-25-040` 中“又方：犀角汤”应独立作为第二个
`formula`。当前 Evidence Tree 已满足该口径，因此将
`audit-jin_gui_yao_lue-formula-001` 从 `fail/type_error` 更新为
`pass/correct`，未修改审核表中的任何来源或结构字段。

更新后的结果：

```text
pass=129
fail=11
boundary_error=5
type_error=6
parent_error=0
text_error=0
```

审核 CSV SHA256 更新为：

```text
571F5940C08A3B54AC9BF53CCA68EBFA6ED461B799C2E80A1867163C33C9944D
```

`review-audit` 仍将 Quality Gate 写为 `blocked`，审核模块测试 `8/8`
通过。剩余 11 条失败记录对应 9 个唯一 clause，后续 parser 修复不再调整
“又方”独立 formula 的既定结构。

---

## 2026-06-14：完成解析质量门禁修复并生成复审集

### 目标

修复人工审核确认的 9 个 clause 解析缺陷，保持“又方：犀角汤”为独立
第二个 formula，重建 C0-C4，并仅继承结构完全未变化的旧审核结论。

### 规则与实现

本轮只修改离线实验代码 `experiments/rag_v1_5`，未修改线上 `app/rag`。
主要规则为：

1. formula marker 分离 `boundary_start` 和 `body_start`，下一方从真实方名
   起点切断上一方；
2. 区分显式方名、独立方名标题、方数标记、通用替代方和附方分节；
3. `方二/方三` 后存在明确方名标题时只作计数元数据，不创建幽灵 formula；
4. 支持 `上先`、`上为`、`上各` 作为 preparation 起点；
5. `（方未见）` 生成 note，不生成 ingredients/preparation；
6. 审核迁移按 `(book_id, sample_type, clause_id)` 匹配，且只有
   `chapter_id/evidence_ids/original_text/structured_summary` 全部一致时继承。

对应提交：

```text
0c457e6 test: 覆盖V1.5解析质量门禁缺陷
7206fc3 fix: 修复V1.5方剂解析边界
6abd76a feat: 增加严格人工审核迁移
e999900 fix: 收紧方数与标题泛化边界
e94e8fe data: 重建V1.5解析分块与审核样本
```

### 执行命令

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe -m compileall -q experiments\rag_v1_5 tests\rag_v1_5
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli parse-corpus
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli build-chunks
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli build-chunks
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli sample-audit
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli migrate-audit-review `
  --previous-source data/rag_v1_5/archive/parser-repair-20260614/audit/audit-140.jsonl `
  --previous-reviewed-csv data/rag_v1_5/archive/parser-repair-20260614/audit/audit-140.csv `
  --new-source data/rag_v1_5/audit/audit-140.jsonl `
  --output-csv data/rag_v1_5/audit/audit-140.csv `
  --summary data/rag_v1_5/audit/audit-migration-summary.json
```

修复前 Evidence、Chunk、Audit 和门禁产物已归档到本地：

```text
data/rag_v1_5/archive/parser-repair-20260614/
```

### 目标 clause 验收

| clause_id | 修复后结构 |
| --- | --- |
| `jgy-chapter-23-001` | 1 formula / 1 ingredients / 1 preparation |
| `jgy-chapter-23-002` | 1 formula / 1 ingredients / 1 preparation |
| `jgy-chapter-03-002` | 1 formula / 1 ingredients / 1 preparation |
| `jgy-chapter-16-012` | 1 formula / 1 ingredients / 1 preparation |
| `jgy-chapter-19-002` | 1 formula / 0 ingredients / 0 preparation / 1 note |
| `shl-chapter-10-394` | 1 formula / 1 ingredients / 1 preparation / 1 note |
| `shl-chapter-09-386` | 2 formula / 2 ingredients / 2 preparation |
| `shl-chapter-04-208` | 2 formula / 2 ingredients / 2 preparation |
| `shl-chapter-06-279` | 2 formula / 2 ingredients / 2 preparation |

`jgy-chapter-25-040` 仍为两个 formula：

```text
饮食中毒，烦满，治之方
犀角汤
```

全库 Child Parent 类型检查错误数为 `0`。

### 修复前后全库统计

| 类型 | 修复前 | 修复后 | 差值 |
| --- | ---: | ---: | ---: |
| clause | 918 | 918 | 0 |
| formula | 437 | 440 | +3 |
| ingredients | 351 | 361 | +10 |
| preparation | 325 | 343 | +18 |
| note | 204 | 207 | +3 |

共有 `44` 个 clause 的 Evidence 表示发生变化，其中 `29` 个结构计数变化，
`15` 个只发生 formula 边界文本变化。范围扩大来自同一确定性规则命中的
其他真实实例；调试中已排除“用前方”引用误建 formula 和标题跨入下一方正文
两类范围外变化。

### 测试与确定性结果

- 完整单元测试：`109/109` 通过；
- Python 编译检查通过；
- Evidence Graph Parent 校验通过；
- C0-C4 连续两次构建的 JSONL SHA256 完全一致。

```text
evidence.jsonl
9DBD2ED476F4AD7B89D628CACF6A2F963987C77276F8CFFC8E8FE1BE29F7FF4B

c0.jsonl
2E6AD8225E5ECE5EA305EFEF3D1DFC8532E672178823637BECE2728BF8B3F5B4

c1.jsonl
92F73997A233BF910213AED5A9355EF91F6B719F3F5EFD94A489940636199411

c2.jsonl
4737A00E3AF7F50B680CDF10579FEB98B083A83D8B482A5D19B2629E7E837DC5

c3.jsonl
75978E60EF81805B90D94ACEE4E4C620C87C7D6F3323477E0ACB315A6C0711AC

c4.jsonl
E80F94C71FAA91C812F2E91660FC1E202282D1C19D9D903BABC1655D7D14B090
```

C3 从 `2237` 增加到 `2271`，C4 从 `2281` 增加到 `2315`；C0、C1、C2
哈希不变，说明 clause 原文与篇章级输入未变化，变化集中在结构化 Child。

### 审核迁移结果

新 audit-140 仍按固定种子 `20260612` 生成。迁移结果：

```text
total=140
inherited=126
reset/pending=14
structure_changed=12
missing_new_sample=2
ambiguous=0
```

待人工复审的 14 条为：

```text
audit-jin_gui_yao_lue-clause-003
audit-jin_gui_yao_lue-clause-028
audit-jin_gui_yao_lue-formula-003
audit-jin_gui_yao_lue-formula-016
audit-jin_gui_yao_lue-formula-019
audit-jin_gui_yao_lue-note-or-boundary-010
audit-jin_gui_yao_lue-note-or-boundary-014
audit-shang_han_lun-clause-027
audit-shang_han_lun-formula-010
audit-shang_han_lun-formula-011
audit-shang_han_lun-formula-012
audit-shang_han_lun-formula-015
audit-shang_han_lun-formula-016
audit-shang_han_lun-formula-017
```

迁移后 CSV 中这 14 行的审核字段均为空，其他 126 行保留原审核结论。

```text
audit-140.jsonl
CDBE575495BF1D68CE14978F649F4788967ACA507229A228BD0A47CB2931FA1C

audit-140.csv
2B7B7685C7C6E213441AACA969F9924D1C9A7EC7DF3BDB128336C1D92A7FDE7D

audit-migration-summary.json
546C68B5603A906542BCB1D05D29D67444BFB4CE0C44BD60FED53B74B1FBE46F
```

### 当前结论与下一步

解析修复、真实语料重建、Chunk 确定性验证和审核迁移已完成。
当前没有执行 `review-audit`，Quality Gate 尚未基于新 Evidence 和新审核表
重新冻结，状态仍应视为 `blocked`。下一步只复审 CSV 中 `status=pending`
的 14 行；全部完成后再导入审核并判断能否进入真实索引和 Smoke-10。

## 2026-06-14：Quality Gate ready、真实索引与 Smoke-10 自动运行

### 人工审核导入与门禁冻结

14 条复审完成后，`audit-140.csv` 共 `140/140` 行为 `pass/correct`，
`pending=0`。Excel 将文件保存为 CP936，因此先保留原始备份，再做无损
UTF-8 BOM 转换：

```text
data/rag_v1_5/audit/audit-140.cp936-backup-20260614-170036.csv
SHA256=1407E8F4007FF594907C4C594ABDB67A3C48B38C6917622BEA155D06050C7BE9
```

转换前后 Unicode 文本逐字符一致，随后执行：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli review-audit
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli retrieval-doctor
```

冻结结果：

```text
quality_gate_status=ready
reviewed_count=140
pending_count=0
unresolved boundary/type/parent/text errors=0
reviewer=陈力恒
reviewed_at=2026/6/14
quality_gate_sha256=D3F049B9E5856343ADE07FE054E95EABB1EDDA65EEA9737E09D89F91AF08136B
```

### C0-C4 真实索引

执行：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli build-indexes
```

总耗时 `38.767 s`。索引记录数：

| 策略 | 条数 | Dense 形状 |
| --- | ---: | --- |
| C0 | 198 | `198 × 1024` |
| C1 | 414 | `414 × 1024` |
| C2 | 918 | `918 × 1024` |
| C3 | 2271 | `2271 × 1024` |
| C4 | 2315 | `2315 × 1024` |

独立复核确认每套 rows、BM25 tokens、Dense 和 manifest 的 SHA256 一致，
条数与 Chunk manifest 一致，向量均为有限 `float32`，L2 范数为 `1.0`。

```text
indexes-v1.5.0.json
997F99DAF226AC6D87476759B7075ACC29034886E71CFFA2F0579F551C7D537E
```

### 真实 Reranker 兼容修复

首条 `c4 + hybrid_rerank` 查询暴露环境问题：

```text
AttributeError: XLMRobertaTokenizer has no attribute prepare_for_model
```

当时环境为 `FlagEmbedding 1.4.0 + transformers 5.12.0`。将实验依赖明确
冻结为 `transformers==4.57.6`，同步 `huggingface_hub==0.36.2`，并增加
依赖契约回归测试。`pip check` 无冲突，随后同一查询成功；目标
`jgy-chapter-25-040` 排名第 1，ingredients Child 位于 Top 5，恢复上下文
同时包含“饮食中毒，烦满”和“犀角汤亦佳”。

### Smoke-10 数据集与自动结果

按 evidence-first 原则建立本地 10 条问题：

```text
9 条 answerable
1 条 unanswerable
10 条 approved
dataset_sha256=EC5F2478E51EBED90FC4C7B2A7135AF86F1D1DF50F6E0DEC7A9B6E475007F826
```

执行：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli validate-dataset `
  --dataset data\rag_v1_5\evaluation\smoke-10.jsonl
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli run-smoke `
  --dataset data\rag_v1_5\evaluation\smoke-10.jsonl `
  --strategy c4 `
  --mode hybrid_rerank `
  --output-dir data\rag_v1_5\runs\smoke
```

自动结果：

```text
runtime_errors=0
answerable Hit@5=1.0000
answerable Parent recovery=1.0000
Recall@1=0.6667
Recall@5=1.0000
Recall@10=1.0000
MRR@10=0.7481
nDCG@10=0.8082
Top5 Chunk/Evidence/Clause traceability=PASS
median total latency=414.159 ms
mean total latency=693.363 ms
max total latency=3079.623 ms
```

首题和组成题的 gold 分别位于第 5；煎服法 gold 位于第 3；其余六条
可回答问题 gold 位于第 1。无答案题 Top-1 reranker 分数为 `0.1007`，
仅记录分布，不在本阶段设置拒答阈值。

### 当前结论与限制

自动 Smoke 已超过计划的 `Hit@5 >= 0.70` 门槛，且所有 Top 5 ID 可回溯。
但 `smoke-review.csv` 的 `manual_comment/reviewer/reviewed_at` 仍为
`10/10 pending`，所以 `smoke-10-v1.5.0.json` 状态为
`pending_manual_review`，此时不得将 Task 7 写成最终人工验收通过，也不得
直接进入 Pilot-40 冻结。

提交前完整验证为 `116/116` 单元测试通过，`compileall`、`pip check`、
`git diff --check`、Smoke manifest 文件哈希复核和 `retrieval-doctor`
均通过。

### Smoke-10 人工复核冻结

人工 Top 5 复核于 `2026/6/14` 完成：

```text
reviewed_count=10
pending_count=0
reviewer=陈力恒
answerable hit_at_5 TRUE=9/9
answerable parent_recovery_ok TRUE=9/9
unanswerable 自动列留空=1/1
smoke-review.csv SHA256=
2849E3BC4577AC9E3E20045D471E4F8C73C12D70F36193F571BFE903CC8A5E3D
```

Excel 再次将 CSV 保存为 CP936。已保留原始文件：

```text
data/rag_v1_5/evaluation/smoke-review.cp936-backup-20260614-181712.csv
SHA256=4A6CE7AE30F05D1873B4B3385AF134CADFD3D72EC08CFE09400D39DF371EA080
```

转换后的 UTF-8 BOM 文件与备份的 CSV 字段和值完全一致。

辅助表 `smoke-review-details.csv` 的 `is_gold_clause=FALSE` 只表示该候选
不是预先标注的 gold clause，不表示候选一定无关，也不表示整题失败。
辅助表共 50 个候选，系统原始结果为 `TRUE=11`、`FALSE=39`，人工复核后
与运行产物逐项比对仍为 `0` 处差异。最终门禁只依据正式复核表中的题目级
`hit_at_5`、`parent_recovery_ok` 和人工意见。

Smoke Manifest 已从 `pending_manual_review` 冻结为 `passed`。Task 7
人工门禁完成，可以进入 Pilot-40 的 evidence-first 试标阶段。

## 2026-06-14：Pilot-40 契约、Evidence Group 与审核草稿

### 数据契约和审核工具

按 `2026-06-14-tcm-flow-v1.5-pilot-40.md` 完成前三个代码任务：

```text
794287a test: 定义Pilot-40数据契约
0782b11 feat: 建立Pilot证据组选择
26d4ce2 feat: 建立Pilot双轮人工复核
```

新增门禁覆盖固定 `40/32/8` 分布、两书 × 五类型 × 每格 4 条、
问题文本归一化去重、Evidence Group 映射、Smoke gold 防泄漏、
multi-evidence 同书双条文、UTF-8 BOM 审核表、固定 10 条分层二审、
CP936/GB18030 无损迁移和 immutable 字段防篡改。该阶段数据集测试为
`28/28` 通过，`compileall` 和 `git diff --check` 通过。

### Evidence Group 真实抽样

固定 seed `20260614` 生成 40 个本地 Evidence Group：

```text
answerable_group_count=32
unanswerable_group_count=8
Smoke Evidence overlap=0
Smoke clause overlap=0
answerable anchor clause=40/40 unique
answerable anchor Evidence=55/55 unique
```

方剂组均包含 `formula/ingredients/preparation` Child；source-location
组在候选充足时全部优先选择 `note`。同一 seed 连续运行两次时三份产物
哈希完全一致。无答案检查回填后：

```text
pilot-evidence-groups.jsonl
SHA256=4FBD984DBA9A0B98CEBA17F75735F4E490ACE9C81D1A71BDC569C2C96B88EC27

pilot-exclusions.json
SHA256=A64C1CA29E51DA714BAA078D5513A2690A1A42E4AB0F7FBB2784E181F42536BA

pilot-candidate-report.json
SHA256=1624759F3DE90626FE0ACF69C03977E229EEBB5A723F6544BE624E1F06CF89F0
```

### 40 条草稿和无答案检查

按 evidence-first 顺序本地编写 `32` 条可回答题和 `8` 条无答案题。
可回答题的 support span 均逐字存在于 gold Evidence；未调用外部 LLM API
批量生成问题或答案。

8 条无答案题覆盖 HbA1c、CT、MIC 药敏、随机双盲试验、肌钙蛋白、
MRI、基因测序和胰岛素泵参数。16 个核心词/同义改写在冻结 Evidence
全文中均为零命中。随后保持 C4、模型、`candidate_k=40` 和其他参数不变，
仅在内存中将返回深度扩为 Top 10，执行 `hybrid_rerank`；逐题人工查看
Top 10 后均未发现能够支持现代检测、影像、试验或设备参数答案的条文。

草稿自动校验结果：

```text
question_count=40
answerable_count=32
unanswerable_count=8
approved_count=0
duplicate_question_count=0
multi_evidence_count=8
pilot-40-draft.jsonl SHA256=
7214571F84DD595D2F1DEC48D643421C34308E70F7CB8EE849B07F70924F6FF7
```

已生成 UTF-8 BOM `pilot-review.csv`：

```text
row_count=40
first_status=pending: 40
second_review_required=true: 10
pilot-review.csv SHA256=
2E68603BC4FCC494546D3CE2CE917C6951392A1939F2E7E5C0E6CAD99100ADED
```

### 当前门禁

当前只完成了草稿构造、无答案检索检查和审核表导出。人工第一轮仍为
`0/40 pass`、`40/40 pending`，第二轮也尚未开始，因此尚未生成
`pilot-40.jsonl`，Task 4 仍处于人工审核检查点，不得进入 Pilot Manifest
冻结或真实 8 组矩阵运行。
