# TCM-Flow V1.6 实验日志

## 2026-06-16：公开 TCM-QG 离线代理实验

### 计划复核

执行前先复核了用户指定计划，并做了三项边界修正：

- 原计划指向 `rag_v1_5`，但本轮公开 TCM-QG 实验实际隔离到
  `experiments/rag_v1_6`、`tests/rag_v1_6` 和 `data/rag_v1_6`。
- 原计划中的完整 LLM 回答矩阵会触发大量外部 API 调用。因此 V1.6 默认先运行
  `extractive_oracle_proxy`，并把所有回答结果标注为代理/上界结果，不把它写成模型
  生成结果。
- 原计划中的 BGE-M3 + reranker 检索栈没有沿用 V1.5 产物。V1.6 离线代理阶段使用
  本地 `jieba_bm25_plus_overlap_rerank` 后端，并在 manifest 中记录该后端。

Formal-400 不包含在本次公开实验中，也没有被本阶段修改。

### 数据源与数据集

```text
source_path=train.json
source_sha256=5C79F24114AA01184F19247689D18D412BCBB39CDDE3292A80FAC0DBBDBC7CB6
source_document_count=5881
source_qa_pair_count=18478
normalized_qa_pair_count=18456
train_pool_qa_count=12956
dev_qa_count=2714
test_qa_count=2786
source_manifest_sha256=11673F5B05D930D7D905138FE3730DBBC22FE4841A57B2CFE3075ADDB68F1CDE
dataset_manifest_sha256=97DF11F3DDA98323EF47ED9962280C6E8A821CC5E3BF3816D1167421CA513CA6
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-source
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli prepare-public-tcm-qg-dataset
```

### 分块与索引

```text
b4_chunk_count=6067
child_chunk_count=18108
chunk_manifest_sha256=AABAA8E4BC99A26811D6243AB518B409B5B4365D8EF0477C37962799D935D648
b4_index_manifest_sha256=E41E33A38789A5E4AA415AB228BCFAA389D9D38F5618808C7DF216D361FE701F
child_index_manifest_sha256=A2F6E8E0D11F094ED43A2DC5C5FE1E0F86A41CB6BADB575F097ADD5B3C6EEFE9
backend=jieba_bm25_plus_overlap_rerank
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli build-public-tcm-qg-chunks
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli build-public-tcm-qg-indexes
```

### 检索矩阵

Dev 运行：

```text
run_id=public_tcm_qg_dev-20260616T023519Z-0F5DDE83
question_count=2714
config_count=3
completed_config_count=3
failed_config_count=0
B4_doc_recall_at_5=0.6112748710390568
P_doc_recall_at_5=0.5913780397936624
P_no_parent_doc_recall_at_5=0.5913780397936624
B4_answer_span_hit_at_5=0.7394988946204863
P_answer_span_hit_at_5=0.7295504789977892
P_no_parent_answer_span_hit_at_5=0.5759027266028003
```

Test 运行：

```text
run_id=public_tcm_qg_test-20260616T023812Z-0F5DDE83
question_count=2786
config_count=3
completed_config_count=3
failed_config_count=0
B4_doc_recall_at_5=0.5865039483129936
P_doc_recall_at_5=0.5707106963388371
P_no_parent_doc_recall_at_5=0.5707106963388371
B4_answer_span_hit_at_5=0.7110552763819096
P_answer_span_hit_at_5=0.7020818377602297
P_no_parent_answer_span_hit_at_5=0.5674802584350324
retrieval_matrix_sha256=B3A96BACD8A5D4867CF590DF96E9AA9E2052FB896424F8A538185E3508AD1C3F
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-retrieval-dev
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-retrieval-test
```

### 代理回答与指标

```text
answer_mode=extractive_oracle_proxy
dev_answer_run_id=public_tcm_qg_answer_dev-20260616T024131Z-0F5DDE83
dev_expected_runs=10856
dev_completed_count=10856
dev_error_count=0
test_answer_run_id=public_tcm_qg_answer_test-20260616T024131Z-0F5DDE83
test_expected_runs=11144
test_completed_count=11144
test_error_count=0
answer_matrix_sha256=641D07A71884FCD0A1126138D21A6CFA6940A951FD2CAEBA2D8E001460807048
automatic_metrics_sha256=1D783CD5C27AA09770EF6DC6369A86FF312A74D511F4511E72793FDECA5A00C4
paired_bootstrap_sha256=314FBC7DDA5F87B2F36D72DD9583ABEFADA370982D3726B4D5581E3C7F2DEC55
```

测试集聚合指标：

| 方法 | Char F1 | ROUGE-L F1 | Citation R | Abstain Rate |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.022314 | 0.044871 | N/A | 1.000000 |
| B4 | 0.810695 | 0.815096 | 0.806533 | 0.193467 |
| P | 0.796199 | 0.800729 | 0.791816 | 0.208184 |
| P-no-parent | 0.650481 | 0.657141 | 0.642139 | 0.357861 |

Paired bootstrap 使用 `source_doc_id` 作为重采样单位，重采样 10000 次：

```text
B4-B0 char_f1_delta=0.7883809252760563
B4-B0 95ci=[0.7705816160939957, 0.8057444258949562]
P-B4 char_f1_delta=-0.014495935249461828
P-B4 95ci=[-0.025042544576834466, -0.0040850690804512604]
P-P-no-parent char_f1_delta=0.14571768092581205
P-P-no-parent 95ci=[0.13257302330019544, 0.1592254020181177]
P-P-no-parent citation_recall_delta=0.14967695620961952
P-P-no-parent citation_recall_95ci=[0.1361853832442068, 0.16344086021505377]
```

成功门控：

```text
public_tcm_qg_success=false
skip_mtcmb_tcm_litdata=false
```

原因：在这次离线代理运行中，`P` 相比 `P-no-parent` 有明显提升，但 `P` 在
Char F1 和 citation recall 上低于 `B4`。

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-answer-dev
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-answer-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli summarize-public-tcm-qg-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-runs
```

### 验证

修正重复公开文档下的 citation 指标定义后，V1.6 单元测试通过：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests\rag_v1_6 -p "test_*.py" -v
```

最终验证命令记录在任务收尾中。

## 2026-06-16：公开 TCM-QG 正式 BGE/Reranker 检索启动

### 计划复核

正式计划
`docs/superpowers/plans/2026-06-16-tcm-flow-v1.6-public-tcm-qg-formal-experiment.md`
执行到回答层成本门控。本阶段不把早期 proxy run 当作正式结论，而是恢复 BGE-M3
dense embeddings 与 BGE reranker 检索，冻结正式预注册，构建正式索引，运行完整
dev/test 检索矩阵，并在真实 LLM 回答调用前停在成本审批点。

由于可用的 subagent 工具需要用户显式请求委派，因此本阶段没有使用 subagent；
实际工作在本地通过测试优先修改和命令验证继续推进。

### 正式预注册

```text
status=ready
stage=public_tcm_qg_formal_preregistered
retrieval_config_count=7
answer_methods=B0,B4,P,P-no-parent
answer_model=deepseek-chat
base_url_origin=https://api.deepseek.com
source_manifest_sha256=11673F5B05D930D7D905138FE3730DBBC22FE4841A57B2CFE3075ADDB68F1CDE
dataset_manifest_sha256=97DF11F3DDA98323EF47ED9962280C6E8A821CC5E3BF3816D1167421CA513CA6
chunk_manifest_sha256=AABAA8E4BC99A26811D6243AB518B409B5B4365D8EF0477C37962799D935D648
dataset_sha256=0F5DDE83FA6E3DEC342714458E7F2E0EB1F65E6684E024E673D383EE3962FDBF
prereg_manifest_sha256=8035FE7906913CB2BCC9B0918C9E6BA28395D61A943B921E565BB7B30CA3F313
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-prereg
```

### 正式 BGE-M3 索引

```text
status=ready
stage=public_tcm_qg_formal_indexes_built
backend=bm25_dense_rrf_rerank_ready
embedding_model=BAAI/bge-m3
embedding_revision=5617a9f61b028005a4858fdac845db406aefb181
b4.row_count=6067
child.row_count=18108
b4_index_manifest_sha256=D6C559AFB10AC5FED576160C738B7CAEF107FC9549BC71F2D6CDFB6CEBD92520
child_index_manifest_sha256=4BFBCCFD796E799D47779FF1BD52C803400F8513C956C5421898F5B20E071678
formal_index_manifest_sha256=DD166D6B721918098EEAB2BA092A218B7638B9786E7F1D3D5050CFF32267C762
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli build-public-tcm-qg-formal-indexes
```

### 正式检索矩阵

Dev 运行：

```text
run_id=public_tcm_qg_formal_dev-20260616T034340Z-0F5DDE83
question_count=2714
config_count=7
completed_config_count=7
failed_config_count=0
top5_traceability_rate=1.0 for every config
dev_matrix_summary_sha256=E1FCD14970198F5C909A82E6603AAA9303D92E1185B4602BFE36266298635832
```

Test 运行：

```text
run_id=public_tcm_qg_formal_test-20260616T044542Z-0F5DDE83
question_count=2786
config_count=7
completed_config_count=7
failed_config_count=0
test_matrix_summary_sha256=3117710A141BD04AA638F72361F2D91E8884E12535587AF0E1FCD79842EA2D3C
retrieval_report_sha256=B9A22C912E895B5C9DF8352EAD0D04B3A86B56FC819CBF73CFE20B32E17E75D4
retrieval_runs_manifest_sha256=available at experiments/rag_v1_6/manifests/public-tcm-qg-formal-retrieval-runs-v1.6.0.json
```

测试集检索指标：

| 配置 | 方法角色 | Doc Recall@5 | Answer Span Hit@5 | Answer Span Coverage@5 |
| --- | --- | ---: | ---: | ---: |
| b1-public-bm25 | B1 | 0.591888 | 0.714286 | 0.591855 |
| b2-public-dense | B2 | 0.510768 | 0.634243 | 0.510746 |
| b3-public-hybrid | B3 | 0.589375 | 0.711773 | 0.589346 |
| b4-public-hybrid-rerank | B4 | 0.633166 | 0.751615 | 0.633144 |
| p-public-hybrid-rerank | P | 0.636037 | 0.756281 | 0.636037 |
| p-public-no-parent | P-no-parent | 0.636037 | 0.628500 | 0.596312 |
| p-public-no-reranker | P | 0.584709 | 0.712491 | 0.584709 |

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-formal-retrieval-dev
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-formal-retrieval-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli summarize-public-tcm-qg-formal-retrieval-test
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-retrieval-runs
```

### 回答层成本门控

本阶段没有执行真实 LLM 回答调用，只对完整 test 回答矩阵运行了成本估算：

```text
split=test
question_count=2786
methods=4
repeats=1
expected_calls=11144
estimated_prompt_tokens=10029600
estimated_completion_tokens=2852864
model_name=deepseek-chat
base_url_origin=https://api.deepseek.com
estimated_cost_by_model.deepseek-chat=2.2029459200000003 USD
estimated_wall_time_seconds=11144
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli estimate-public-tcm-qg-formal-answer-cost
```

下一步：用户批准回答层成本后，运行
`freeze-public-tcm-qg-formal-answer-prereg`，再先执行 formal answer dev gate，
然后才允许进行单次冻结的 test answer run。

## 2026-06-16：公开 TCM-QG 正式真实 LLM 回答层

### 成本审批与回答预注册

用户在看到完整 test 成本估算后，确认开始真实 LLM 回答层。估算前复核了 DeepSeek
官方价格：本次运行使用的 `deepseek-chat` 模型别名，cache-miss input tokens 价格为
USD 0.14 / 1M，output tokens 价格为 USD 0.28 / 1M。

正式运行前收紧了回答 prompt，要求 evidence methods 必须：

```text
只能依据给定公开文档证据回答
证据直接支持时必须回答
证据不足时拒答
只输出 JSON，不要输出解释文字
citations 只能使用 E1-E5
```

随后冻结回答预注册：

```text
status=ready
stage=public_tcm_qg_formal_answer_preregistered
answer_methods=B0,B4,P,P-no-parent
temperature=0
repeats=1
max_tokens=256
model_name=deepseek-chat
base_url_origin=https://api.deepseek.com
answer_prereg_sha256=5D5B0F4DBCD5ABA61F003E1CF328894364DDF2E8B5DD0863AAF328DC4CD0F0D0
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-answer-prereg
```

### 正式回答 Dev 门控

```text
run_id=public_tcm_qg_formal_answer_dev-20260616T062209Z-5D5B0F4D
split=dev
question_count=2714
methods=B0,B4,P,P-no-parent
expected_runs=10856
completed_count=10856
error_count=0
json_parse_error_count=0
json_parse_error_rate=0
dev_freeze_status=ready
per_answer_sha256=9DF5155288BF93B7A907775C00949702D63551532786453E02886F4FDA5BA183
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-formal-answer-dev
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-answer-dev
```

### 正式回答 Test 单次冻结运行

```text
run_id=public_tcm_qg_formal_answer_test-20260616T070217Z-5D5B0F4D
split=test
question_count=2786
methods=B0,B4,P,P-no-parent
expected_runs=11144
completed_count=11144
error_count=0
json_parse_error_count=0
json_parse_error_rate=0
per_answer_sha256=C68A29B738307D254716A85CA6142329574F558AC6891055EE75DE2165BCF451
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli run-public-tcm-qg-formal-answer-test
```

### 正式回答自动指标

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli summarize-public-tcm-qg-formal-answer-test
```

测试集聚合指标：

| 方法 | EM | Char F1 | ROUGE-L F1 | Citation R | Unsupported Rate | Abstain Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.001795 | 0.227163 | 0.170536 | N/A | N/A | 0.014716 |
| B4 | 0.201364 | 0.589163 | 0.569148 | 0.823762 | 0.166547 | 0.009691 |
| P | 0.206030 | 0.592457 | 0.572896 | 0.830940 | 0.156856 | 0.012204 |
| P-no-parent | 0.180905 | 0.577587 | 0.558405 | 0.695980 | 0.288586 | 0.015434 |

Paired bootstrap 使用 `source_doc_id` 作为重采样单位，重采样 10000 次：

```text
B4-B0 char_f1_delta=0.36199952472065444
B4-B0 char_f1_95ci=[0.34580876030145596, 0.3782791551276759]
P-B4 char_f1_delta=0.0032942458520593375
P-B4 char_f1_95ci=[-0.0033319224332869365, 0.009995737749563197]
P-B4 citation_recall_delta=0.007178750897343862
P-B4 citation_recall_95ci=[-0.0031768443346276033, 0.01735985533453888]
P-B4 unsupported_answer_rate_delta=-0.009691313711414214
P-P-no-parent char_f1_delta=0.014869524643935912
P-P-no-parent char_f1_95ci=[0.007331003065296627, 0.022610144713476554]
P-P-no-parent citation_recall_delta=0.1349605168700646
P-P-no-parent citation_recall_95ci=[0.12059571376679985, 0.14965239663373583]
P-P-no-parent unsupported_answer_rate_delta=-0.13173007896625988
```

成功门控：

```text
success_gate=parent_ablation_only
strong_success=false
parent_ablation_only=true
failed=false
```

解释：自动指标支持 parent context 相比 child-only `P-no-parent` 的贡献，但不支持
强主张 `P > B4`，因为 `P-B4` 的 Char F1 95% 置信区间下界低于 0。

指标产物哈希：

```text
automatic_metrics_sha256=CD3668B2135851AA7C1294CB692D33C5FE9AC007AA1AAB717448EC8A792E6BF0
paired_bootstrap_sha256=AF3D2007557770DD13F86690399F3C61CD8201ABC4236192FC67AA3547B935E2
success_gate_sha256=68DD4EC8B71CBECDFC4D69BA0CCC8F576ACA98A3EFF5ED3D82DF5A0EAE0643F9
per_question_metrics_sha256=1D6B74C7D0AA53F3976AE17C1B36C15FA168E2F3D1433E22F5392547EE4AC1C9
```

### 人工盲审包

已生成人工盲审包，但此时还没有导入审核标签。因此最终 answer-run 冻结被有意阻塞，
直到 reviewed CSV 标签完成导入。

```text
main_review_questions=160
main_review_rows=640
second_review_rows=64
parent_ablation_focus_questions=120
parent_ablation_rows=240
blind_key_written=true
review_csv_contains_qa_and_generated_answers=true
manifest_contains_raw_content=false
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli prepare-public-tcm-qg-formal-answer-review
```

本地审核文件：

```text
data/rag_v1_6/public_tcm_qg/formal/answer/review/formal-answer-review-main.csv
data/rag_v1_6/public_tcm_qg/formal/answer/review/formal-answer-review-second.csv
data/rag_v1_6/public_tcm_qg/formal/answer/review/formal-answer-review-parent-ablation.csv
data/rag_v1_6/public_tcm_qg/formal/answer/review/formal-answer-review-blind-key.csv
```

### 人工审核导入与最终回答冻结

人工审核完成后，先检查三份 review CSV 的结构和一致性。第一次一致性检查发现 10 行
存在 `answer_correct=1`、`evidence_supported=0` 且 `citation_correct=1` 的组合，
与审核标准冲突。用户修订并重新保存三份 CSV 后，第二次检查结果为：

```text
main_review_rows=640
second_review_rows=64
parent_ablation_rows=240
blank_label_count=0
long_reviewer_comment_count=0
consistency_issue_count=0
```

导入逻辑按最终审核标准修正了 `answer_completeness` 的解释：

```text
3 = 覆盖证据中的关键要点
2 = 覆盖主要要点，但遗漏少量信息
1 = 仅覆盖部分关键要点
0 = 严重遗漏或基本未回答问题
```

执行命令：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli import-public-tcm-qg-formal-answer-review
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli freeze-public-tcm-qg-formal-answer-runs
```

导入后的人工审核摘要：

```text
status=ready
answer_review_completed=true
reviewed_count=640
pending_count=0
second_review_count=64
second_review_pending_count=0
parent_ablation_reviewed_count=240
parent_ablation_pending_count=0
disagreement_count=0
answer_correct_rate=0.69375
evidence_supported_rate=0.740625
citation_correct_rate=0.740625
hallucination_rate=0.0
clinical_safety_issue_rate=0.0046875
answer_completeness_score_3=27
answer_completeness_score_2=64
answer_completeness_score_1=353
answer_completeness_score_0=196
review_summary_sha256=71D84C5B2482DACCCC3402301FF59E68639FE2E679AABC9CD1532FA24D38B7BE
```

最终 answer-run manifest：

```text
status=ready
stage=public_tcm_qg_formal_answer_runs_frozen
success_gate=parent_ablation_only
strong_success=false
parent_ablation_only=true
failed=false
answer_runs_manifest=experiments/rag_v1_6/manifests/public-tcm-qg-formal-answer-runs-v1.6.0.json
answer_runs_manifest_sha256=53BA6D46C8A2C3B69E8DE50E2AF63DCD8C6209E93A0BE294A02D31B1B21D7B32
```

### 中文文档与文件索引

最终实验冻结后，为中文论文交接补充了 V1.6 文档：

```text
experiments/rag_v1_6/README.md
docs/experiments/experiment-file-index.md
docs/experiments/v1.6-public-tcm-qg-results.md
docs/experiments/v1.6-public-tcm-qg-formal-results.md
```

仓库级索引覆盖当前实验报告、计划、V1.6 代码、配置、可提交 manifest、测试和仅本地
保留的产物目录。V1.6 README 也包含该实验专属的中文文件索引。每一项都说明文件作用
和是否应该提交到 Git。文件路径保持原英文形式，确保既有命令、import 和 manifest
哈希仍可复现。

### 实验日志中文化

按用户要求，`experiments/rag_v1_6/EXPERIMENT_LOG.md` 已整体转为中文叙述。
命令、路径、run_id、指标、哈希和成功门控保持不变；历史记录中的 prompt 约束和
`answer_completeness` 审核标准也恢复为正常中文。
