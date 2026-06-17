# 《景岳全书》生产 RAG 运维说明

## 范围

项目内生产 RAG 只使用以下古籍：

```text
G:\work\TCM-Ancient-Books-master\637-景岳全书.txt
```

系统不扫描同目录其他书籍，也不从实验目录加载语料或索引。生产配置固定在
`app/rag/config/ancient_books.yaml`，其中 `books` 必须且只能包含
`jing_yue_quan_shu`。

路由支持头痛、眩晕、咳嗽、喘促、心悸、不寐、胃脘痛、腹痛、泄泻、便秘十类主症。
当前《景岳全书》结构化章节可为其中九类返回证据；该书没有独立的成人内科
“胃脘痛/胃痛”篇，因此胃脘痛查询允许返回 `insufficient_evidence`，不得把宽泛的
“脾胃”内容伪标为胃脘痛证据。

## 本地数据

以下目录均被 Git 忽略，只保留在本机：

```text
data/rag/ancient_books/corpus
data/rag/ancient_books/index
data/rag/ancient_books/models
```

原始 CP936 文本、过滤后的证据正文、向量和模型文件不得提交。仓库只提交
`app/rag/ancient_books/manifests` 下不含正文的哈希、计数、章节名和模型版本信息。

## 首次构建

在仓库根目录执行：

```powershell
G:\work\tcm-flow\.venv\Scripts\python.exe -m app.rag.ancient_books.cli prepare-models
G:\work\tcm-flow\.venv\Scripts\python.exe -m app.rag.ancient_books.cli build-corpus --source-root "G:\work\TCM-Ancient-Books-master"
G:\work\tcm-flow\.venv\Scripts\python.exe -m app.rag.ancient_books.cli doctor
G:\work\tcm-flow\.venv\Scripts\python.exe -m app.rag.ancient_books.cli build-index
G:\work\tcm-flow\.venv\Scripts\python.exe -m app.rag.ancient_books.cli export-manifests
G:\work\tcm-flow\.venv\Scripts\python.exe -m app.rag.ancient_books.cli smoke
```

固定模型为：

```text
BAAI/bge-m3 @ 5617a9f61b028005a4858fdac845db406aefb181
BAAI/bge-reranker-v2-m3 @ 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
```

启动应用前，`doctor` 必须是 `status=ready`，`smoke` 必须显示
`degraded_count=0`。当前单书构建允许 `ok_count=9`、`insufficient_count=1`；证据不足项
应为胃脘痛。

## 运行时接入

Agent 继续调用现有工具名 `retrieve_tcm_knowledge`。`retrieve_tcm_docs()` 公开签名也保持
不变，内部已改为读取预构建生产索引：

```text
主症路由 -> BM25 Top-20 + BGE-M3 Top-20 -> RRF(k=60)
-> BGE Reranker(最多40条) -> Parent恢复与去重 -> Top-5
```

结果中的 E1-E5 映射到书名、卷、章、篇、`parent_id` 和 `chunk_id`。回答只能引用本轮
返回的编号与证候/病机术语。

状态含义：

- `ok`：返回了经过标签过滤和 Parent 恢复的证据。
- `insufficient_evidence`：单书中没有满足当前主症和安全过滤的证据，应继续追问或说明依据不足。
- `degraded=true`：Dense 或 Reranker 不可用，仅完成显式 BM25 降级；不得对用户隐瞒。

## 安全边界

导入时会排除妇科、儿科、痘疹、外科等目录，并删除方名、药名、剂量、制法、服法、针灸
和其他治疗指令。生产回答不得推荐方剂、药物、剂量或煎服法。古籍证据只用于健康咨询和
问诊信息整理，不等同于诊断。

## 重建与排错

源文件、配置、过滤规则或模型版本变化后，按“build-corpus -> doctor -> build-index ->
export-manifests -> smoke”的顺序重建。不要复用与新 corpus manifest 哈希不一致的旧索引。

- CP936 解码失败：确认源文件是 `637-景岳全书.txt` 原始版本，且未被编辑器转码。
- source/index hash mismatch：删除本机对应生成目录后按完整顺序重建，不要手工改 JSONL。
- 缺少 CUDA：模型命令无法完成完整索引或 smoke；运行时会显式降级，不能视为验收通过。
- 缺少模型快照：重新执行 `prepare-models`，并核对固定 revision 和模型树哈希。
- stale index：重新执行 `build-index`；运行时加载器会拒绝 corpus manifest 哈希不匹配的索引。

该流程只做项目工程验证，不创建 `experiments/rag_*`、对照矩阵、统计检验或论文结果文件。
