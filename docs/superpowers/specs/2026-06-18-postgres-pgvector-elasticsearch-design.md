# TCM-Flow Postgres + pgvector + Elasticsearch 持久化设计

> 状态：设计已确认，等待书面复核
> 日期：2026-06-18
> 方案：C，Postgres + pgvector + Elasticsearch 一次性接入
> 定位：工程化持久化与检索底座升级，不新增论文实验

## 1. 目标

为 `tcm-flow` 增加持久化数据库和可部署检索底座，使系统能够保存对话、运行记录、LangGraph 上下文、RAG 语料元数据、向量索引和关键词索引。

本设计要解决三个问题：

```text
1. 服务重启后 Thread / Run / Message / Checkpointer 历史不丢失；
2. 生产 RAG 不再只依赖 rows.jsonl / bm25_tokens.jsonl / dense.npy 运行；
3. Dense 向量检索和 BM25 关键词检索具备可部署、可重建、可校验的存储边界。
```

当前 V1.7 生产 RAG 已经将《景岳全书》构建为稳定的 Parent-Child 索引：

```text
parent_count = 2288
chunk_count = 2288
vector_dimension = 1024
embedding_model = BAAI/bge-m3
```

数据库化应复用这些已验证产物和 manifest 校验，不在本阶段扩书、不调参、不创建新的实验矩阵。

## 2. 总体架构

系统采用三层存储职责：

```text
Postgres
  -> 唯一事实来源
  -> 保存业务状态、Agent 历史、RAG 语料和索引元数据

pgvector
  -> Postgres 扩展
  -> 保存 RetrievalChunk 的 dense embedding
  -> 执行向量 Top-K 召回

Elasticsearch
  -> 派生关键词索引
  -> 保存可重建的 chunk 搜索文档
  -> 执行 BM25、字段权重、中文分词和调试高亮
```

运行时链路：

```text
FastAPI / Agent Runtime
  -> Postgres: threads / runs / messages / checkpoints
  -> Postgres: rag_sources / rag_parents / rag_chunks
  -> pgvector: dense search over rag_chunk_embeddings.embedding
  -> Elasticsearch: keyword search over chunk documents
  -> RRF fusion
  -> reranker
  -> parent recovery from Postgres
  -> E1-E5 evidence
```

Postgres 是唯一事实来源。Elasticsearch 中的任何文档都必须能从 Postgres 重建；ES 不保存不可恢复的业务状态。

## 3. 范围边界

### 3.1 本阶段提供

- 持久化 Thread、Run、Message、前端可见 conversation；
- 持久化 LangGraph checkpointer 历史，支持服务重启后的同一 Thread 多轮继续；
- 将现有 `parents.jsonl`、`chunks.jsonl`、`bm25_tokens.jsonl`、`dense.npy` 导入 Postgres；
- 使用 pgvector 执行 dense 检索；
- 使用 Elasticsearch 执行关键词 BM25 检索；
- 保留现有 `retrieve_tcm_docs()` 和 `retrieve_tcm_knowledge` 公共入口；
- 提供从 Postgres 重建 Elasticsearch 索引的命令；
- 提供文件引擎回退开关，便于迁移期排查。

### 3.2 本阶段不提供

- 不扩展新书、新症状或外部知识库；
- 不重新运行 RAG 正式实验或生成论文结果；
- 不改变回答安全边界，不生成方剂、药物、剂量和煎服法建议；
- 不把 Elasticsearch 作为主数据库；
- 不把 `conversation` 拼回模型输入；
- 不在生产请求中重新解析原始古籍 TXT。

## 4. 状态所有权

当前系统有三个状态层，数据库化后必须继续分清：

| 状态 | 归属 | 用途 |
| --- | --- | --- |
| LangGraph messages / checkpoint | Checkpointer | 模型真实上下文和工具调用链 |
| Thread status / Run status | Runtime store | API 生命周期和 SSE 状态 |
| conversation | Thread history view | 前端展示用户可见历史 |

关键约束：

```text
每次 Run 仍只向 Agent 传入本轮用户消息；
历史上下文由同一 thread_id 对应的持久化 Checkpointer 提供；
conversation 只用于 GET /history 和前端显示，不得重新拼回 Agent input。
```

这保持 V1.3 以来的澄清恢复设计，也避免数据库化后重新引入重复历史。

## 5. Postgres 数据模型

### 5.1 运行态表

`app_threads`

```text
thread_id uuid primary key
status text not null
created_at timestamptz not null
updated_at timestamptz not null
metadata jsonb not null default '{}'
```

`app_runs`

```text
run_id uuid primary key
thread_id uuid not null references app_threads(thread_id)
assistant_id text not null
status text not null
error text null
input jsonb not null default '{}'
context jsonb not null default '{}'
created_at timestamptz not null
updated_at timestamptz not null
```

`app_messages`

```text
id bigserial primary key
thread_id uuid not null references app_threads(thread_id)
run_id uuid null references app_runs(run_id)
message_id text null
ordinal integer not null
type text not null
role text null
name text null
tool_call_id text null
content jsonb not null
tool_calls jsonb null
visible boolean not null default false
created_at timestamptz not null
```

`visible=true` 表示该消息可进入前端 conversation 视图。Tool message、带 `tool_calls` 的中间 AI message 默认不可见，但仍保存到 `app_messages` 供调试和追踪。

`app_agent_trace_events`

```text
id bigserial primary key
run_id uuid not null references app_runs(run_id)
thread_id uuid not null references app_threads(thread_id)
event_type text not null
agent text null
summary text null
payload jsonb not null default '{}'
created_at timestamptz not null
```

`app_validation_results`

```text
id bigserial primary key
run_id uuid not null references app_runs(run_id)
thread_id uuid not null references app_threads(thread_id)
validation jsonb not null
allowed_terms text[] not null default '{}'
rewritten boolean not null default false
created_at timestamptz not null
```

LangGraph checkpoint 优先使用官方 Postgres checkpointer 适配层；如果当前依赖暂不可用，则用独立 `langgraph_checkpoints` 表封装相同语义。无论采用哪种实现，都不把 checkpoint 历史退化为只存 `conversation`。

### 5.2 RAG 事实表

`rag_corpora`

```text
corpus_id text primary key
version text not null
status text not null
source_manifest_sha256 text not null
index_manifest_sha256 text not null
embedding_model jsonb not null
reranker_model jsonb null
vector_dimension integer not null
parent_count integer not null
chunk_count integer not null
created_at timestamptz not null
```

`rag_sources`

```text
source_id text primary key
corpus_id text not null references rag_corpora(corpus_id)
source_type text not null
book_id text not null
book_title text not null
source_file text not null
source_hash text not null
encoding text not null
metadata jsonb not null default '{}'
```

`rag_sections`

```text
section_id text primary key
corpus_id text not null references rag_corpora(corpus_id)
source_id text not null references rag_sources(source_id)
volume text not null
chapter text not null
section text not null
symptom_tags text[] not null default '{}'
metadata jsonb not null default '{}'
```

`rag_parents`

```text
parent_id text primary key
corpus_id text not null references rag_corpora(corpus_id)
source_id text not null references rag_sources(source_id)
section_id text null references rag_sections(section_id)
source_type text not null
book_id text not null
book_title text not null
source_file text not null
source_hash text not null
volume text not null
chapter text not null
section text not null
symptom_tags text[] not null default '{}'
evidence_role text not null
original_text text not null
normalized_text text not null
created_at timestamptz not null
```

`rag_chunks`

```text
chunk_id text primary key
parent_id text not null references rag_parents(parent_id)
corpus_id text not null references rag_corpora(corpus_id)
row_index integer not null
text text not null
source_type text not null
symptom_tags text[] not null default '{}'
evidence_role text not null
created_at timestamptz not null
```

`rag_chunk_embeddings`

```text
chunk_id text primary key references rag_chunks(chunk_id)
corpus_id text not null references rag_corpora(corpus_id)
embedding vector(1024) not null
embedding_model text not null
embedding_revision text not null
created_at timestamptz not null
```

`rag_bm25_tokens`

```text
chunk_id text primary key references rag_chunks(chunk_id)
corpus_id text not null references rag_corpora(corpus_id)
tokens text[] not null
created_at timestamptz not null
```

`rag_retrieval_logs`

```text
id bigserial primary key
run_id uuid null references app_runs(run_id)
thread_id uuid null references app_threads(thread_id)
corpus_id text not null references rag_corpora(corpus_id)
original_query text not null
rewritten_query text not null
chief_symptom text null
retrieval_mode text not null
degraded boolean not null default false
degraded_reason text null
dense_hits jsonb not null default '[]'
keyword_hits jsonb not null default '[]'
fused_hits jsonb not null default '[]'
final_results jsonb not null default '[]'
created_at timestamptz not null
```

## 6. Postgres 索引策略

所有外键列必须建索引，避免 join、cascade 和查询退化为全表扫描。

运行态索引：

```sql
create index app_runs_thread_created_idx
  on app_runs (thread_id, created_at desc);

create index app_runs_status_updated_idx
  on app_runs (status, updated_at desc);

create index app_messages_thread_ordinal_idx
  on app_messages (thread_id, ordinal);

create index app_messages_run_idx
  on app_messages (run_id);

create index app_messages_visible_idx
  on app_messages (thread_id, ordinal)
  where visible = true;
```

RAG 索引：

```sql
create index rag_chunks_parent_idx
  on rag_chunks (parent_id);

create index rag_chunks_corpus_role_idx
  on rag_chunks (corpus_id, evidence_role);

create index rag_chunks_symptom_tags_idx
  on rag_chunks using gin (symptom_tags);

create index rag_parents_symptom_tags_idx
  on rag_parents using gin (symptom_tags);
```

pgvector 索引：

```sql
create index rag_chunk_embeddings_hnsw_idx
  on rag_chunk_embeddings
  using hnsw (embedding vector_cosine_ops);
```

当前只有 2288 条 chunk，精确向量扫描也足够快；但 schema 设计按可扩展路径保留 HNSW 索引。导入后必须执行 `analyze`，并用 `explain analyze` 验证过滤和 join 路径。

## 7. Elasticsearch 索引设计

ES 使用版本化索引和 alias：

```text
tcm_rag_chunks_v1_0_0
tcm_rag_chunks_current -> tcm_rag_chunks_v1_0_0
```

文档 `_id` 固定为 `chunk_id`，便于幂等重建。

文档字段：

```json
{
  "chunk_id": "...",
  "parent_id": "...",
  "corpus_id": "ancient-books-v1.0.0",
  "book_id": "jing_yue_quan_shu",
  "book_title": "景岳全书",
  "source_file": "637-景岳全书.txt",
  "source_hash": "...",
  "volume": "卷之十七理集·杂证谟",
  "chapter": "眩运",
  "section": "论证（共四条）",
  "text": "child chunk text",
  "symptom_tags": ["眩晕"],
  "evidence_role": "syndrome_pattern",
  "row_index": 123,
  "index_version": "v1.0.0"
}
```

字段权重建议：

```text
text: 主要 BM25 字段
chapter / section: 中等权重，用于症状章节命中
symptom_tags: filter 字段，不作为自由文本主召回
book_title / volume: 展示和调试字段
```

中文分词采用可配置 analyzer。首版允许两种路径：

```text
1. 有 IK 或等价中文分词插件时，使用中文 analyzer；
2. 无插件时，使用默认 analyzer + symptom_tags/filter 保底，但 doctor 检查应提示关键词召回质量可能下降。
```

ES 不保存 parent 原文作为事实来源。最终展示的 `original_text` 必须从 Postgres `rag_parents` 读取。

## 8. 导入与同步

导入命令从现有生产产物读取：

```text
data/rag/ancient_books/corpus/manifest.json
data/rag/ancient_books/corpus/parents.jsonl
data/rag/ancient_books/corpus/chunks.jsonl
data/rag/ancient_books/index/manifest.json
data/rag/ancient_books/index/rows.jsonl
data/rag/ancient_books/index/bm25_tokens.jsonl
data/rag/ancient_books/index/dense.npy
```

导入步骤：

```text
1. 读取 corpus manifest 和 index manifest；
2. 校验 status=ready、row_count、vector_dimension、文件 SHA256；
3. 校验 rows、bm25_tokens、dense 行顺序一致；
4. 事务内 upsert rag_corpora / sources / sections / parents / chunks；
5. 批量写入 embeddings 和 bm25_tokens；
6. 提交后执行 Postgres analyze；
7. 从 Postgres 重建 ES versioned index；
8. ES 文档数和 chunk_count 一致后切换 alias；
9. 写入导入报告和 doctor 结果。
```

一致性规则：

```text
Postgres 导入失败：事务回滚，不切 ES alias；
ES 重建失败：Postgres 保持成功状态，但 database engine 标记 keyword branch unavailable；
ES 文档数不等于 chunk_count：不得切换 alias；
源文件或 manifest hash 不一致：拒绝导入。
```

## 9. 运行时检索设计

新增 database retrieval engine，但保留原 file engine：

```text
RAG_ENGINE=file | database
RAG_FALLBACK_FILE_ENGINE=true | false
```

database engine 流程：

```text
用户输入
  -> detect_chief_symptom
  -> rewrite_query
  -> pgvector dense search
  -> Elasticsearch BM25 search
  -> RRF fusion
  -> reranker
  -> load parent evidence from Postgres
  -> assign E1-E5 citation_id
  -> format_retrieval_results
```

Dense 查询：

```text
1. 使用同一 BGE-M3 encoder 生成 query embedding；
2. 校验维度为 1024；
3. 按 corpus_id、chief_symptom、evidence_role 过滤；
4. 使用 cosine distance 排序；
5. 返回 chunk_id、parent_id、dense_rank、dense_score。
```

Keyword 查询：

```text
1. 使用 rewritten_query 查询 ES；
2. 按 corpus_id、chief_symptom、evidence_role 过滤；
3. 返回 chunk_id、parent_id、bm25_rank、bm25_score；
4. ES 不可用时返回 degraded=true，不静默伪装为 hybrid。
```

Fusion 与 parent recovery：

```text
1. dense 和 keyword 的 chunk_id 进入 RRF；
2. 候选上限沿用当前 reranker_candidate_k；
3. reranker 对 child text 排序；
4. 根据 parent_id 从 Postgres 读取完整 parent；
5. 同一 parent 多个 child 命中时只保留最高排名；
6. 最终最多返回 E1-E5。
```

## 10. 应用层改造边界

运行态存储：

```text
app/store/thread_store.py
  -> 抽象接口保持 create/get/list/update_status/update_values
  -> 新增 Postgres 实现

app/store/run_manager.py
  -> 抽象接口保持 create/get/set_status
  -> 新增 Postgres 实现

app/runtime/state.py
  -> 根据配置选择 in-memory 或 Postgres store
```

Agent checkpointer：

```text
app/agents/lead_agent/agent.py
  -> InMemorySaver 替换为可配置 checkpointer factory
  -> 默认生产配置使用 Postgres checkpointer
  -> 测试可继续使用 InMemorySaver
```

RAG：

```text
app/rag/vector_store.py
  -> 继续保留 file engine
  -> 新增 database engine factory

app/rag/ancient_books/runtime.py
  -> 抽出 shared fusion / parent recovery 逻辑
  -> database engine 复用当前返回结构

app/rag/retriever.py
  -> 公共入口不变
  -> 根据 RAG_ENGINE 路由到 file 或 database
```

路由接口不因本阶段改名。现有 `/api/threads`、`/api/threads/{thread_id}/history` 和 `/api/threads/{thread_id}/runs/stream` 保持兼容。

## 11. 配置

新增环境变量：

```env
DATABASE_URL=postgresql+asyncpg://...
POSTGRES_POOL_SIZE=10
CHECKPOINT_BACKEND=postgres

RAG_ENGINE=database
RAG_FALLBACK_FILE_ENGINE=true

ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_RAG_INDEX_ALIAS=tcm_rag_chunks_current
ELASTICSEARCH_ANALYZER=standard
```

连接管理要求：

```text
1. FastAPI 使用进程级连接池；
2. 不允许每个请求新建数据库连接；
3. 长事务只包裹导入和必要状态更新；
4. SSE 流式返回期间不得持有长时间数据库事务；
5. 批量导入使用分批提交或 copy，避免单条 insert 循环拖慢。
```

## 12. 异常与降级

| 场景 | 行为 |
| --- | --- |
| Postgres 不可用 | 服务启动失败，不能进入生产模式 |
| Checkpointer 写入失败 | 当前 Run 返回 error，不能只保存 conversation |
| pgvector 查询失败 | 检索返回 degraded=true，若允许 fallback 则切 file engine |
| Elasticsearch 不可用 | keyword branch degraded，dense branch 可继续；最终 retrieval_mode 标记实际模式 |
| ES 文档数不匹配 | doctor 失败，不切 alias |
| manifest hash 不一致 | 拒绝导入和加载 |
| parent recovery 找不到 parent | 当前候选丢弃并记录错误；若最终为空返回 insufficient_evidence |
| vector_dimension 不等于 1024 | 拒绝导入 |

任何降级都必须出现在 retrieval result 和 `rag_retrieval_logs` 中，不能静默吞掉。

## 13. 验证计划

### 13.1 Schema 与导入验证

- Postgres migration 能从空库执行成功；
- `vector` extension 可用；
- 导入后 `rag_corpora.chunk_count = 2288`；
- `rag_chunks`、`rag_chunk_embeddings`、`rag_bm25_tokens` 数量一致；
- `dense.npy` 的第 N 行与 `rag_chunks.row_index = N` 对齐；
- 所有 `rag_chunks.parent_id` 都能在 `rag_parents` 找到；
- ES index 文档数等于 `rag_chunks` 数量；
- ES alias 指向最新成功构建的 versioned index。

### 13.2 Runtime 验证

- 创建 thread 后重启服务，thread 仍可查询；
- 首轮触发 clarification 后重启服务，用户补充信息仍能在同一 thread 继续；
- `conversation` 只包含用户可见消息；
- `app_messages` 保留 tool message、tool_call_id 和中间 AI tool_calls；
- Guardrail 重写后，最终 AIMessage 与 checkpoint 中的消息保持一致；
- `/api/threads/{thread_id}/history` 响应字段保持兼容。

### 13.3 RAG 验证

沿用当前十条 smoke 查询：

```text
头痛、眩晕、咳嗽、喘促、心悸、不寐、胃脘痛、腹痛、泄泻、便秘
```

预期：

```text
1. database engine 和 file engine 对支持症状都返回 ok 或明确 degraded；
2. 胃脘痛仍允许返回 insufficient_evidence，不强行用弱相关章节补齐；
3. 每条 ok 结果都包含 E1-E5、parent_id、chunk_id、book_title、volume、chapter、section；
4. ES 关闭时 retrieval_mode 不能标记为完整 hybrid；
5. pgvector 关闭或 Postgres 不可用时生产模式不能假装成功。
```

本验证是工程回归，不产生新的正式实验结论。

## 14. 迁移与回滚

迁移顺序：

```text
1. 新增依赖、配置和 docker compose 服务；
2. 新增 Postgres schema migration；
3. 新增导入命令和 doctor 命令；
4. 导入当前 V1.7 生产 RAG 产物；
5. 建立 ES versioned index 和 alias；
6. 新增 Postgres thread/run store；
7. 新增 Postgres checkpointer；
8. 新增 database retrieval engine；
9. 开启 RAG_ENGINE=database 运行 smoke；
10. 保留 file engine 回退直到 database engine 稳定。
```

回滚策略：

```text
1. RAG_ENGINE=file 可回到当前文件索引检索；
2. CHECKPOINT_BACKEND=memory 可回到当前开发态；
3. ES alias 切换失败不影响 Postgres facts；
4. 导入脚本使用 corpus_id/version 幂等写入，不覆盖未确认版本；
5. 删除失败 ES index 不影响 Postgres 主库。
```

## 15. 完成标准

满足以下条件视为本阶段完成：

- Postgres、pgvector、Elasticsearch 可以通过本地配置启动；
- 对话、run、message 和 checkpoint 重启后仍可恢复；
- 当前 V1.7 RAG 产物完整导入 Postgres；
- pgvector dense branch 能返回候选 chunk；
- Elasticsearch keyword branch 能返回候选 chunk；
- database engine 能执行 dense + keyword + RRF + reranker + parent recovery；
- 公共 RAG 工具入口不破坏现有调用方；
- file engine 回退可用；
- doctor 和 smoke 验证通过；
- 没有新增外部语料、正式实验和论文结果文件。

## 16. 一句话总结

方案 C 的核心不是简单替换存储，而是明确：

```text
Postgres 管事实和状态，
pgvector 管语义召回，
Elasticsearch 管关键词召回，
Checkpointer 管模型上下文，
conversation 只管前端可见历史。
```

只要这五条边界不混，数据库化后系统会更稳，而不是更复杂。
