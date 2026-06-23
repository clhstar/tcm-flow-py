# TCM-Flow 运行手册

本文档整理当前项目的本地开发、服务启动、RAG artifacts 生成、Postgres/pgvector 导入、Elasticsearch 建索引、API 冒烟检查和测试命令。

所有命令默认在项目根目录执行：

```powershell
cd G:\work\tcm-flow
```

## 1. Python 环境

激活项目虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装或刷新依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

执行项目命令时，建议显式使用 `.venv` 里的 Python：

```powershell
.\.venv\Scripts\python.exe -m <module>
```

## 2. 必需的 `.env`

应用会通过 `app.config.get_settings()` 加载 `.env`。

本地数据库/RAG 的最低配置如下：

```env
DATABASE_URL=postgresql://tcm:tcm@localhost:15432/tcm_flow
POSTGRES_POOL_SIZE=10
CHECKPOINT_BACKEND=postgres

ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_RAG_INDEX_ALIAS=tcm_rag_chunks_current
ELASTICSEARCH_ANALYZER=standard
```

注意：`docker-compose.persistence.yml` 把宿主机端口 `15432` 映射到容器内端口 `5432`，所以本地工具和应用都应该连接 `localhost:15432`。

同一个 Postgres 容器还会在首次创建 volume 时初始化 Java 后端本地业务库：

```text
数据库: tcm_consultation
用户: tcm_app
密码: tcm_app_dev_password
连接: jdbc:postgresql://localhost:15432/tcm_consultation
```

这个初始化由 `docker/postgres/init/01-create-shared-databases.sh` 完成，只在新的 Postgres volume 上自动执行。如果本机已经存在旧的 `tcm_flow_postgres` volume，需要手动创建 `tcm_consultation`，或在确认不需要保留旧数据后重建 volume。

不要提交 `.env`，里面可能包含私钥或真实 API Key。

## 3. 启动持久化服务

启动 Postgres/pgvector 和 Elasticsearch：

```powershell
docker compose -f docker-compose.persistence.yml up -d
```

查看容器状态：

```powershell
docker compose -f docker-compose.persistence.yml ps
```

期望看到的服务：

```text
tcm-flow-postgres-1        localhost:15432 -> 5432
tcm-flow-elasticsearch-1   localhost:9200  -> 9200
```

查看日志：

```powershell
docker compose -f docker-compose.persistence.yml logs --tail=80 postgres
docker compose -f docker-compose.persistence.yml logs --tail=80 elasticsearch
```

停止服务，但不删除 volume：

```powershell
docker compose -f docker-compose.persistence.yml stop
```

停止服务并删除容器/网络；默认保留 named volumes，除非额外加 `-v`：

```powershell
docker compose -f docker-compose.persistence.yml down
```

## 4. 启动后端

在 `2027` 端口启动 FastAPI：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 2027
```

健康检查：

```powershell
curl.exe http://localhost:2027/health
```

应用启动生命周期会做这些事情：

- 读取 `.env`；
- 创建 Postgres 连接池，线上 RAG 固定使用 Postgres/pgvector + Elasticsearch；
- 执行 `app/db/schema.sql`；
- 当 `CHECKPOINT_BACKEND=postgres` 时启用 Postgres 版 thread/run 存储；
- 启用数据库版 RAG 引擎。

## 5. 连接 Postgres/pgvector

通过 Docker 进入 psql：

```powershell
docker compose -f docker-compose.persistence.yml exec -T postgres psql -U tcm -d tcm_flow
```

本地连接串：

```text
postgresql://tcm:tcm@localhost:15432/tcm_flow
```

Navicat 连接配置：

```text
类型: PostgreSQL
主机: localhost
端口: 15432
数据库: tcm_flow
用户: tcm
密码: tcm
```

检查 pgvector 扩展：

```powershell
docker compose -f docker-compose.persistence.yml exec -T postgres psql -U tcm -d tcm_flow -c "select extname, extversion from pg_extension where extname = 'vector';"
```

检查 RAG 表数据量：

```powershell
docker compose -f docker-compose.persistence.yml exec -T postgres psql -U tcm -d tcm_flow -c "select 'rag_corpora' as table_name, count(*) from rag_corpora union all select 'rag_sources', count(*) from rag_sources union all select 'rag_parents', count(*) from rag_parents union all select 'rag_chunks', count(*) from rag_chunks union all select 'rag_chunk_embeddings', count(*) from rag_chunk_embeddings union all select 'rag_bm25_tokens', count(*) from rag_bm25_tokens union all select 'rag_retrieval_logs', count(*) from rag_retrieval_logs;"
```

## 6. 连接 Elasticsearch

Elasticsearch 地址：

```text
http://localhost:9200
```

查看健康状态：

```powershell
curl.exe http://localhost:9200/_cluster/health?pretty
```

查看索引列表：

```powershell
curl.exe http://localhost:9200/_cat/indices?v
```

查看 alias：

```powershell
curl.exe http://localhost:9200/_cat/aliases?v
```

统计当前 RAG alias 下的文档数：

```powershell
curl.exe http://localhost:9200/tcm_rag_chunks_current/_count?pretty
```

本地单节点 ES 索引可能显示 `yellow`，通常是因为副本分片没有分配。只要主分片正常、查询可用，本地开发可以接受。

## 7. 当前 ancient-books artifacts

当前生产用 artifacts 在这里：

```text
data/rag/ancient_books/corpus
data/rag/ancient_books/index
data/rag/ancient_books/models
```

关键文件：

```text
data/rag/ancient_books/corpus/parents.jsonl
data/rag/ancient_books/corpus/chunks.jsonl
data/rag/ancient_books/corpus/sections.jsonl
data/rag/ancient_books/corpus/manifest.json
data/rag/ancient_books/index/rows.jsonl
data/rag/ancient_books/index/bm25_tokens.jsonl
data/rag/ancient_books/index/dense.npy
data/rag/ancient_books/index/manifest.json
```

## 8. 导入 RAG artifacts 到 Postgres/pgvector

这个命令会把 `corpus` 和 `index` artifacts 导入到 Postgres 表中，包括 `rag_corpora`、`rag_parents`、`rag_chunks`、`rag_chunk_embeddings`、`rag_bm25_tokens` 等。

```powershell
.\.venv\Scripts\python.exe -m app.rag.database.cli import-artifacts --corpus-dir data\rag\ancient_books\corpus --index-dir data\rag\ancient_books\index
```

当前 artifacts 的期望结果：

```text
{'corpus_id': 'ancient-books-v1.0.0', 'chunk_count': 2288}
```

验证 Postgres 导入结果：

```powershell
docker compose -f docker-compose.persistence.yml exec -T postgres psql -U tcm -d tcm_flow -c "select count(*) from rag_chunks;"
docker compose -f docker-compose.persistence.yml exec -T postgres psql -U tcm -d tcm_flow -c "select count(*) from rag_chunk_embeddings;"
```

当前期望数量：

```text
2288
```

## 9. 重建 Elasticsearch RAG 索引

这个命令会创建关键词检索索引，并把 alias 指向新索引。

```powershell
.\.venv\Scripts\python.exe -m app.rag.database.cli rebuild-elasticsearch --corpus-dir data\rag\ancient_books\corpus --index-dir data\rag\ancient_books\index
```

当前 artifacts 的期望结果：

```text
{'index': 'tcm_rag_chunks_v1_0_0', 'alias': 'tcm_rag_chunks_current', 'document_count': 2288}
```

验证 ES 导入结果：

```powershell
curl.exe http://localhost:9200/_cat/indices?v
curl.exe http://localhost:9200/_cat/aliases?v
curl.exe http://localhost:9200/tcm_rag_chunks_current/_count?pretty
```

期望 alias：

```text
tcm_rag_chunks_current -> tcm_rag_chunks_v1_0_0
```

期望数量：

```text
2288
```

## 10. 完整本地启动流程

从一个新终端打开项目时，可以按下面顺序执行：

```powershell
cd G:\work\tcm-flow
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
docker compose -f docker-compose.persistence.yml up -d
.\.venv\Scripts\python.exe -m app.rag.database.cli import-artifacts --corpus-dir data\rag\ancient_books\corpus --index-dir data\rag\ancient_books\index
.\.venv\Scripts\python.exe -m app.rag.database.cli rebuild-elasticsearch --corpus-dir data\rag\ancient_books\corpus --index-dir data\rag\ancient_books\index
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 2027
```

如果 Postgres 和 ES volumes 里已经有最新数据，可以跳过两个导入/建索引命令。

## 11. 从源文件重建 ancient-books 语料和文件索引

这些命令会重建 `data/rag/ancient_books` 下的本地文件 artifacts。它们比直接导入现有 artifacts 慢，并且可能依赖本地模型快照或 GPU，具体取决于配置。

从原始古籍文件构建语料：

```powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli build-corpus --source-root G:\work\TCM-Ancient-Books-master
```

检查语料：

```powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli doctor
```

准备 BGE 模型快照：

```powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli prepare-models
```

构建本地 BM25 和 dense 文件索引：

```powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli build-index
```

等价的短入口：

```powershell
.\.venv\Scripts\python.exe -m app.rag.build_index
```

导出公开 manifest 快照到 `app/rag/ancient_books/manifests`：

```powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli export-manifests
```

离线冒烟测试本地 artifacts：

```powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli smoke
```

典型的“源文件 -> 数据库”重建流程：

```powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli build-corpus --source-root G:\work\TCM-Ancient-Books-master
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli doctor
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli prepare-models
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli build-index
.\.venv\Scripts\python.exe -m app.rag.database.cli import-artifacts --corpus-dir data\rag\ancient_books\corpus --index-dir data\rag\ancient_books\index
.\.venv\Scripts\python.exe -m app.rag.database.cli rebuild-elasticsearch --corpus-dir data\rag\ancient_books\corpus --index-dir data\rag\ancient_books\index
```

## 12. Database RAG CLI 参考

查看帮助：

```powershell
.\.venv\Scripts\python.exe -m app.rag.database.cli --help
```

命令说明：

```text
import-artifacts         导入本地 RAG artifacts 到 Postgres/pgvector。
rebuild-elasticsearch    重建 ES 关键词索引并切换 alias。
doctor                   占位命令；当前输出 not_connected。
smoke                    占位命令；当前输出 not_connected。
```

## 13. Ancient-Books CLI 参考

查看帮助：

```powershell
.\.venv\Scripts\python.exe -m app.rag.ancient_books.cli --help
```

命令说明：

```text
build-corpus       将选定古籍源文本解析为 sections/parents/chunks。
doctor             校验语料 artifacts。
prepare-models     下载或准备配置中的 embedding/reranker 模型快照。
build-index        构建离线 rows/bm25_tokens/dense artifacts，用于导入数据库。
export-manifests   导出适合提交的 manifest 文件，用于记录来源和版本。
smoke              对本地 artifacts 做离线冒烟检查。
```

## 14. API 冒烟调用

先启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 2027
```

创建 thread：

```powershell
curl.exe -s -X POST http://localhost:2027/api/threads
```

调用流式 run 接口。先替换 `<thread_id>`：

```powershell
curl.exe -N -X POST http://localhost:2027/api/threads/<thread_id>/runs/stream -H "Content-Type: application/json" -d "{\"assistant_id\":\"lead_agent\",\"input\":{\"messages\":[{\"type\":\"human\",\"content\":[{\"type\":\"text\",\"text\":\"请分析头痛伴恶风\"}]}]},\"stream_mode\":[\"messages\"],\"context\":{\"thinking_enabled\":true,\"is_plan_mode\":true,\"subagent_enabled\":false}}"
```

在 Git Bash 或 WSL 中，可以用 helper 脚本自动创建 thread，并打印最终 AI 消息：

```bash
MINI_DEERFLOW_URL=http://localhost:2027 ./scripts/chat.sh "请分析头痛伴恶风"
```

第二个参数可以传已有 thread id：

```bash
MINI_DEERFLOW_URL=http://localhost:2027 ./scripts/chat.sh "继续分析" "<thread_id>"
```

## 15. 测试

运行持久化/RAG 相关的聚焦测试：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config tests.rag.database.test_artifacts tests.rag.database.test_repository tests.rag.database.test_elasticsearch_index tests.rag.database.test_cli tests.rag.database.test_engine tests.rag.database.test_engine_routing tests.store.test_postgres_store tests.test_app_lifespan tests.test_runtime_state -v
```

运行全部自动发现的测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

提交前检查暂存区空白字符问题：

```powershell
git diff --cached --check
```

## 16. 实验 CLI 入口

这些是研究/评测脚本，不是普通后端启动所必需的。

V1.5 实验 CLI：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_5.cli --help
```

V1.5 常见命令族包括语料准备、解析、切块、索引、检索冒烟、pilot run、formal run、answer-level run、review 导入/导出等。运行前优先查看已提交的实验计划和日志，因为这些命令会写入 `data/rag_v1_5`。

V1.6 public TCM-QG CLI：

```powershell
.\.venv\Scripts\python.exe -m experiments.rag_v1_6.cli --help
```

V1.6 常见命令族包括公开数据集冻结、切块/建索引、检索矩阵运行、formal retrieval/answer 运行、review 准备/导入、summary freeze 等。运行前优先查看已提交的 V1.6 计划和 manifest，因为这些命令会写入 `data/rag_v1_6`。

## 17. 常见问题排查

如果提示缺少 `asyncpg` 或 `elasticsearch`，通常是用了系统 Python。请使用：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 2027
```

如果 Postgres 的 RAG 表是空的，执行：

```powershell
.\.venv\Scripts\python.exe -m app.rag.database.cli import-artifacts --corpus-dir data\rag\ancient_books\corpus --index-dir data\rag\ancient_books\index
```

如果 Elasticsearch 里没有 RAG 索引，执行：

```powershell
.\.venv\Scripts\python.exe -m app.rag.database.cli rebuild-elasticsearch --corpus-dir data\rag\ancient_books\corpus --index-dir data\rag\ancient_books\index
```

如果 Navicat 连不上 Postgres，先确认端口是 `15432`，不是 `5432`。

如果 ES client 报 `compatible-with=9` 相关兼容错误，在 `requirements.txt` 固定版本后重新安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
