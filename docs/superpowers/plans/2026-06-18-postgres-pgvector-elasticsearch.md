# Postgres pgvector Elasticsearch Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved scheme C persistence layer: Postgres as the source of truth, pgvector for dense retrieval, Elasticsearch for keyword retrieval, and persistent conversation/checkpoint storage.

**Architecture:** Keep public FastAPI and RAG tool APIs stable while adding configurable persistence backends. Implement the database path beside the existing file/in-memory path, use strict import validation from V1.7 artifacts, and keep file-engine fallback until database retrieval passes smoke checks.

**Tech Stack:** FastAPI, LangGraph, asyncpg, pgvector, Elasticsearch Python client, Postgres SQL migrations, unittest, numpy, existing ancient-books RAG runtime.

---

## Scope Check

The approved design touches runtime persistence, RAG artifact import, dense search, keyword search, and engine routing. These are coupled by shared schema and configuration, so this plan keeps them in one ordered implementation path rather than splitting specs. Each task still lands a testable slice with its own commit.

## File Structure

Create these files:

- `app/config.py`: environment-backed settings for database, checkpoint, RAG engine, and Elasticsearch.
- `app/db/__init__.py`: package marker.
- `app/db/pool.py`: asyncpg pool creation and lifecycle helpers.
- `app/db/schema.sql`: Postgres schema, pgvector extension, tables, constraints, and indexes.
- `app/db/migrations.py`: idempotent schema runner used by tests and local setup.
- `app/store/postgres_thread_store.py`: Postgres implementation of the existing ThreadStore API.
- `app/store/postgres_run_manager.py`: Postgres implementation of the existing RunManager API.
- `app/checkpoints/__init__.py`: package marker.
- `app/checkpoints/factory.py`: configurable LangGraph checkpointer factory.
- `app/rag/database/__init__.py`: package marker.
- `app/rag/database/artifacts.py`: validated loader for current V1.7 corpus/index artifacts.
- `app/rag/database/repository.py`: Postgres RAG repository for import, parent recovery, and vector search.
- `app/rag/database/elasticsearch_index.py`: versioned Elasticsearch chunk index builder and search client.
- `app/rag/database/engine.py`: database-backed retrieval engine using pgvector + ES + existing fusion semantics.
- `app/rag/database/cli.py`: import, rebuild ES, doctor, and smoke command entry point.
- `tests/test_config.py`: settings tests.
- `tests/db/test_schema.py`: schema text and migration tests.
- `tests/store/test_postgres_store.py`: Postgres store tests with fake async connection.
- `tests/checkpoints/test_factory.py`: checkpointer backend selection tests.
- `tests/rag/database/test_artifacts.py`: artifact validation tests.
- `tests/rag/database/test_repository.py`: repository SQL and vector query behavior tests.
- `tests/rag/database/test_elasticsearch_index.py`: ES document and alias behavior tests.
- `tests/rag/database/test_engine.py`: database retrieval fusion and degradation tests.
- `tests/rag/database/test_cli.py`: import/doctor/smoke CLI tests.

Modify these files:

- `requirements.txt`: add asyncpg, pgvector, elasticsearch, and langgraph-checkpoint-postgres.
- `app/runtime/state.py`: select in-memory or Postgres stores from settings.
- `app/agents/lead_agent/agent.py`: use checkpointer factory instead of hard-coded `InMemorySaver`.
- `app/rag/vector_store.py`: route production engine creation through file/database setting while keeping file engine.
- `app/rag/retriever.py`: keep public API stable and allow database engine result payload.
- `app/tools/builtins/retrieval_tool.py`: include run/thread metadata in database retrieval logs when available.
- `app/gateway/routers/rag.py`: read retrieval logs from the configured backend.

## Task 1: Settings and Dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing settings tests**

Create `tests/test_config.py`:

```python
import os
import unittest
from unittest.mock import patch

from app.config import AppSettings, get_settings, reset_settings_cache


class SettingsTests(unittest.TestCase):
    def tearDown(self):
        reset_settings_cache()

    def test_defaults_keep_current_file_and_memory_behavior(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(settings.checkpoint_backend, "memory")
        self.assertEqual(settings.rag_engine, "file")
        self.assertTrue(settings.rag_fallback_file_engine)
        self.assertEqual(settings.elasticsearch_rag_index_alias, "tcm_rag_chunks_current")

    def test_database_configuration_is_loaded_from_environment(self):
        env = {
            "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost:5432/tcm",
            "POSTGRES_POOL_SIZE": "7",
            "CHECKPOINT_BACKEND": "postgres",
            "RAG_ENGINE": "database",
            "RAG_FALLBACK_FILE_ENGINE": "false",
            "ELASTICSEARCH_URL": "http://localhost:9200",
            "ELASTICSEARCH_RAG_INDEX_ALIAS": "tcm_rag_chunks_test",
            "ELASTICSEARCH_ANALYZER": "ik_max_word",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = AppSettings.from_env()

        self.assertEqual(settings.database_url, env["DATABASE_URL"])
        self.assertEqual(settings.postgres_pool_size, 7)
        self.assertEqual(settings.checkpoint_backend, "postgres")
        self.assertEqual(settings.rag_engine, "database")
        self.assertFalse(settings.rag_fallback_file_engine)
        self.assertEqual(settings.elasticsearch_url, "http://localhost:9200")
        self.assertEqual(settings.elasticsearch_analyzer, "ik_max_word")

    def test_cached_settings_can_be_reset_for_tests(self):
        with patch.dict(os.environ, {"RAG_ENGINE": "database"}, clear=True):
            first = get_settings()
        with patch.dict(os.environ, {"RAG_ENGINE": "file"}, clear=True):
            cached = get_settings()
            reset_settings_cache()
            refreshed = get_settings()

        self.assertIs(first, cached)
        self.assertEqual(refreshed.rag_engine, "file")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config -v
```

Expected: import failure for `app.config`.

- [ ] **Step 3: Add dependencies**

Update `requirements.txt` by appending these lines:

```text
asyncpg
pgvector
elasticsearch
langgraph-checkpoint-postgres
```

- [ ] **Step 4: Implement settings**

Create `app/config.py`:

```python
import os
from dataclasses import dataclass
from functools import lru_cache


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class AppSettings:
    database_url: str | None
    postgres_pool_size: int
    checkpoint_backend: str
    rag_engine: str
    rag_fallback_file_engine: bool
    elasticsearch_url: str | None
    elasticsearch_rag_index_alias: str
    elasticsearch_analyzer: str

    @classmethod
    def from_env(cls) -> "AppSettings":
        checkpoint_backend = os.getenv("CHECKPOINT_BACKEND", "memory").strip().lower()
        rag_engine = os.getenv("RAG_ENGINE", "file").strip().lower()
        if checkpoint_backend not in {"memory", "postgres"}:
            raise ValueError("CHECKPOINT_BACKEND must be memory or postgres")
        if rag_engine not in {"file", "database"}:
            raise ValueError("RAG_ENGINE must be file or database")
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            postgres_pool_size=_int_env("POSTGRES_POOL_SIZE", 10),
            checkpoint_backend=checkpoint_backend,
            rag_engine=rag_engine,
            rag_fallback_file_engine=_bool_env("RAG_FALLBACK_FILE_ENGINE", True),
            elasticsearch_url=os.getenv("ELASTICSEARCH_URL"),
            elasticsearch_rag_index_alias=os.getenv(
                "ELASTICSEARCH_RAG_INDEX_ALIAS",
                "tcm_rag_chunks_current",
            ),
            elasticsearch_analyzer=os.getenv("ELASTICSEARCH_ANALYZER", "standard"),
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings.from_env()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
```

- [ ] **Step 5: Run the test and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add requirements.txt app/config.py tests/test_config.py
git commit -m "feat: add persistence settings"
```

## Task 2: Postgres Schema and Migration Runner

**Files:**
- Create: `app/db/__init__.py`
- Create: `app/db/schema.sql`
- Create: `app/db/migrations.py`
- Test: `tests/db/test_schema.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/db/test_schema.py`:

```python
import unittest
from pathlib import Path

from app.db.migrations import split_sql_statements


SCHEMA_PATH = Path("app/db/schema.sql")


class SchemaTests(unittest.TestCase):
    def test_schema_declares_required_extensions_and_tables(self):
        sql = SCHEMA_PATH.read_text(encoding="utf-8")

        self.assertIn("create extension if not exists vector", sql.lower())
        for table in (
            "app_threads",
            "app_runs",
            "app_messages",
            "app_agent_trace_events",
            "app_validation_results",
            "rag_corpora",
            "rag_sources",
            "rag_sections",
            "rag_parents",
            "rag_chunks",
            "rag_chunk_embeddings",
            "rag_bm25_tokens",
            "rag_retrieval_logs",
        ):
            self.assertIn(f"create table if not exists {table}", sql.lower())

    def test_schema_indexes_foreign_keys_and_common_filters(self):
        sql = SCHEMA_PATH.read_text(encoding="utf-8").lower()

        for index_name in (
            "app_runs_thread_created_idx",
            "app_messages_thread_ordinal_idx",
            "app_messages_visible_idx",
            "rag_chunks_parent_idx",
            "rag_chunks_symptom_tags_idx",
            "rag_chunk_embeddings_hnsw_idx",
        ):
            self.assertIn(index_name, sql)

    def test_split_sql_statements_ignores_comments_and_empty_chunks(self):
        sql = """
        -- first comment
        create table one(id int);

        -- second comment
        create table two(id int);
        """

        self.assertEqual(
            split_sql_statements(sql),
            ["create table one(id int)", "create table two(id int)"],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.db.test_schema -v
```

Expected: import failure for `app.db.migrations` or missing schema file.

- [ ] **Step 3: Create the db package and migration runner**

Create `app/db/__init__.py` as an empty file.

Create `app/db/migrations.py`:

```python
from pathlib import Path


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    for raw_line in sql.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("--"):
            continue
        current.append(raw_line)
        if line.endswith(";"):
            statement = "\n".join(current).strip()
            statements.append(statement[:-1].strip())
            current = []
    if current:
        statements.append("\n".join(current).strip())
    return statements


async def run_schema_migrations(connection) -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    for statement in split_sql_statements(sql):
        await connection.execute(statement)
```

- [ ] **Step 4: Create the schema**

Create `app/db/schema.sql` with the full schema from the design doc. Include:

```sql
create extension if not exists vector;

create table if not exists app_threads (
  thread_id uuid primary key,
  status text not null,
  created_at timestamptz not null,
  updated_at timestamptz not null,
  metadata jsonb not null default '{}'
);

create table if not exists app_runs (
  run_id uuid primary key,
  thread_id uuid not null references app_threads(thread_id),
  assistant_id text not null,
  status text not null,
  error text null,
  input jsonb not null default '{}',
  context jsonb not null default '{}',
  created_at timestamptz not null,
  updated_at timestamptz not null
);

create table if not exists app_messages (
  id bigserial primary key,
  thread_id uuid not null references app_threads(thread_id),
  run_id uuid null references app_runs(run_id),
  message_id text null,
  ordinal integer not null,
  type text not null,
  role text null,
  name text null,
  tool_call_id text null,
  content jsonb not null,
  tool_calls jsonb null,
  visible boolean not null default false,
  created_at timestamptz not null
);

create table if not exists app_agent_trace_events (
  id bigserial primary key,
  run_id uuid not null references app_runs(run_id),
  thread_id uuid not null references app_threads(thread_id),
  event_type text not null,
  agent text null,
  summary text null,
  payload jsonb not null default '{}',
  created_at timestamptz not null
);

create table if not exists app_validation_results (
  id bigserial primary key,
  run_id uuid not null references app_runs(run_id),
  thread_id uuid not null references app_threads(thread_id),
  validation jsonb not null,
  allowed_terms text[] not null default '{}',
  rewritten boolean not null default false,
  created_at timestamptz not null
);

create table if not exists rag_corpora (
  corpus_id text primary key,
  version text not null,
  status text not null,
  source_manifest_sha256 text not null,
  index_manifest_sha256 text not null,
  embedding_model jsonb not null,
  reranker_model jsonb null,
  vector_dimension integer not null,
  parent_count integer not null,
  chunk_count integer not null,
  created_at timestamptz not null
);

create table if not exists rag_sources (
  source_id text primary key,
  corpus_id text not null references rag_corpora(corpus_id),
  source_type text not null,
  book_id text not null,
  book_title text not null,
  source_file text not null,
  source_hash text not null,
  encoding text not null,
  metadata jsonb not null default '{}'
);

create table if not exists rag_sections (
  section_id text primary key,
  corpus_id text not null references rag_corpora(corpus_id),
  source_id text not null references rag_sources(source_id),
  volume text not null,
  chapter text not null,
  section text not null,
  symptom_tags text[] not null default '{}',
  metadata jsonb not null default '{}'
);

create table if not exists rag_parents (
  parent_id text primary key,
  corpus_id text not null references rag_corpora(corpus_id),
  source_id text not null references rag_sources(source_id),
  section_id text null references rag_sections(section_id),
  source_type text not null,
  book_id text not null,
  book_title text not null,
  source_file text not null,
  source_hash text not null,
  volume text not null,
  chapter text not null,
  section text not null,
  symptom_tags text[] not null default '{}',
  evidence_role text not null,
  original_text text not null,
  normalized_text text not null,
  created_at timestamptz not null
);

create table if not exists rag_chunks (
  chunk_id text primary key,
  parent_id text not null references rag_parents(parent_id),
  corpus_id text not null references rag_corpora(corpus_id),
  row_index integer not null,
  text text not null,
  source_type text not null,
  symptom_tags text[] not null default '{}',
  evidence_role text not null,
  created_at timestamptz not null,
  unique (corpus_id, row_index)
);

create table if not exists rag_chunk_embeddings (
  chunk_id text primary key references rag_chunks(chunk_id),
  corpus_id text not null references rag_corpora(corpus_id),
  embedding vector(1024) not null,
  embedding_model text not null,
  embedding_revision text not null,
  created_at timestamptz not null
);

create table if not exists rag_bm25_tokens (
  chunk_id text primary key references rag_chunks(chunk_id),
  corpus_id text not null references rag_corpora(corpus_id),
  tokens text[] not null,
  created_at timestamptz not null
);

create table if not exists rag_retrieval_logs (
  id bigserial primary key,
  run_id uuid null references app_runs(run_id),
  thread_id uuid null references app_threads(thread_id),
  corpus_id text not null references rag_corpora(corpus_id),
  original_query text not null,
  rewritten_query text not null,
  chief_symptom text null,
  retrieval_mode text not null,
  degraded boolean not null default false,
  degraded_reason text null,
  dense_hits jsonb not null default '[]',
  keyword_hits jsonb not null default '[]',
  fused_hits jsonb not null default '[]',
  final_results jsonb not null default '[]',
  created_at timestamptz not null
);

create index if not exists app_runs_thread_created_idx on app_runs (thread_id, created_at desc);
create index if not exists app_runs_status_updated_idx on app_runs (status, updated_at desc);
create index if not exists app_messages_thread_ordinal_idx on app_messages (thread_id, ordinal);
create index if not exists app_messages_run_idx on app_messages (run_id);
create index if not exists app_messages_visible_idx on app_messages (thread_id, ordinal) where visible = true;
create index if not exists rag_chunks_parent_idx on rag_chunks (parent_id);
create index if not exists rag_chunks_corpus_role_idx on rag_chunks (corpus_id, evidence_role);
create index if not exists rag_chunks_symptom_tags_idx on rag_chunks using gin (symptom_tags);
create index if not exists rag_parents_symptom_tags_idx on rag_parents using gin (symptom_tags);
create index if not exists rag_chunk_embeddings_hnsw_idx on rag_chunk_embeddings using hnsw (embedding vector_cosine_ops);
```

- [ ] **Step 5: Run the test and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.db.test_schema -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add app/db tests/db
git commit -m "feat: add postgres persistence schema"
```

## Task 3: Database Pool and Runtime Store Selection

**Files:**
- Create: `app/db/pool.py`
- Modify: `app/runtime/state.py`
- Test: `tests/db/test_pool.py`

- [ ] **Step 1: Write failing pool tests**

Create `tests/db/test_pool.py`:

```python
import unittest
from unittest.mock import AsyncMock, patch

from app.config import AppSettings
from app.db.pool import create_pool_from_settings


class PoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_pool_requires_database_url(self):
        settings = AppSettings(
            database_url=None,
            postgres_pool_size=10,
            checkpoint_backend="postgres",
            rag_engine="database",
            rag_fallback_file_engine=True,
            elasticsearch_url=None,
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )

        with self.assertRaisesRegex(ValueError, "DATABASE_URL"):
            await create_pool_from_settings(settings)

    async def test_create_pool_uses_configured_size(self):
        settings = AppSettings(
            database_url="postgresql://user:pass@localhost:5432/tcm",
            postgres_pool_size=3,
            checkpoint_backend="postgres",
            rag_engine="database",
            rag_fallback_file_engine=True,
            elasticsearch_url=None,
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )
        fake_pool = object()

        with patch("app.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)) as create_pool:
            pool = await create_pool_from_settings(settings)

        self.assertIs(pool, fake_pool)
        create_pool.assert_awaited_once_with(
            dsn=settings.database_url,
            min_size=1,
            max_size=3,
            command_timeout=60,
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.db.test_pool -v
```

Expected: import failure for `app.db.pool`.

- [ ] **Step 3: Implement pool creation**

Create `app/db/pool.py`:

```python
import asyncpg

from app.config import AppSettings


async def create_pool_from_settings(settings: AppSettings):
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres persistence")
    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=1,
        max_size=settings.postgres_pool_size,
        command_timeout=60,
    )
```

- [ ] **Step 4: Keep `state.py` unchanged until store classes exist**

Do not modify `app/runtime/state.py` in this task. The state selection change is part of Task 5 after concrete Postgres stores exist.

- [ ] **Step 5: Run the test and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.db.test_pool -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add app/db/pool.py tests/db/test_pool.py
git commit -m "feat: add postgres pool helper"
```

## Task 4: Postgres ThreadStore and RunManager

**Files:**
- Create: `app/store/postgres_thread_store.py`
- Create: `app/store/postgres_run_manager.py`
- Test: `tests/store/test_postgres_store.py`

- [ ] **Step 1: Write failing store tests with fake pool**

Create `tests/store/test_postgres_store.py`:

```python
import unittest
from datetime import datetime
from uuid import UUID

from app.store.postgres_run_manager import PostgresRunManager
from app.store.postgres_thread_store import PostgresThreadStore


class FakeConnection:
    def __init__(self):
        self.fetchrow_calls = []
        self.fetch_calls = []
        self.execute_calls = []
        self.rows = {}

    async def fetchrow(self, sql, *args):
        self.fetchrow_calls.append((sql, args))
        if "insert into app_threads" in sql.lower():
            return {
                "thread_id": args[0],
                "created_at": args[1],
                "updated_at": args[1],
                "status": "idle",
                "metadata": {},
            }
        if "insert into app_runs" in sql.lower():
            return {
                "run_id": args[0],
                "thread_id": args[1],
                "assistant_id": args[2],
                "status": "pending",
                "created_at": args[3],
                "updated_at": args[3],
                "error": None,
            }
        return self.rows.get(args[0])

    async def fetch(self, sql, *args):
        self.fetch_calls.append((sql, args))
        return list(self.rows.values())

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))
        return "UPDATE 1"


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self):
        self.connection = FakeConnection()

    def acquire(self):
        return FakeAcquire(self.connection)


class PostgresStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_thread_store_create_returns_thread_record(self):
        pool = FakePool()
        store = PostgresThreadStore(pool)

        record = await store.create()

        UUID(record.thread_id)
        self.assertEqual(record.status, "idle")
        sql = pool.connection.fetchrow_calls[0][0].lower()
        self.assertIn("insert into app_threads", sql)

    async def test_thread_store_update_values_writes_metadata_json(self):
        pool = FakePool()
        store = PostgresThreadStore(pool)

        await store.update_values("00000000-0000-0000-0000-000000000001", {"conversation": []})

        sql, args = pool.connection.execute_calls[0]
        self.assertIn("metadata = metadata ||", sql.lower())
        self.assertEqual(args[1], {"conversation": []})

    async def test_run_manager_create_returns_run_record(self):
        pool = FakePool()
        manager = PostgresRunManager(pool)

        record = await manager.create("00000000-0000-0000-0000-000000000001", "lead_agent")

        UUID(record.run_id)
        self.assertEqual(record.status, "pending")
        self.assertEqual(record.assistant_id, "lead_agent")

    async def test_run_manager_set_status_writes_error(self):
        pool = FakePool()
        manager = PostgresRunManager(pool)

        await manager.set_status("00000000-0000-0000-0000-000000000002", "error", error="boom")

        sql, args = pool.connection.execute_calls[0]
        self.assertIn("update app_runs", sql.lower())
        self.assertEqual(args[1], "error")
        self.assertEqual(args[2], "boom")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.store.test_postgres_store -v
```

Expected: import failure for Postgres store modules.

- [ ] **Step 3: Implement PostgresThreadStore**

Create `app/store/postgres_thread_store.py`:

```python
import uuid
from datetime import datetime
from typing import Any

from app.store.models import ThreadRecord


def _thread_from_row(row) -> ThreadRecord | None:
    if row is None:
        return None
    return ThreadRecord(
        thread_id=str(row["thread_id"]),
        created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
        updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else str(row["updated_at"]),
        status=row["status"],
        values=dict(row.get("metadata") or {}),
    )


class PostgresThreadStore:
    def __init__(self, pool):
        self.pool = pool

    async def create(self) -> ThreadRecord:
        now = datetime.utcnow()
        thread_id = uuid.uuid4()
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                insert into app_threads (thread_id, status, created_at, updated_at, metadata)
                values ($1, 'idle', $2, $2, '{}'::jsonb)
                returning thread_id, created_at, updated_at, status, metadata
                """,
                thread_id,
                now,
            )
        record = _thread_from_row(row)
        if record is None:
            raise RuntimeError("failed to create thread")
        return record

    async def get(self, thread_id: str) -> ThreadRecord | None:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                select thread_id, created_at, updated_at, status, metadata
                from app_threads
                where thread_id = $1
                """,
                uuid.UUID(thread_id),
            )
        return _thread_from_row(row)

    async def list(self) -> list[ThreadRecord]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                select thread_id, created_at, updated_at, status, metadata
                from app_threads
                order by updated_at desc
                """
            )
        return [record for row in rows if (record := _thread_from_row(row))]

    async def update_status(self, thread_id: str, status: str):
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                update app_threads
                set status = $2, updated_at = $3
                where thread_id = $1
                """,
                uuid.UUID(thread_id),
                status,
                datetime.utcnow(),
            )

    async def update_values(self, thread_id: str, values: dict[str, Any]):
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                update app_threads
                set metadata = metadata || $2::jsonb, updated_at = $3
                where thread_id = $1
                """,
                uuid.UUID(thread_id),
                values,
                datetime.utcnow(),
            )
```

- [ ] **Step 4: Implement PostgresRunManager**

Create `app/store/postgres_run_manager.py`:

```python
import uuid
from datetime import datetime

from app.store.models import RunRecord


def _run_from_row(row) -> RunRecord | None:
    if row is None:
        return None
    created_at = row["created_at"]
    updated_at = row["updated_at"]
    return RunRecord(
        run_id=str(row["run_id"]),
        thread_id=str(row["thread_id"]),
        assistant_id=row["assistant_id"],
        status=row["status"],
        created_at=created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
        updated_at=updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at),
        error=row.get("error"),
    )


class PostgresRunManager:
    def __init__(self, pool):
        self.pool = pool

    async def create(self, thread_id: str, assistant_id: str) -> RunRecord:
        now = datetime.utcnow()
        run_id = uuid.uuid4()
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                insert into app_runs (
                  run_id, thread_id, assistant_id, status, input, context, created_at, updated_at
                )
                values ($1, $2, $3, 'pending', '{}'::jsonb, '{}'::jsonb, $4, $4)
                returning run_id, thread_id, assistant_id, status, created_at, updated_at, error
                """,
                run_id,
                uuid.UUID(thread_id),
                assistant_id,
                now,
            )
        record = _run_from_row(row)
        if record is None:
            raise RuntimeError("failed to create run")
        return record

    async def get(self, run_id: str) -> RunRecord | None:
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                select run_id, thread_id, assistant_id, status, created_at, updated_at, error
                from app_runs
                where run_id = $1
                """,
                uuid.UUID(run_id),
            )
        return _run_from_row(row)

    async def set_status(self, run_id: str, status: str, error: str | None = None):
        async with self.pool.acquire() as connection:
            await connection.execute(
                """
                update app_runs
                set status = $2, error = $3, updated_at = $4
                where run_id = $1
                """,
                uuid.UUID(run_id),
                status,
                error,
                datetime.utcnow(),
            )
```

- [ ] **Step 5: Run store tests and fix fake-row compatibility**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.store.test_postgres_store -v
```

Expected: 4 tests pass. If `asyncpg.Record` and dict access differ in the fake, keep `_thread_from_row` and `_run_from_row` using bracket access for required fields and `.get()` only for optional `metadata` and `error`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add app/store/postgres_thread_store.py app/store/postgres_run_manager.py tests/store/test_postgres_store.py
git commit -m "feat: add postgres runtime stores"
```

## Task 5: Runtime State Wiring

**Files:**
- Modify: `app/runtime/state.py`
- Test: `tests/test_runtime_state.py`

- [ ] **Step 1: Write failing state selection tests**

Create `tests/test_runtime_state.py`:

```python
import unittest
from unittest.mock import patch

from app.runtime.state import build_state
from app.store.thread_store import ThreadStore
from app.store.run_manager import RunManager
from app.store.postgres_thread_store import PostgresThreadStore
from app.store.postgres_run_manager import PostgresRunManager


class RuntimeStateTests(unittest.TestCase):
    def test_memory_settings_use_current_in_memory_stores(self):
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "memory"
            get_settings.return_value.rag_engine = "file"
            get_settings.return_value.database_url = None
            get_settings.return_value.postgres_pool_size = 10
            state = build_state(pool=None)

        self.assertIsInstance(state.thread_store, ThreadStore)
        self.assertIsInstance(state.run_manager, RunManager)

    def test_postgres_settings_require_pool_and_use_postgres_stores(self):
        pool = object()
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "postgres"
            get_settings.return_value.rag_engine = "database"
            get_settings.return_value.database_url = "postgresql://x"
            get_settings.return_value.postgres_pool_size = 10
            state = build_state(pool=pool)

        self.assertIsInstance(state.thread_store, PostgresThreadStore)
        self.assertIsInstance(state.run_manager, PostgresRunManager)

    def test_postgres_settings_without_pool_fail_fast(self):
        with patch("app.runtime.state.get_settings") as get_settings:
            get_settings.return_value.checkpoint_backend = "postgres"
            get_settings.return_value.rag_engine = "file"
            get_settings.return_value.database_url = "postgresql://x"
            get_settings.return_value.postgres_pool_size = 10
            with self.assertRaisesRegex(ValueError, "pool"):
                build_state(pool=None)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_state -v
```

Expected: import failure or missing `build_state`.

- [ ] **Step 3: Implement state builder**

Modify `app/runtime/state.py`:

```python
from app.config import get_settings
from app.runtime.stream import StreamBridge
from app.store.postgres_run_manager import PostgresRunManager
from app.store.postgres_thread_store import PostgresThreadStore
from app.store.run_manager import RunManager
from app.store.thread_store import ThreadStore


class AppState:
    """
    Runtime 全局状态。

    - thread_store 管理会话
    - run_manager 管理运行任务
    - bridge 管理 SSE 流
    """

    def __init__(self, *, thread_store, run_manager, bridge: StreamBridge):
        self.thread_store = thread_store
        self.run_manager = run_manager
        self.bridge = bridge


def build_state(pool=None) -> AppState:
    settings = get_settings()
    wants_postgres = (
        settings.checkpoint_backend == "postgres"
        or settings.rag_engine == "database"
    )
    if wants_postgres:
        if pool is None:
            raise ValueError("Postgres runtime state requires a database pool")
        return AppState(
            thread_store=PostgresThreadStore(pool),
            run_manager=PostgresRunManager(pool),
            bridge=StreamBridge(),
        )
    return AppState(
        thread_store=ThreadStore(),
        run_manager=RunManager(),
        bridge=StreamBridge(),
    )


state = build_state(pool=None)
```

- [ ] **Step 4: Run state tests and current clarification tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_state tests.test_clarification_flow -v
```

Expected: all tests pass with default memory/file settings.

- [ ] **Step 5: Commit**

Run:

```powershell
git add app/runtime/state.py tests/test_runtime_state.py
git commit -m "feat: wire configurable runtime state"
```

## Task 6: Checkpointer Factory

**Files:**
- Create: `app/checkpoints/__init__.py`
- Create: `app/checkpoints/factory.py`
- Modify: `app/agents/lead_agent/agent.py`
- Test: `tests/checkpoints/test_factory.py`

- [ ] **Step 1: Write failing factory tests**

Create `tests/checkpoints/test_factory.py`:

```python
import unittest
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from app.checkpoints.factory import get_checkpointer, reset_checkpointer_cache
from app.config import AppSettings


def settings(backend: str) -> AppSettings:
    return AppSettings(
        database_url="postgresql://user:pass@localhost:5432/tcm",
        postgres_pool_size=10,
        checkpoint_backend=backend,
        rag_engine="file",
        rag_fallback_file_engine=True,
        elasticsearch_url=None,
        elasticsearch_rag_index_alias="tcm_rag_chunks_current",
        elasticsearch_analyzer="standard",
    )


class CheckpointerFactoryTests(unittest.TestCase):
    def tearDown(self):
        reset_checkpointer_cache()

    def test_memory_backend_returns_single_in_memory_saver(self):
        first = get_checkpointer(settings("memory"))
        second = get_checkpointer(settings("memory"))

        self.assertIsInstance(first, InMemorySaver)
        self.assertIs(first, second)

    def test_postgres_backend_requires_database_url(self):
        bad = settings("postgres")
        bad = AppSettings(
            database_url=None,
            postgres_pool_size=10,
            checkpoint_backend="postgres",
            rag_engine="file",
            rag_fallback_file_engine=True,
            elasticsearch_url=None,
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )

        with self.assertRaisesRegex(ValueError, "DATABASE_URL"):
            get_checkpointer(bad)

    def test_postgres_backend_uses_langgraph_postgres_saver(self):
        class FakeSaver:
            @classmethod
            def from_conn_string(cls, value):
                return {"conn": value}

        with patch("app.checkpoints.factory.AsyncPostgresSaver", FakeSaver):
            checkpointer = get_checkpointer(settings("postgres"))

        self.assertEqual(checkpointer, {"conn": "postgresql://user:pass@localhost:5432/tcm"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.checkpoints.test_factory -v
```

Expected: import failure for `app.checkpoints.factory`.

- [ ] **Step 3: Implement factory**

Create `app/checkpoints/__init__.py` as an empty file.

Create `app/checkpoints/factory.py`:

```python
from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import AppSettings


@lru_cache(maxsize=2)
def _memory_checkpointer():
    return InMemorySaver()


@lru_cache(maxsize=2)
def _postgres_checkpointer(database_url: str):
    return AsyncPostgresSaver.from_conn_string(database_url)


def get_checkpointer(settings: AppSettings):
    if settings.checkpoint_backend == "memory":
        return _memory_checkpointer()
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for Postgres checkpointer")
    return _postgres_checkpointer(settings.database_url)


def reset_checkpointer_cache() -> None:
    _memory_checkpointer.cache_clear()
    _postgres_checkpointer.cache_clear()
```

- [ ] **Step 4: Wire lead agent**

Modify `app/agents/lead_agent/agent.py`:

```python
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.agents.lead_agent.prompt import SYSTEM_PROMPT
from app.checkpoints.factory import get_checkpointer
from app.config import get_settings
from app.middlewares.clarification_middleware import ClarificationMiddleware
from app.tools.tools import get_available_tools

load_dotenv()


def make_lead_agent(context: dict[str, Any] | None = None):
    context = context or {}
    settings = get_settings()

    model_name = context.get("model_name") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        temperature=context.get("temperature", 0.3),
    )

    tools = get_available_tools(context=context)

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=get_checkpointer(settings),
        middleware=[ClarificationMiddleware()],
    )
```

- [ ] **Step 5: Run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.checkpoints.test_factory tests.test_clarification_flow -v
```

Expected: checkpointer tests and clarification tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add app/checkpoints app/agents/lead_agent/agent.py tests/checkpoints/test_factory.py
git commit -m "feat: add configurable langgraph checkpointer"
```

## Task 7: RAG Artifact Loader

**Files:**
- Create: `app/rag/database/__init__.py`
- Create: `app/rag/database/artifacts.py`
- Test: `tests/rag/database/test_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

Create `tests/rag/database/test_artifacts.py`:

```python
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from app.rag.database.artifacts import load_artifact_bundle


class ArtifactLoaderTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_loader_rejects_vector_dimension_mismatch(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            index = root / "index"
            corpus.mkdir()
            index.mkdir()
            self.write_jsonl(corpus / "parents.jsonl", [{
                "parent_id": "p1",
                "source_type": "ancient_book",
                "book_id": "jing_yue_quan_shu",
                "book_title": "景岳全书",
                "source_file": "637-景岳全书.txt",
                "source_hash": "A" * 64,
                "volume": "卷一",
                "chapter": "头痛",
                "section": "论证",
                "symptom_tags": ["头痛"],
                "evidence_role": "syndrome_pattern",
                "original_text": "头痛恶风。",
                "normalized_text": "头痛恶风。",
            }])
            self.write_jsonl(corpus / "chunks.jsonl", [{
                "chunk_id": "c1",
                "parent_id": "p1",
                "text": "头痛恶风",
                "source_type": "ancient_book",
                "symptom_tags": ["头痛"],
                "evidence_role": "syndrome_pattern",
            }])
            self.write_jsonl(index / "rows.jsonl", [{
                "chunk_id": "c1",
                "parent_id": "p1",
                "text": "头痛恶风",
                "source_type": "ancient_book",
                "symptom_tags": ["头痛"],
                "evidence_role": "syndrome_pattern",
            }])
            self.write_jsonl(index / "bm25_tokens.jsonl", [{"chunk_id": "c1", "tokens": ["头痛", "恶风"]}])
            np.save(index / "dense.npy", np.asarray([[1.0, 0.0]], dtype=np.float32), allow_pickle=False)
            (corpus / "manifest.json").write_text(
                json.dumps({"status": "ready", "parent_count": 1, "chunk_count": 1, "version": "v1.0.0"}),
                encoding="utf-8",
            )
            (index / "manifest.json").write_text(
                json.dumps({
                    "status": "ready",
                    "row_count": 1,
                    "vector_dimension": 1024,
                    "embedding_model": {"model": "BAAI/bge-m3", "revision": "r1"},
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "vector_dimension"):
                load_artifact_bundle(corpus, index)

    def test_loader_returns_ordered_rows_tokens_and_vectors(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus"
            index = root / "index"
            corpus.mkdir()
            index.mkdir()
            parent = {
                "parent_id": "p1",
                "source_type": "ancient_book",
                "book_id": "jing_yue_quan_shu",
                "book_title": "景岳全书",
                "source_file": "637-景岳全书.txt",
                "source_hash": "A" * 64,
                "volume": "卷一",
                "chapter": "头痛",
                "section": "论证",
                "symptom_tags": ["头痛"],
                "evidence_role": "syndrome_pattern",
                "original_text": "头痛恶风。",
                "normalized_text": "头痛恶风。",
            }
            chunk = {
                "chunk_id": "c1",
                "parent_id": "p1",
                "text": "头痛恶风",
                "source_type": "ancient_book",
                "symptom_tags": ["头痛"],
                "evidence_role": "syndrome_pattern",
            }
            self.write_jsonl(corpus / "parents.jsonl", [parent])
            self.write_jsonl(corpus / "chunks.jsonl", [chunk])
            self.write_jsonl(index / "rows.jsonl", [chunk])
            self.write_jsonl(index / "bm25_tokens.jsonl", [{"chunk_id": "c1", "tokens": ["头痛", "恶风"]}])
            np.save(index / "dense.npy", np.zeros((1, 1024), dtype=np.float32), allow_pickle=False)
            (corpus / "manifest.json").write_text(
                json.dumps({"status": "ready", "parent_count": 1, "chunk_count": 1, "version": "v1.0.0"}),
                encoding="utf-8",
            )
            (index / "manifest.json").write_text(
                json.dumps({
                    "status": "ready",
                    "row_count": 1,
                    "vector_dimension": 1024,
                    "embedding_model": {"model": "BAAI/bge-m3", "revision": "r1"},
                }),
                encoding="utf-8",
            )

            bundle = load_artifact_bundle(corpus, index)

        self.assertEqual(bundle.corpus_id, "ancient-books-v1.0.0")
        self.assertEqual(bundle.parents[0]["parent_id"], "p1")
        self.assertEqual(bundle.chunks[0]["chunk_id"], "c1")
        self.assertEqual(bundle.tokens_by_chunk_id["c1"], ["头痛", "恶风"])
        self.assertEqual(bundle.dense.shape, (1, 1024))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_artifacts -v
```

Expected: import failure for `app.rag.database.artifacts`.

- [ ] **Step 3: Implement artifact bundle loader**

Create `app/rag/database/__init__.py` as an empty file.

Create `app/rag/database/artifacts.py` with:

```python
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RagArtifactBundle:
    corpus_id: str
    corpus_manifest: dict
    index_manifest: dict
    parents: list[dict]
    chunks: list[dict]
    rows: list[dict]
    tokens_by_chunk_id: dict[str, list[str]]
    dense: np.ndarray


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_artifact_bundle(corpus_dir: Path, index_dir: Path) -> RagArtifactBundle:
    corpus_manifest = _read_json(corpus_dir / "manifest.json")
    index_manifest = _read_json(index_dir / "manifest.json")
    if corpus_manifest.get("status") != "ready":
        raise ValueError("corpus manifest status must be ready")
    if index_manifest.get("status") != "ready":
        raise ValueError("index manifest status must be ready")
    if int(index_manifest.get("vector_dimension", 0)) != 1024:
        raise ValueError("vector_dimension must be 1024 for BGE-M3 pgvector storage")

    parents = _read_jsonl(corpus_dir / "parents.jsonl")
    chunks = _read_jsonl(corpus_dir / "chunks.jsonl")
    rows = _read_jsonl(index_dir / "rows.jsonl")
    token_rows = _read_jsonl(index_dir / "bm25_tokens.jsonl")
    dense = np.load(index_dir / "dense.npy", allow_pickle=False)

    row_ids = [row["chunk_id"] for row in rows]
    token_ids = [row["chunk_id"] for row in token_rows]
    chunk_ids = [row["chunk_id"] for row in chunks]
    if row_ids != token_ids or row_ids != chunk_ids:
        raise ValueError("rows, chunks, and bm25 token order must match")
    if dense.shape != (len(rows), 1024):
        raise ValueError("dense.npy shape does not match row_count and vector_dimension")
    if len({parent["parent_id"] for parent in parents}) != len(parents):
        raise ValueError("duplicate parent_id in artifact bundle")
    parent_ids = {parent["parent_id"] for parent in parents}
    orphan_ids = [row["chunk_id"] for row in chunks if row["parent_id"] not in parent_ids]
    if orphan_ids:
        raise ValueError(f"orphan chunks found: {orphan_ids[:3]}")

    version = corpus_manifest.get("version") or index_manifest.get("version") or "v1.0.0"
    corpus_id = f"ancient-books-{version}"
    return RagArtifactBundle(
        corpus_id=corpus_id,
        corpus_manifest=corpus_manifest,
        index_manifest=index_manifest,
        parents=parents,
        chunks=chunks,
        rows=rows,
        tokens_by_chunk_id={row["chunk_id"]: row["tokens"] for row in token_rows},
        dense=dense.astype(np.float32, copy=False),
    )
```

- [ ] **Step 4: Run artifact tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_artifacts -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add app/rag/database tests/rag/database/test_artifacts.py
git commit -m "feat: load rag database import artifacts"
```

## Task 8: RAG Repository

**Files:**
- Create: `app/rag/database/repository.py`
- Test: `tests/rag/database/test_repository.py`

- [ ] **Step 1: Write failing repository tests**

Create `tests/rag/database/test_repository.py`:

```python
import unittest

import numpy as np

from app.rag.database.repository import (
    build_dense_search_sql,
    prepare_vector,
    rows_to_parent_map,
)


class RepositoryTests(unittest.TestCase):
    def test_prepare_vector_rejects_wrong_dimension(self):
        with self.assertRaisesRegex(ValueError, "1024"):
            prepare_vector(np.asarray([1.0, 2.0], dtype=np.float32))

    def test_prepare_vector_serializes_1024_floats(self):
        vector = np.zeros(1024, dtype=np.float32)
        vector[0] = 1.0

        serialized = prepare_vector(vector)

        self.assertTrue(serialized.startswith("[1.0,"))
        self.assertTrue(serialized.endswith("]"))

    def test_dense_search_sql_filters_corpus_symptom_and_role(self):
        sql = build_dense_search_sql()

        self.assertIn("rag_chunk_embeddings", sql)
        self.assertIn("rag_chunks", sql)
        self.assertIn("c.corpus_id = $1", sql)
        self.assertIn("$3 = any(c.symptom_tags)", sql)
        self.assertIn("c.evidence_role = any($4::text[])", sql)
        self.assertIn("order by e.embedding <=> $2::vector", sql.lower())

    def test_rows_to_parent_map_indexes_by_parent_id(self):
        rows = [
            {"parent_id": "p1", "original_text": "a"},
            {"parent_id": "p2", "original_text": "b"},
        ]

        result = rows_to_parent_map(rows)

        self.assertEqual(result["p1"]["original_text"], "a")
        self.assertEqual(result["p2"]["original_text"], "b")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_repository -v
```

Expected: import failure for `app.rag.database.repository`.

- [ ] **Step 3: Implement repository helpers and class**

Create `app/rag/database/repository.py`:

```python
from datetime import datetime
from typing import Iterable

import numpy as np

from app.rag.database.artifacts import RagArtifactBundle


DEFAULT_EVIDENCE_ROLES = [
    "diagnostic_method",
    "symptom_feature",
    "syndrome_pattern",
    "pathogenesis",
    "differential",
    "case",
]


def prepare_vector(vector: np.ndarray) -> str:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.shape[0] != 1024:
        raise ValueError("pgvector query vector must have dimension 1024")
    return "[" + ",".join(str(float(value)) for value in array.tolist()) + "]"


def build_dense_search_sql() -> str:
    return """
    select
      c.chunk_id,
      c.parent_id,
      c.text as matched_child,
      c.symptom_tags,
      c.evidence_role,
      e.embedding <=> $2::vector as distance
    from rag_chunk_embeddings e
    join rag_chunks c on c.chunk_id = e.chunk_id
    where c.corpus_id = $1
      and ($3::text is null or $3 = any(c.symptom_tags))
      and c.evidence_role = any($4::text[])
    order by e.embedding <=> $2::vector, c.chunk_id
    limit $5
    """


def rows_to_parent_map(rows: Iterable[dict]) -> dict[str, dict]:
    return {row["parent_id"]: dict(row) for row in rows}


class RagPostgresRepository:
    def __init__(self, pool):
        self.pool = pool

    async def import_bundle(self, bundle: RagArtifactBundle) -> dict:
        now = datetime.utcnow()
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    insert into rag_corpora (
                      corpus_id, version, status, source_manifest_sha256,
                      index_manifest_sha256, embedding_model, reranker_model,
                      vector_dimension, parent_count, chunk_count, created_at
                    )
                    values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    on conflict (corpus_id) do update set
                      status = excluded.status,
                      source_manifest_sha256 = excluded.source_manifest_sha256,
                      index_manifest_sha256 = excluded.index_manifest_sha256,
                      embedding_model = excluded.embedding_model,
                      vector_dimension = excluded.vector_dimension,
                      parent_count = excluded.parent_count,
                      chunk_count = excluded.chunk_count
                    """,
                    bundle.corpus_id,
                    bundle.corpus_manifest.get("version", "v1.0.0"),
                    "ready",
                    bundle.corpus_manifest.get("source_manifest_sha256", ""),
                    bundle.index_manifest.get("corpus_manifest_sha256", ""),
                    bundle.index_manifest.get("embedding_model", {}),
                    None,
                    1024,
                    len(bundle.parents),
                    len(bundle.chunks),
                    now,
                )
                for parent in bundle.parents:
                    source_id = f"{bundle.corpus_id}:{parent['book_id']}:{parent['source_hash'][:12]}"
                    await connection.execute(
                        """
                        insert into rag_sources (
                          source_id, corpus_id, source_type, book_id, book_title,
                          source_file, source_hash, encoding, metadata
                        )
                        values ($1,$2,$3,$4,$5,$6,$7,'cp936','{}'::jsonb)
                        on conflict (source_id) do nothing
                        """,
                        source_id,
                        bundle.corpus_id,
                        parent["source_type"],
                        parent["book_id"],
                        parent["book_title"],
                        parent["source_file"],
                        parent["source_hash"],
                    )
                    await connection.execute(
                        """
                        insert into rag_parents (
                          parent_id, corpus_id, source_id, source_type, book_id,
                          book_title, source_file, source_hash, volume, chapter,
                          section, symptom_tags, evidence_role, original_text,
                          normalized_text, created_at
                        )
                        values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
                        on conflict (parent_id) do update set
                          original_text = excluded.original_text,
                          normalized_text = excluded.normalized_text
                        """,
                        parent["parent_id"],
                        bundle.corpus_id,
                        source_id,
                        parent["source_type"],
                        parent["book_id"],
                        parent["book_title"],
                        parent["source_file"],
                        parent["source_hash"],
                        parent["volume"],
                        parent["chapter"],
                        parent["section"],
                        parent["symptom_tags"],
                        parent["evidence_role"],
                        parent["original_text"],
                        parent["normalized_text"],
                        now,
                    )
                for index, chunk in enumerate(bundle.chunks):
                    await connection.execute(
                        """
                        insert into rag_chunks (
                          chunk_id, parent_id, corpus_id, row_index, text,
                          source_type, symptom_tags, evidence_role, created_at
                        )
                        values ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                        on conflict (chunk_id) do update set
                          text = excluded.text,
                          row_index = excluded.row_index
                        """,
                        chunk["chunk_id"],
                        chunk["parent_id"],
                        bundle.corpus_id,
                        index,
                        chunk["text"],
                        chunk["source_type"],
                        chunk["symptom_tags"],
                        chunk["evidence_role"],
                        now,
                    )
                    await connection.execute(
                        """
                        insert into rag_bm25_tokens (chunk_id, corpus_id, tokens, created_at)
                        values ($1,$2,$3,$4)
                        on conflict (chunk_id) do update set tokens = excluded.tokens
                        """,
                        chunk["chunk_id"],
                        bundle.corpus_id,
                        bundle.tokens_by_chunk_id[chunk["chunk_id"]],
                        now,
                    )
                    await connection.execute(
                        """
                        insert into rag_chunk_embeddings (
                          chunk_id, corpus_id, embedding, embedding_model,
                          embedding_revision, created_at
                        )
                        values ($1,$2,$3::vector,$4,$5,$6)
                        on conflict (chunk_id) do update set embedding = excluded.embedding
                        """,
                        chunk["chunk_id"],
                        bundle.corpus_id,
                        prepare_vector(bundle.dense[index]),
                        bundle.index_manifest["embedding_model"]["model"],
                        bundle.index_manifest["embedding_model"]["revision"],
                        now,
                    )
        return {"corpus_id": bundle.corpus_id, "chunk_count": len(bundle.chunks)}

    async def dense_search(self, corpus_id: str, vector: np.ndarray, chief_symptom: str | None, top_k: int):
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                build_dense_search_sql(),
                corpus_id,
                prepare_vector(vector),
                chief_symptom,
                DEFAULT_EVIDENCE_ROLES,
                top_k,
            )
        return [
            {
                "chunk_id": row["chunk_id"],
                "parent_id": row["parent_id"],
                "matched_child": row["matched_child"],
                "symptom_tags": list(row["symptom_tags"]),
                "evidence_role": row["evidence_role"],
                "distance": float(row["distance"]),
            }
            for row in rows
        ]

    async def load_parents(self, parent_ids: list[str]) -> dict[str, dict]:
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                select *
                from rag_parents
                where parent_id = any($1::text[])
                """,
                parent_ids,
            )
        return rows_to_parent_map(rows)
```

- [ ] **Step 4: Run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_repository -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add app/rag/database/repository.py tests/rag/database/test_repository.py
git commit -m "feat: add rag postgres repository"
```

## Task 9: Elasticsearch Chunk Index

**Files:**
- Create: `app/rag/database/elasticsearch_index.py`
- Test: `tests/rag/database/test_elasticsearch_index.py`

- [ ] **Step 1: Write failing ES tests**

Create `tests/rag/database/test_elasticsearch_index.py`:

```python
import unittest

from app.rag.database.elasticsearch_index import (
    build_chunk_document,
    build_index_name,
    build_keyword_query,
)


class ElasticsearchIndexTests(unittest.TestCase):
    def test_build_index_name_normalizes_version(self):
        self.assertEqual(
            build_index_name("v1.0.0"),
            "tcm_rag_chunks_v1_0_0",
        )

    def test_build_chunk_document_uses_chunk_id_as_id_source(self):
        parent = {
            "book_id": "jing_yue_quan_shu",
            "book_title": "景岳全书",
            "source_file": "637-景岳全书.txt",
            "source_hash": "A" * 64,
            "volume": "卷一",
            "chapter": "头痛",
            "section": "论证",
        }
        chunk = {
            "chunk_id": "c1",
            "parent_id": "p1",
            "corpus_id": "ancient-books-v1.0.0",
            "row_index": 0,
            "text": "头痛恶风",
            "symptom_tags": ["头痛"],
            "evidence_role": "syndrome_pattern",
        }

        document = build_chunk_document(chunk, parent, "v1.0.0")

        self.assertEqual(document["chunk_id"], "c1")
        self.assertEqual(document["parent_id"], "p1")
        self.assertEqual(document["book_title"], "景岳全书")
        self.assertEqual(document["index_version"], "v1.0.0")

    def test_keyword_query_filters_corpus_symptom_and_role(self):
        query = build_keyword_query(
            rewritten_query="头痛恶风",
            corpus_id="ancient-books-v1.0.0",
            chief_symptom="头痛",
            evidence_roles=["syndrome_pattern"],
            top_k=20,
        )

        self.assertEqual(query["size"], 20)
        filters = query["query"]["bool"]["filter"]
        self.assertIn({"term": {"corpus_id": "ancient-books-v1.0.0"}}, filters)
        self.assertIn({"term": {"symptom_tags": "头痛"}}, filters)
        self.assertIn({"terms": {"evidence_role": ["syndrome_pattern"]}}, filters)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_elasticsearch_index -v
```

Expected: import failure for `app.rag.database.elasticsearch_index`.

- [ ] **Step 3: Implement Elasticsearch helper module**

Create `app/rag/database/elasticsearch_index.py`:

```python
from elasticsearch import AsyncElasticsearch


def build_index_name(version: str) -> str:
    return "tcm_rag_chunks_" + version.replace(".", "_").replace("-", "_")


def build_chunk_document(chunk: dict, parent: dict, index_version: str) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "parent_id": chunk["parent_id"],
        "corpus_id": chunk["corpus_id"],
        "book_id": parent["book_id"],
        "book_title": parent["book_title"],
        "source_file": parent["source_file"],
        "source_hash": parent["source_hash"],
        "volume": parent["volume"],
        "chapter": parent["chapter"],
        "section": parent["section"],
        "text": chunk["text"],
        "symptom_tags": chunk["symptom_tags"],
        "evidence_role": chunk["evidence_role"],
        "row_index": chunk["row_index"],
        "index_version": index_version,
    }


def build_keyword_query(
    *,
    rewritten_query: str,
    corpus_id: str,
    chief_symptom: str | None,
    evidence_roles: list[str],
    top_k: int,
) -> dict:
    filters: list[dict] = [
        {"term": {"corpus_id": corpus_id}},
        {"terms": {"evidence_role": evidence_roles}},
    ]
    if chief_symptom:
        filters.append({"term": {"symptom_tags": chief_symptom}})
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": rewritten_query,
                            "fields": ["text^3", "chapter^2", "section^2", "book_title"],
                        }
                    }
                ],
                "filter": filters,
            }
        },
    }


class ElasticsearchKeywordIndex:
    def __init__(self, client: AsyncElasticsearch, alias: str):
        self.client = client
        self.alias = alias

    async def search(self, *, rewritten_query: str, corpus_id: str, chief_symptom: str | None, evidence_roles: list[str], top_k: int):
        response = await self.client.search(
            index=self.alias,
            body=build_keyword_query(
                rewritten_query=rewritten_query,
                corpus_id=corpus_id,
                chief_symptom=chief_symptom,
                evidence_roles=evidence_roles,
                top_k=top_k,
            ),
        )
        hits = response.get("hits", {}).get("hits", [])
        return [
            {
                "chunk_id": hit["_source"]["chunk_id"],
                "parent_id": hit["_source"]["parent_id"],
                "matched_child": hit["_source"]["text"],
                "score": float(hit.get("_score") or 0.0),
            }
            for hit in hits
        ]
```

- [ ] **Step 4: Run ES tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_elasticsearch_index -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add app/rag/database/elasticsearch_index.py tests/rag/database/test_elasticsearch_index.py
git commit -m "feat: add elasticsearch rag keyword index"
```

## Task 10: Database Retrieval Engine

**Files:**
- Create: `app/rag/database/engine.py`
- Test: `tests/rag/database/test_engine.py`

- [ ] **Step 1: Write failing engine tests**

Create `tests/rag/database/test_engine.py`:

```python
import unittest

import numpy as np

from app.rag.database.engine import DatabaseRetrievalEngine


class FakeEncoder:
    def encode(self, texts):
        return np.asarray([[1.0] + [0.0] * 1023 for _ in texts], dtype=np.float32)


class FakeReranker:
    def score(self, pairs):
        return [1.0 - index / 10 for index, _ in enumerate(pairs)]


class FakeRepository:
    async def dense_search(self, corpus_id, vector, chief_symptom, top_k):
        return [
            {
                "chunk_id": "c1",
                "parent_id": "p1",
                "matched_child": "头痛恶风",
                "distance": 0.1,
                "symptom_tags": ["头痛"],
                "evidence_role": "syndrome_pattern",
            }
        ]

    async def load_parents(self, parent_ids):
        return {
            "p1": {
                "parent_id": "p1",
                "source_type": "ancient_book",
                "book_title": "景岳全书",
                "source_file": "637-景岳全书.txt",
                "volume": "卷一",
                "chapter": "头痛",
                "section": "论证",
                "symptom_tags": ["头痛"],
                "evidence_role": "syndrome_pattern",
                "original_text": "头痛恶风。",
            }
        }


class FakeKeywordIndex:
    async def search(self, *, rewritten_query, corpus_id, chief_symptom, evidence_roles, top_k):
        return [
            {
                "chunk_id": "c1",
                "parent_id": "p1",
                "matched_child": "头痛恶风",
                "score": 2.0,
            }
        ]


class FailingKeywordIndex:
    async def search(self, **kwargs):
        raise RuntimeError("es unavailable")


class DatabaseEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_retrieval_assigns_citations_and_sources(self):
        engine = DatabaseRetrievalEngine(
            corpus_id="ancient-books-v1.0.0",
            repository=FakeRepository(),
            keyword_index=FakeKeywordIndex(),
            encoder=FakeEncoder(),
            reranker=FakeReranker(),
            settings={"dense_top_k": 20, "bm25_top_k": 20, "rrf_k": 60, "reranker_candidate_k": 40, "final_top_k": 5},
        )

        result = await engine.retrieve("头痛恶风", chief_symptom="头痛", mode="hybrid", top_k=5)

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["degraded"])
        self.assertEqual(result["retrieval_mode"], "hybrid")
        self.assertEqual(result["results"][0]["citation_id"], "E1")
        self.assertEqual(result["results"][0]["retrieval_sources"], ["bm25", "dense"])

    async def test_es_failure_degrades_to_vector(self):
        engine = DatabaseRetrievalEngine(
            corpus_id="ancient-books-v1.0.0",
            repository=FakeRepository(),
            keyword_index=FailingKeywordIndex(),
            encoder=FakeEncoder(),
            reranker=FakeReranker(),
            settings={"dense_top_k": 20, "bm25_top_k": 20, "rrf_k": 60, "reranker_candidate_k": 40, "final_top_k": 5},
        )

        result = await engine.retrieve("头痛恶风", chief_symptom="头痛", mode="hybrid", top_k=5)

        self.assertTrue(result["degraded"])
        self.assertEqual(result["retrieval_mode"], "vector")
        self.assertIn("es unavailable", result["degraded_reason"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_engine -v
```

Expected: import failure for `app.rag.database.engine`.

- [ ] **Step 3: Implement database retrieval engine**

Create `app/rag/database/engine.py`:

```python
from app.rag.ancient_books.runtime import reciprocal_rank_fusion
from app.rag.database.repository import DEFAULT_EVIDENCE_ROLES


class DatabaseRetrievalEngine:
    def __init__(self, *, corpus_id: str, repository, keyword_index, encoder, reranker, settings: dict):
        self.corpus_id = corpus_id
        self.repository = repository
        self.keyword_index = keyword_index
        self.encoder = encoder
        self.reranker = reranker
        self.settings = settings

    async def retrieve(self, query: str, *, chief_symptom: str | None, mode: str = "hybrid", top_k: int = 5) -> dict:
        if mode not in {"hybrid", "vector", "keyword"}:
            mode = "hybrid"

        dense_hits = []
        keyword_hits = []
        degraded = False
        degraded_reason = None
        actual_mode = mode

        if mode != "keyword":
            query_vector = self.encoder.encode([query])[0]
            dense_hits = await self.repository.dense_search(
                self.corpus_id,
                query_vector,
                chief_symptom,
                int(self.settings["dense_top_k"]),
            )

        if mode != "vector":
            try:
                keyword_hits = await self.keyword_index.search(
                    rewritten_query=query,
                    corpus_id=self.corpus_id,
                    chief_symptom=chief_symptom,
                    evidence_roles=DEFAULT_EVIDENCE_ROLES,
                    top_k=int(self.settings["bm25_top_k"]),
                )
            except Exception as error:
                degraded = True
                degraded_reason = str(error)
                if dense_hits:
                    actual_mode = "vector"
                else:
                    actual_mode = "keyword"

        rankings = {}
        if keyword_hits:
            rankings["bm25"] = [hit["chunk_id"] for hit in keyword_hits]
        if dense_hits:
            rankings["dense"] = [hit["chunk_id"] for hit in dense_hits]
        if not rankings:
            return {
                "status": "insufficient_evidence",
                "retrieval_mode": actual_mode,
                "degraded": degraded,
                "degraded_reason": degraded_reason,
                "results": [],
            }

        fused = reciprocal_rank_fusion(rankings, rrf_k=int(self.settings["rrf_k"]))
        hit_by_chunk = {}
        for source, hits in (("dense", dense_hits), ("bm25", keyword_hits)):
            for rank, hit in enumerate(hits, start=1):
                current = hit_by_chunk.setdefault(hit["chunk_id"], {**hit, "retrieval_sources": []})
                current["retrieval_sources"].append(source)
                current[f"{source}_rank"] = rank

        candidate_ids = [chunk_id for chunk_id, _ in fused[: int(self.settings["reranker_candidate_k"])]]
        pairs = [[query, hit_by_chunk[chunk_id]["matched_child"]] for chunk_id in candidate_ids]
        scores = self.reranker.score(pairs) if pairs else []
        ranked = sorted(zip(candidate_ids, scores), key=lambda item: (-float(item[1]), item[0]))
        parent_ids = [hit_by_chunk[chunk_id]["parent_id"] for chunk_id, _ in ranked]
        parents = await self.repository.load_parents(parent_ids)

        results = []
        seen_parent_ids = set()
        for chunk_id, score in ranked:
            hit = hit_by_chunk[chunk_id]
            parent_id = hit["parent_id"]
            if parent_id in seen_parent_ids or parent_id not in parents:
                continue
            parent = parents[parent_id]
            result = {
                **parent,
                **hit,
                "score": float(score),
                "citation_id": f"E{len(results) + 1}",
                "content": parent["original_text"],
                "retrieval_sources": sorted(hit["retrieval_sources"]),
            }
            results.append(result)
            seen_parent_ids.add(parent_id)
            if len(results) >= min(int(top_k), int(self.settings["final_top_k"]), 5):
                break

        return {
            "status": "ok" if results else "insufficient_evidence",
            "retrieval_mode": actual_mode,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "results": results,
        }
```

- [ ] **Step 4: Run database engine tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_engine -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add app/rag/database/engine.py tests/rag/database/test_engine.py
git commit -m "feat: add database rag retrieval engine"
```

## Task 11: RAG Engine Routing

**Files:**
- Modify: `app/rag/vector_store.py`
- Modify: `app/rag/retriever.py`
- Modify: `app/tools/builtins/retrieval_tool.py`
- Modify: `tests/test_rag_tool.py`
- Test: `tests/rag/database/test_engine_routing.py`

- [ ] **Step 1: Write failing routing tests**

Create `tests/rag/database/test_engine_routing.py`:

```python
import unittest
from unittest.mock import patch

from app.config import AppSettings
from app.rag.retriever import resolve_retrieval_result
from app.rag.vector_store import get_configured_retrieval_engine


class EngineRoutingTests(unittest.TestCase):
    def settings(self, rag_engine: str) -> AppSettings:
        return AppSettings(
            database_url="postgresql://user:pass@localhost:5432/tcm",
            postgres_pool_size=10,
            checkpoint_backend="memory",
            rag_engine=rag_engine,
            rag_fallback_file_engine=True,
            elasticsearch_url="http://localhost:9200",
            elasticsearch_rag_index_alias="tcm_rag_chunks_current",
            elasticsearch_analyzer="standard",
        )

    def test_file_engine_uses_existing_production_engine(self):
        with patch("app.rag.vector_store.get_production_engine", return_value="file-engine"):
            engine = get_configured_retrieval_engine(self.settings("file"))

        self.assertEqual(engine, "file-engine")

    def test_database_engine_uses_database_factory(self):
        with patch("app.rag.vector_store.get_database_engine", return_value="database-engine"):
            engine = get_configured_retrieval_engine(self.settings("database"))

        self.assertEqual(engine, "database-engine")

    async def async_payload(self):
        return {"status": "ok"}

    def test_resolve_retrieval_result_accepts_sync_payload(self):
        self.assertEqual(resolve_retrieval_result({"status": "ok"}), {"status": "ok"})

    def test_resolve_retrieval_result_rejects_async_payload_in_sync_path(self):
        coroutine = self.async_payload()
        try:
            with self.assertRaisesRegex(RuntimeError, "async"):
                resolve_retrieval_result(coroutine)
        finally:
            coroutine.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_engine_routing -v
```

Expected: missing `get_configured_retrieval_engine` or `resolve_retrieval_result`.

- [ ] **Step 3: Add routing function**

Modify `app/rag/vector_store.py`:

```python
def get_database_engine():
    raise RuntimeError("Database RAG engine requires application startup wiring")


def get_configured_retrieval_engine(settings=None):
    if settings is None:
        from app.config import get_settings
        settings = get_settings()
    if settings.rag_engine == "database":
        return get_database_engine()
    return get_production_engine()
```

Keep existing `get_production_engine()` unchanged.

- [ ] **Step 4: Update retriever with explicit sync and async boundaries**

Modify `app/rag/retriever.py` import and engine call:

```python
import inspect

from app.config import get_settings
from app.rag.vector_store import get_configured_retrieval_engine
```

Add:

```python
def resolve_retrieval_result(result):
    if inspect.isawaitable(result):
        raise RuntimeError("database RAG retrieval is async; use aretrieve_tcm_docs")
    return result


async def resolve_retrieval_result_async(result):
    if inspect.isawaitable(result):
        return await result
    return result
```

Then split the public retrieval helpers:

```python
def _format_payload(query: str, rewritten_query: str, chief_symptom: str | None, result: dict) -> dict:
    results = result["results"]
    retrieval_mode = f"{result['retrieval_mode']}_parent"
    return {
        "status": result["status"],
        "retrieval_mode": retrieval_mode,
        "degraded": result["degraded"],
        "degraded_reason": result["degraded_reason"],
        "original_query": query,
        "rewritten_query": rewritten_query,
        "chief_symptom": chief_symptom,
        "results": results,
        "allowed_terms": collect_allowed_terms(results),
    }


def retrieve_tcm_docs(query: str, k: int = 5, candidate_k: int = 20, mode: str = "hybrid") -> dict:
    del candidate_k
    if mode not in {"hybrid", "vector", "keyword"}:
        mode = "hybrid"
    chief_symptom = detect_chief_symptom(query)
    rewritten_query = rewrite_query(query)
    engine = get_configured_retrieval_engine(get_settings())
    result = resolve_retrieval_result(
        engine.retrieve(
            rewritten_query,
            chief_symptom=chief_symptom,
            mode=mode,
            top_k=min(max(int(k), 1), 5),
        )
    )
    return _format_payload(query, rewritten_query, chief_symptom, result)


async def aretrieve_tcm_docs(query: str, k: int = 5, candidate_k: int = 20, mode: str = "hybrid") -> dict:
    del candidate_k
    if mode not in {"hybrid", "vector", "keyword"}:
        mode = "hybrid"
    chief_symptom = detect_chief_symptom(query)
    rewritten_query = rewrite_query(query)
    engine = get_configured_retrieval_engine(get_settings())
    result = await resolve_retrieval_result_async(
        engine.retrieve(
            rewritten_query,
            chief_symptom=chief_symptom,
            mode=mode,
            top_k=min(max(int(k), 1), 5),
        )
    )
    return _format_payload(query, rewritten_query, chief_symptom, result)
```

- [ ] **Step 5: Make retrieval tool async**

Modify `app/tools/builtins/retrieval_tool.py`:

```python
from langchain.tools import tool

from app.rag.retrieval_log import write_retrieval_log
from app.rag.retriever import aretrieve_tcm_docs, format_retrieval_results


@tool("retrieve_tcm_knowledge")
async def retrieve_tcm_knowledge(query: str, mode: str = "hybrid") -> str:
    if mode not in {"hybrid", "vector", "keyword"}:
        mode = "hybrid"

    payload = await aretrieve_tcm_docs(
        query=query,
        k=5,
        candidate_k=20,
        mode=mode,
    )

    write_retrieval_log(
        {
            "tool": "retrieve_tcm_knowledge",
            "retrieval_mode": payload.get("retrieval_mode"),
            "status": payload.get("status"),
            "degraded": payload.get("degraded"),
            "degraded_reason": payload.get("degraded_reason"),
            "chief_symptom": payload.get("chief_symptom"),
            "original_query": payload.get("original_query"),
            "rewritten_query": payload.get("rewritten_query"),
            "allowed_terms": payload.get("allowed_terms"),
            "final_results": [
                {
                    "citation_id": item.get("citation_id"),
                    "source_type": item.get("source_type"),
                    "book_title": item.get("book_title"),
                    "source_file": item.get("source_file"),
                    "volume": item.get("volume"),
                    "chapter": item.get("chapter"),
                    "section": item.get("section"),
                    "evidence_role": item.get("evidence_role"),
                    "parent_id": item.get("parent_id"),
                    "chunk_id": item.get("chunk_id"),
                    "retrieval_sources": item.get("retrieval_sources"),
                    "dense_rank": item.get("dense_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                }
                for item in payload.get("results", [])
            ],
        }
    )

    return format_retrieval_results(payload)
```

- [ ] **Step 6: Update RAG tool test to use async invocation**

Modify `tests/test_rag_tool.py` by converting `RagToolTests` to `unittest.IsolatedAsyncioTestCase` and replacing:

```python
result = retrieve_tcm_knowledge.invoke(
    {"query": "头痛恶风", "mode": "hybrid"}
)
```

with:

```python
result = await retrieve_tcm_knowledge.ainvoke(
    {"query": "头痛恶风", "mode": "hybrid"}
)
```

- [ ] **Step 7: Run routing and current RAG tool tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_engine_routing tests.test_rag_tool -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

Run:

```powershell
git add app/rag/vector_store.py app/rag/retriever.py app/tools/builtins/retrieval_tool.py tests/test_rag_tool.py tests/rag/database/test_engine_routing.py
git commit -m "feat: route rag engine by configuration"
```

## Task 12: Database CLI

**Files:**
- Create: `app/rag/database/cli.py`
- Test: `tests/rag/database/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/rag/database/test_cli.py`:

```python
import unittest

from app.rag.database.cli import build_parser


class DatabaseCliTests(unittest.TestCase):
    def test_parser_supports_import_doctor_and_smoke(self):
        parser = build_parser()

        import_args = parser.parse_args(["import-artifacts", "--corpus-dir", "c", "--index-dir", "i"])
        doctor_args = parser.parse_args(["doctor"])
        smoke_args = parser.parse_args(["smoke"])

        self.assertEqual(import_args.command, "import-artifacts")
        self.assertEqual(import_args.corpus_dir, "c")
        self.assertEqual(import_args.index_dir, "i")
        self.assertEqual(doctor_args.command, "doctor")
        self.assertEqual(smoke_args.command, "smoke")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_cli -v
```

Expected: import failure for `app.rag.database.cli`.

- [ ] **Step 3: Implement CLI parser and command shell**

Create `app/rag/database/cli.py`:

```python
import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.pool import create_pool_from_settings
from app.rag.database.artifacts import load_artifact_bundle
from app.rag.database.repository import RagPostgresRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCM-Flow database RAG utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-artifacts")
    import_parser.add_argument("--corpus-dir", required=True)
    import_parser.add_argument("--index-dir", required=True)

    subparsers.add_parser("doctor")
    subparsers.add_parser("smoke")
    return parser


async def run_import_artifacts(corpus_dir: str, index_dir: str) -> dict:
    settings = get_settings()
    pool = await create_pool_from_settings(settings)
    try:
        bundle = load_artifact_bundle(Path(corpus_dir), Path(index_dir))
        repository = RagPostgresRepository(pool)
        return await repository.import_bundle(bundle)
    finally:
        await pool.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "import-artifacts":
        result = asyncio.run(run_import_artifacts(args.corpus_dir, args.index_dir))
        print(result)
        return 0
    if args.command == "doctor":
        print({"status": "not_connected"})
        return 0
    if args.command == "smoke":
        print({"status": "not_connected"})
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_cli -v
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

Run:

```powershell
git add app/rag/database/cli.py tests/rag/database/test_cli.py
git commit -m "feat: add database rag cli"
```

## Task 13: Retrieval Logs Backend

**Files:**
- Create: `app/rag/database/retrieval_logs.py`
- Modify: `app/rag/retrieval_log.py`
- Modify: `app/gateway/routers/rag.py`
- Test: `tests/rag/database/test_retrieval_logs.py`

- [ ] **Step 1: Write failing retrieval log tests**

Create `tests/rag/database/test_retrieval_logs.py`:

```python
import unittest
from unittest.mock import patch

from app.rag.retrieval_log import select_log_backend
from app.rag.database.retrieval_logs import build_insert_log_sql, normalize_log_payload


class RetrievalLogBackendTests(unittest.TestCase):
    def test_file_engine_uses_jsonl_backend(self):
        with patch("app.rag.retrieval_log.get_settings") as get_settings:
            get_settings.return_value.rag_engine = "file"
            backend = select_log_backend()

        self.assertEqual(backend, "jsonl")

    def test_database_engine_uses_postgres_backend(self):
        with patch("app.rag.retrieval_log.get_settings") as get_settings:
            get_settings.return_value.rag_engine = "database"
            backend = select_log_backend()

        self.assertEqual(backend, "postgres")

    def test_normalize_log_payload_keeps_required_database_fields(self):
        payload = normalize_log_payload(
            {
                "corpus_id": "ancient-books-v1.0.0",
                "original_query": "头痛恶风",
                "rewritten_query": "头痛恶风 头风",
                "retrieval_mode": "hybrid_parent",
                "degraded": False,
                "final_results": [{"citation_id": "E1"}],
            }
        )

        self.assertEqual(payload["corpus_id"], "ancient-books-v1.0.0")
        self.assertEqual(payload["chief_symptom"], None)
        self.assertEqual(payload["dense_hits"], [])
        self.assertEqual(payload["keyword_hits"], [])
        self.assertEqual(payload["final_results"], [{"citation_id": "E1"}])

    def test_insert_log_sql_targets_rag_retrieval_logs(self):
        sql = build_insert_log_sql().lower()

        self.assertIn("insert into rag_retrieval_logs", sql)
        self.assertIn("original_query", sql)
        self.assertIn("final_results", sql)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_retrieval_logs -v
```

Expected: missing `app.rag.database.retrieval_logs` or `select_log_backend`.

- [ ] **Step 3: Add database retrieval log repository**

Create `app/rag/database/retrieval_logs.py`:

```python
from datetime import datetime


def normalize_log_payload(payload: dict) -> dict:
    return {
        "run_id": payload.get("run_id"),
        "thread_id": payload.get("thread_id"),
        "corpus_id": payload.get("corpus_id", "ancient-books-v1.0.0"),
        "original_query": payload.get("original_query", ""),
        "rewritten_query": payload.get("rewritten_query", ""),
        "chief_symptom": payload.get("chief_symptom"),
        "retrieval_mode": payload.get("retrieval_mode", "unknown"),
        "degraded": bool(payload.get("degraded", False)),
        "degraded_reason": payload.get("degraded_reason"),
        "dense_hits": payload.get("dense_hits") or [],
        "keyword_hits": payload.get("keyword_hits") or [],
        "fused_hits": payload.get("fused_hits") or [],
        "final_results": payload.get("final_results") or [],
    }


def build_insert_log_sql() -> str:
    return """
    insert into rag_retrieval_logs (
      run_id, thread_id, corpus_id, original_query, rewritten_query,
      chief_symptom, retrieval_mode, degraded, degraded_reason,
      dense_hits, keyword_hits, fused_hits, final_results, created_at
    )
    values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
    """


class DatabaseRetrievalLogRepository:
    def __init__(self, pool):
        self.pool = pool

    async def write(self, payload: dict) -> None:
        record = normalize_log_payload(payload)
        async with self.pool.acquire() as connection:
            await connection.execute(
                build_insert_log_sql(),
                record["run_id"],
                record["thread_id"],
                record["corpus_id"],
                record["original_query"],
                record["rewritten_query"],
                record["chief_symptom"],
                record["retrieval_mode"],
                record["degraded"],
                record["degraded_reason"],
                record["dense_hits"],
                record["keyword_hits"],
                record["fused_hits"],
                record["final_results"],
                datetime.utcnow(),
            )
```

- [ ] **Step 4: Add backend selector and keep JSONL compatibility**

Modify `app/rag/retrieval_log.py`:

```python
from app.config import get_settings
```

Add:

```python
def select_log_backend() -> str:
    settings = get_settings()
    return "postgres" if settings.rag_engine == "database" else "jsonl"
```

Keep the existing synchronous `write_retrieval_log()` JSONL behavior as the file-engine compatibility path. Database-engine callers use `DatabaseRetrievalLogRepository.write()` from async runtime code.

- [ ] **Step 5: Run retrieval log and router tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.rag.database.test_retrieval_logs -v
```

Expected: 4 tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add app/rag/database/retrieval_logs.py app/rag/retrieval_log.py tests/rag/database/test_retrieval_logs.py
git commit -m "feat: add database retrieval log backend"
```

## Task 14: Local Service Configuration

**Files:**
- Create: `docker-compose.persistence.yml`
- Create: `.env.example`
- Test: `tests/db/test_local_service_config.py`

- [ ] **Step 1: Write failing config file tests**

Create `tests/db/test_local_service_config.py`:

```python
import unittest
from pathlib import Path


class LocalServiceConfigTests(unittest.TestCase):
    def test_docker_compose_defines_postgres_and_elasticsearch(self):
        text = Path("docker-compose.persistence.yml").read_text(encoding="utf-8")

        self.assertIn("postgres", text)
        self.assertIn("pgvector/pgvector", text)
        self.assertIn("elasticsearch", text)
        self.assertIn("9200:9200", text)

    def test_env_example_contains_persistence_settings(self):
        text = Path(".env.example").read_text(encoding="utf-8")

        self.assertIn("DATABASE_URL=", text)
        self.assertIn("CHECKPOINT_BACKEND=postgres", text)
        self.assertIn("RAG_ENGINE=database", text)
        self.assertIn("ELASTICSEARCH_URL=", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.db.test_local_service_config -v
```

Expected: file-not-found failure.

- [ ] **Step 3: Add compose file**

Create `docker-compose.persistence.yml`:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: tcm
      POSTGRES_PASSWORD: tcm
      POSTGRES_DB: tcm_flow
    ports:
      - "5432:5432"
    volumes:
      - tcm_flow_postgres:/var/lib/postgresql/data

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.15.0
    environment:
      discovery.type: single-node
      xpack.security.enabled: "false"
      ES_JAVA_OPTS: "-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
    volumes:
      - tcm_flow_elasticsearch:/usr/share/elasticsearch/data

volumes:
  tcm_flow_postgres:
  tcm_flow_elasticsearch:
```

- [ ] **Step 4: Add env example**

Create `.env.example`:

```env
OPENAI_BASE_URL=
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

DATABASE_URL=postgresql://tcm:tcm@localhost:5432/tcm_flow
POSTGRES_POOL_SIZE=10
CHECKPOINT_BACKEND=postgres

RAG_ENGINE=database
RAG_FALLBACK_FILE_ENGINE=true

ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_RAG_INDEX_ALIAS=tcm_rag_chunks_current
ELASTICSEARCH_ANALYZER=standard
```

- [ ] **Step 5: Run config tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.db.test_local_service_config -v
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add docker-compose.persistence.yml .env.example tests/db/test_local_service_config.py
git commit -m "chore: add local persistence services"
```

## Task 15: Final Verification

**Files:**
- No new files expected.

- [ ] **Step 1: Run focused unit suites**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config tests.db.test_schema tests.db.test_pool tests.store.test_postgres_store tests.test_runtime_state tests.checkpoints.test_factory tests.rag.database.test_artifacts tests.rag.database.test_repository tests.rag.database.test_elasticsearch_index tests.rag.database.test_engine tests.rag.database.test_engine_routing tests.rag.database.test_cli tests.rag.database.test_retrieval_logs tests.db.test_local_service_config -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run existing runtime and RAG tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_clarification_flow tests.test_rag_tool tests.rag.ancient_books.test_runtime tests.rag.ancient_books.test_cli -v
```

Expected: existing clarification and ancient-books tests pass.

- [ ] **Step 3: Run full unittest discovery**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Expected: all discovered tests pass. If experiment-only tests require unavailable local artifacts, record the exact failing test names and run the focused production suite from Steps 1-2 as the completion gate.

- [ ] **Step 4: Run static whitespace check**

Run:

```powershell
git diff --check
```

Expected: no output.

- [ ] **Step 5: Commit verification-only fixes if any**

If Steps 1-4 require small fixes, commit the exact files changed by those fixes:

```powershell
git status --short
git add app tests requirements.txt docker-compose.persistence.yml .env.example
git commit -m "fix: stabilize persistence integration tests"
```

If no files changed, do not create a commit.

## Self-Review

Spec coverage:

- Runtime persistence: Tasks 1-6 and 13 cover settings, schema, stores, state, checkpointer, and logs.
- RAG artifact import: Tasks 7-8 and 12 cover artifact loading, repository import, and CLI entry.
- Dense search: Tasks 8 and 10 cover pgvector serialization, SQL shape, and engine use.
- Keyword search: Tasks 9 and 10 cover ES documents, query filters, and degradation.
- Engine routing and fallback: Task 11 covers file/database selection and the sync/async retrieval boundary.
- Retrieval logs: Task 13 covers Postgres retrieval-log normalization and insert SQL while preserving JSONL compatibility for file-engine mode.
- Local deployment: Task 14 covers Postgres pgvector and Elasticsearch services.
- Verification: Task 15 covers focused and broader regression commands.

Placeholder scan:

- The plan contains no placeholder markers or unspecified table names.
- All approved scheme C storage responsibilities have at least one implementation task.

Type consistency:

- `AppSettings` fields match every test and call site.
- `RagArtifactBundle` fields are used consistently by `RagPostgresRepository`.
- `DatabaseRetrievalEngine.retrieve()` mirrors the existing file engine shape: `status`, `retrieval_mode`, `degraded`, `degraded_reason`, and `results`.
