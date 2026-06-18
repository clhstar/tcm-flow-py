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
