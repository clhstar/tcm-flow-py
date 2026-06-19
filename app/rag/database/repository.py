import json
from datetime import datetime, timezone
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
        now = datetime.now(timezone.utc)
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    insert into rag_corpora (
                      corpus_id, version, status, source_manifest_sha256,
                      index_manifest_sha256, embedding_model, reranker_model,
                      vector_dimension, parent_count, chunk_count, created_at
                    )
                    values ($1,$2,$3,$4,$5,$6::jsonb,$7::jsonb,$8,$9,$10,$11)
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
                    json.dumps(
                        bundle.index_manifest.get("embedding_model", {}),
                        ensure_ascii=False,
                    ),
                    None,
                    1024,
                    len(bundle.parents),
                    len(bundle.chunks),
                    now,
                )
                for parent in bundle.parents:
                    source_id = (
                        f"{bundle.corpus_id}:{parent['book_id']}:"
                        f"{parent['source_hash'][:12]}"
                    )
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

    async def dense_search(
        self,
        corpus_id: str,
        vector: np.ndarray,
        chief_symptom: str | None,
        top_k: int,
    ):
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
