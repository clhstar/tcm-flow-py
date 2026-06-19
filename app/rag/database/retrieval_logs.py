import json
from datetime import datetime, timezone


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
    values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::jsonb,$12::jsonb,$13::jsonb,$14)
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
                json.dumps(record["dense_hits"], ensure_ascii=False),
                json.dumps(record["keyword_hits"], ensure_ascii=False),
                json.dumps(record["fused_hits"], ensure_ascii=False),
                json.dumps(record["final_results"], ensure_ascii=False),
                datetime.now(timezone.utc),
            )
