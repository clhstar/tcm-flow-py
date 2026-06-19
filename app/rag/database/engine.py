from app.rag.ancient_books.runtime import reciprocal_rank_fusion
from app.rag.database.repository import DEFAULT_EVIDENCE_ROLES


class DatabaseRetrievalEngine:
    def __init__(
        self,
        *,
        corpus_id: str,
        repository,
        keyword_index,
        encoder,
        reranker,
        settings: dict,
    ):
        self.corpus_id = corpus_id
        self.repository = repository
        self.keyword_index = keyword_index
        self.encoder = encoder
        self.reranker = reranker
        self.settings = settings

    async def retrieve(
        self,
        query: str,
        *,
        chief_symptom: str | None,
        mode: str = "hybrid",
        top_k: int = 5,
    ) -> dict:
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
                current = hit_by_chunk.setdefault(
                    hit["chunk_id"],
                    {**hit, "retrieval_sources": []},
                )
                current["retrieval_sources"].append(source)
                current[f"{source}_rank"] = rank

        candidate_ids = [
            chunk_id
            for chunk_id, _score in fused[: int(self.settings["reranker_candidate_k"])]
        ]
        pairs = [
            [query, hit_by_chunk[chunk_id]["matched_child"]]
            for chunk_id in candidate_ids
        ]
        scores = self.reranker.score(pairs) if pairs else []
        ranked = sorted(
            zip(candidate_ids, scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        parent_ids = [hit_by_chunk[chunk_id]["parent_id"] for chunk_id, _ in ranked]
        parents = await self.repository.load_parents(parent_ids)

        results = []
        seen_parent_ids = set()
        limit = min(int(top_k), int(self.settings["final_top_k"]), 5)
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
            if len(results) >= limit:
                break

        return {
            "status": "ok" if results else "insufficient_evidence",
            "retrieval_mode": actual_mode,
            "degraded": degraded,
            "degraded_reason": degraded_reason,
            "results": results,
        }
