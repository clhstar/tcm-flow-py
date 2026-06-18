from typing import Any


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
                            "fields": [
                                "text^3",
                                "chapter^2",
                                "section^2",
                                "book_title",
                            ],
                        }
                    }
                ],
                "filter": filters,
            }
        },
    }


class ElasticsearchKeywordIndex:
    def __init__(self, client: Any, alias: str):
        self.client = client
        self.alias = alias

    async def search(
        self,
        *,
        rewritten_query: str,
        corpus_id: str,
        chief_symptom: str | None,
        evidence_roles: list[str],
        top_k: int,
    ):
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
