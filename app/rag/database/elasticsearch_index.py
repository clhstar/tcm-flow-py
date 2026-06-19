from typing import Any


def build_index_name(version: str) -> str:
    return "tcm_rag_chunks_" + version.replace(".", "_").replace("-", "_")


def build_index_body(analyzer: str = "standard") -> dict:
    return {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "corpus_id": {"type": "keyword"},
                "book_id": {"type": "keyword"},
                "book_title": {"type": "text", "analyzer": analyzer},
                "source_file": {"type": "keyword"},
                "source_hash": {"type": "keyword"},
                "volume": {"type": "text", "analyzer": analyzer},
                "chapter": {"type": "text", "analyzer": analyzer},
                "section": {"type": "text", "analyzer": analyzer},
                "text": {"type": "text", "analyzer": analyzer},
                "symptom_tags": {"type": "keyword"},
                "evidence_role": {"type": "keyword"},
                "row_index": {"type": "integer"},
                "index_version": {"type": "keyword"},
            }
        }
    }


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


def _chunk_for_document(chunk: dict, corpus_id: str, row_index: int) -> dict:
    document_chunk = dict(chunk)
    document_chunk["corpus_id"] = corpus_id
    document_chunk["row_index"] = row_index
    return document_chunk


async def _bulk_index(client: Any, operations: list[dict]) -> None:
    if not operations:
        return
    response = await client.bulk(operations=operations, refresh=True)
    if response.get("errors"):
        failed = [
            item
            for item in response.get("items", [])
            if item.get("index", {}).get("error")
        ]
        raise RuntimeError(f"Elasticsearch bulk index failed: {failed[:3]}")


async def rebuild_keyword_index(
    client: Any,
    bundle,
    *,
    alias: str,
    analyzer: str = "standard",
    batch_size: int = 500,
) -> dict:
    version = bundle.index_manifest.get("version") or bundle.corpus_manifest.get(
        "version",
        "v1.0.0",
    )
    index_name = build_index_name(version)
    parents_by_id = {parent["parent_id"]: parent for parent in bundle.parents}

    await client.indices.delete(index=index_name, ignore_unavailable=True)
    await client.indices.create(index=index_name, body=build_index_body(analyzer))

    operations: list[dict] = []
    indexed_count = 0
    for row_index, chunk in enumerate(bundle.chunks):
        parent = parents_by_id[chunk["parent_id"]]
        document = build_chunk_document(
            _chunk_for_document(chunk, bundle.corpus_id, row_index),
            parent,
            version,
        )
        operations.append({"index": {"_index": index_name, "_id": chunk["chunk_id"]}})
        operations.append(document)
        indexed_count += 1
        if indexed_count % batch_size == 0:
            await _bulk_index(client, operations)
            operations = []
    await _bulk_index(client, operations)

    await client.indices.update_aliases(
        body={
            "actions": [
                {
                    "remove": {
                        "index": "*",
                        "alias": alias,
                        "must_exist": False,
                    }
                },
                {
                    "add": {
                        "index": index_name,
                        "alias": alias,
                        "is_write_index": True,
                    }
                },
            ]
        }
    )
    return {
        "index": index_name,
        "alias": alias,
        "document_count": indexed_count,
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
