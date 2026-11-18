"""
Stage 5: GraphRAG Retrieval.

Combines the two retrieval paths built in stages 2 and 4: dense
similarity search (semantically similar clauses, possibly worded very
differently) and graph traversal (structurally connected clauses --
same entity, or a linked penalty). Kept as two separate, independently
callable functions (dense_candidates / graph_candidates) before merging,
per the build-order note: easier to debug which path is contributing
what before combining them.

The merged candidate set is what gets passed to reranker.py -- this
stage intentionally over-retrieves (does not do the precision filtering
itself) since cross-encoder re-ranking is a separate, more expensive
step that shouldn't run over everything, only over this narrowed set.
"""
from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver
from qdrant_client import QdrantClient

from app.pipeline.embedding import SearchHit, search_similar
from app.pipeline.graph_builder import get_related_clauses


@dataclass
class Candidate:
    clause_id: str
    text: str
    label: str | None
    sources: set[str]  # {"dense"}, {"graph"}, or {"dense", "graph"}
    dense_score: float | None = None


def dense_candidates(
    qdrant_client: QdrantClient,
    collection_name: str,
    document_id: str,
    query_vector: list[float],
    exclude_clause_id: str,
    top_k: int = 10,
) -> list[SearchHit]:
    return search_similar(
        client=qdrant_client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k,
        document_id=document_id,
        exclude_clause_id=exclude_clause_id,
    )


def graph_candidates(
    neo4j_driver: Driver,
    document_id: str,
    clause_id: str,
    hops: int = 1,
) -> list[dict]:
    return get_related_clauses(neo4j_driver, document_id, clause_id, hops=hops)


def retrieve(
    qdrant_client: QdrantClient,
    neo4j_driver: Driver,
    collection_name: str,
    document_id: str,
    query_clause_id: str,
    query_vector: list[float],
    dense_top_k: int = 10,
    graph_hops: int = 1,
) -> list[Candidate]:
    """
    Runs both retrieval paths for a given query clause and merges them
    into a deduplicated candidate list, tagging each candidate with
    which path(s) surfaced it. A clause found by both paths is a
    stronger signal (semantically similar AND structurally connected)
    and reranker.py / contradiction.py can use the `sources` field to
    weight accordingly.
    """
    merged: dict[str, Candidate] = {}

    for hit in dense_candidates(
        qdrant_client, collection_name, document_id, query_vector, query_clause_id, dense_top_k
    ):
        merged[hit.clause_id] = Candidate(
            clause_id=hit.clause_id,
            text=hit.text,
            label=hit.label,
            sources={"dense"},
            dense_score=hit.score,
        )

    for related in graph_candidates(neo4j_driver, document_id, query_clause_id, graph_hops):
        cid = related["clause_id"]
        if cid in merged:
            merged[cid].sources.add("graph")
        else:
            merged[cid] = Candidate(
                clause_id=cid,
                text=related["text"],
                label=related.get("label"),
                sources={"graph"},
            )

    return list(merged.values())
