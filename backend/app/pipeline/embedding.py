"""
Stage 2: Embeddings + Dense Passage Retrieval.

Generates dense vector embeddings for clauses and stores them in Qdrant.
This module is deliberately split into three concerns that other stages
call independently:
  - embed_texts: pure text -> vectors (no I/O), easy to unit test / swap models
  - ensure_collection / index_clauses: writing into Qdrant
  - search_similar: reading from Qdrant

Both the contract document clauses AND the EURLEX57K regulatory reference
corpus go through embed_texts with the same model, so they live in the
same vector space and are directly comparable at query time (see
scripts/ingest_eurlex_sample.py for how the reference corpus gets indexed).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.pipeline.segmentation import Clause


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Loaded once per process. all-MiniLM-L6-v2 (384-dim, CPU-friendly) per
    the project's stack choice -- cached so repeated calls across requests
    don't reload the model from disk each time.
    """
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()


def embedding_dimension() -> int:
    return get_embedding_model().get_sentence_embedding_dimension()


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """Creates the Qdrant collection if it doesn't already exist. Idempotent."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=embedding_dimension(),
            distance=qmodels.Distance.COSINE,
        ),
    )


def _point_id(document_id: str, clause_id: str) -> str:
    """
    Qdrant point IDs must be unsigned ints or UUIDs, so we can't use
    "clause_0007" directly. Deterministic UUID5 keeps re-indexing the
    same document idempotent (same clause -> same point ID -> upsert
    overwrites rather than duplicates).
    """
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{clause_id}"))


def index_clauses(
    client: QdrantClient,
    collection_name: str,
    document_id: str,
    clauses: list[Clause],
) -> int:
    """
    Embeds and upserts a document's clauses into Qdrant. Returns the
    number of points written. Payload carries everything GraphRAG and
    reporting need without a second lookup.
    """
    if not clauses:
        return 0

    ensure_collection(client, collection_name)
    vectors = embed_texts([c.text for c in clauses])

    points = [
        qmodels.PointStruct(
            id=_point_id(document_id, clause.id),
            vector=vector,
            payload={
                "document_id": document_id,
                "clause_id": clause.id,
                "label": clause.label,
                "order": clause.order,
                "text": clause.text,
            },
        )
        for clause, vector in zip(clauses, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


@dataclass
class SearchHit:
    clause_id: str
    document_id: str
    text: str
    label: str | None
    score: float


def search_similar(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int = 10,
    document_id: str | None = None,
    exclude_clause_id: str | None = None,
) -> list[SearchHit]:
    """
    Dense similarity search. If document_id is given, results are scoped
    to that document (used for within-document contradiction search); if
    omitted, search runs across the whole collection (used for compliance
    lookups against the EURLEX57K reference collection).
    """
    query_filter = None
    if document_id is not None:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
        )

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k + (1 if exclude_clause_id else 0),
    )

    hits = []
    for r in results:
        if exclude_clause_id and r.payload.get("clause_id") == exclude_clause_id:
            continue
        hits.append(
            SearchHit(
                clause_id=r.payload["clause_id"],
                document_id=r.payload.get("document_id", ""),
                text=r.payload["text"],
                label=r.payload.get("label"),
                score=r.score,
            )
        )
    return hits[:top_k]
