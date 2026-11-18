"""
One-off script to seed the EURLEX57K-derived regulatory reference
collection in Qdrant with a small sample corpus, so compliance.py has
something to retrieve against during development without needing to
download and preprocess the full EURLEX57K dataset first.

This sample corpus is a handful of illustrative regulatory-style
passages for pipeline testing -- NOT a substitute for the real
EURLEX57K dataset described in the project's dataset section. Swap this
script's data source for the actual preprocessed EURLEX57K corpus
(run through the same text-normalization step as user documents, per
the "Dataset Description" note that all four datasets share
preprocessing) before using compliance results for anything but
pipeline smoke-testing.

Usage:
    cd backend
    python -m scripts.ingest_eurlex_sample
"""
from __future__ import annotations

import json
from pathlib import Path

from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.core.db import get_qdrant_client
from app.pipeline.embedding import embed_texts, ensure_collection

SAMPLE_PATH = Path(__file__).parent.parent / "sample_data" / "eurlex_sample.jsonl"


def load_sample_passages() -> list[dict]:
    passages = []
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    return passages


def main() -> None:
    client = get_qdrant_client()
    ensure_collection(client, settings.eurlex_qdrant_collection)

    passages = load_sample_passages()
    vectors = embed_texts([p["text"] for p in passages])

    points = [
        qmodels.PointStruct(
            id=i,
            vector=vector,
            # No document_id -- this collection is document-agnostic reference
            # material, searched without a document filter (see compliance.py).
            payload={"clause_id": p["id"], "text": p["text"], "label": p["id"]},
        )
        for i, (p, vector) in enumerate(zip(passages, vectors))
    ]
    client.upsert(collection_name=settings.eurlex_qdrant_collection, points=points)
    print(f"Indexed {len(points)} sample regulatory reference passages into "
          f"'{settings.eurlex_qdrant_collection}'.")


if __name__ == "__main__":
    main()
