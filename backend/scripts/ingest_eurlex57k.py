"""
Seeds the EURLEX57K-derived regulatory reference collection in Qdrant
from the REAL EURLEX57K dataset (train.jsonl / dev.jsonl / test.jsonl, as
downloaded from http://archive.org/download/EURLEX57K/dataset.zip --
see http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/), replacing
the 8-passage illustrative sample ingest_eurlex_sample.py seeds.

Each line of the dataset's jsonl files is one legislative document with
(at minimum) these fields: celex_id, title, header, recitals, main_body
(a LIST of per-article paragraph strings -- EU legislative main bodies
are "usually organized in articles", and the dataset ships them already
split that way) and eurovoc_concepts. Some distributions also include an
"attachments" field (annexes/schedules). A record with none of
header/recitals/main_body but a flat "text" field (e.g. an
already-flattened export) is also accepted as a fallback -- see
extract_sections().

Per the project's documented preprocessing requirement ("preprocessed
through the same pipeline as user documents, and indexed" -- see
README's Known Limitations), each section's text is run through the
SAME segment_clauses() used for uploaded contracts, so the reference
corpus and the documents being audited live at a comparable passage
granularity. A whole multi-thousand-word EU regulation embedded as a
single vector would almost never score above compliance.py's
_RELEVANCE_MIN_SCORE against a short contract clause, no matter how
relevant its content actually is -- granularity mismatch, not
irrelevance, would be driving the miss.

Indexing the full 45k-document train split (header + recitals + ~10
articles each, further sub-split wherever an article has internal
(a)/(b)/numbered structure) is impractical for a project-stage CPU
embedding run, and most of it -- tariff quotas, duty-free scientific
apparatus exemptions, EAGGF monthly accounting entries, the kind of
thing that makes up a lot of the real corpus -- has nothing to do with
the kind of commercial contract clauses this tool audits
(confidentiality, data protection, liability, termination, payment
terms). So this script takes a reproducible random SAMPLE of documents
(seeded, so reruns with the same --seed are comparable) rather than
indexing an entire split. Scale --max-documents up if you have the
time/compute and want broader coverage; compliance.py's relevance gate
is what keeps a larger, noisier, mostly-irrelevant corpus from
corrupting individual judgments either way -- it just skips whatever
doesn't score as relevant for a given clause, so there's no correctness
reason to pre-filter by legal topic here, only a time/compute one.

Usage:
    cd backend
    python -m scripts.ingest_eurlex57k --data-dir /path/to/EURLEX57K

    # e.g. given the "NLP Project/Dataset/data/datasets/EURLEX57K/" and
    # "NLP Project/Code Folder/v1/scaffold/backend/" layout described in
    # the project README, run from inside backend/:
    python -m scripts.ingest_eurlex57k \\
        --data-dir "../../../../Dataset/data/datasets/EURLEX57K"

    # broader, slower run across all three splits, no per-split cap:
    python -m scripts.ingest_eurlex57k --data-dir ... --splits all --max-documents 0
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.core.db import get_qdrant_client
from app.pipeline.embedding import embed_texts, ensure_collection
from app.pipeline.segmentation import segment_clauses

_SPLIT_FILENAMES = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}

# Passages shorter than this are dropped as near-empty noise (a bare
# "Article 1" heading with no body, an empty attachments field, etc.) --
# same spirit as segmentation.py's own short-fragment filter.
_DEFAULT_MIN_CHARS = 40

# How many passages to embed per embed_texts() / Qdrant upsert call.
# Keeps memory bounded and gives visible progress on large runs, rather
# than one huge blocking call over the whole sampled set.
_EMBED_BATCH_SIZE = 256


def extract_sections(doc: dict) -> list[tuple[str, str]]:
    """
    Pure, I/O-free: pulls the major zones out of one EURLEX57K record as
    (section_label, section_text) pairs, mirroring the dataset paper's
    own structure (header / recitals / each article of the main body /
    attachments). Falls back to a flat "text" field for records that
    don't have the raw header/recitals/main_body shape (e.g. an
    already-flattened export). Returns [] for a record with none of
    these fields populated -- the caller skips it.
    """
    sections: list[tuple[str, str]] = []

    header = doc.get("header")
    if isinstance(header, str) and header.strip():
        sections.append(("header", header.strip()))

    recitals = doc.get("recitals")
    if isinstance(recitals, str) and recitals.strip():
        sections.append(("recitals", recitals.strip()))

    main_body = doc.get("main_body")
    if isinstance(main_body, str):
        main_body = [main_body]
    if isinstance(main_body, list):
        for i, article_text in enumerate(main_body):
            if isinstance(article_text, str) and article_text.strip():
                sections.append((f"article_{i + 1}", article_text.strip()))

    attachments = doc.get("attachments")
    if isinstance(attachments, list):
        attachments = " ".join(a for a in attachments if isinstance(a, str))
    if isinstance(attachments, str) and attachments.strip():
        sections.append(("attachments", attachments.strip()))

    if not sections:
        text = doc.get("text")
        if isinstance(text, str) and text.strip():
            sections.append(("text", text.strip()))

    return sections


def passages_for_document(doc: dict, min_chars: int = _DEFAULT_MIN_CHARS) -> list[tuple[str, str]]:
    """
    Pure: runs each section of one document through the project's own
    segment_clauses() (the same clause-boundary logic used on uploaded
    contracts, see segmentation.py) and returns (passage_id, text) pairs
    long enough to be useful reference material. passage_id is stable
    across reruns given the same input (celex_id + section + position),
    so re-ingesting is idempotent at the Qdrant point level.
    """
    celex_id = doc.get("celex_id") or "unknown"
    passages: list[tuple[str, str]] = []
    for section_label, section_text in extract_sections(doc):
        for clause in segment_clauses(section_text):
            text = clause.text.strip()
            if len(text) < min_chars:
                continue
            passages.append((f"{celex_id}:{section_label}_{clause.order}", text))
    return passages


def _point_id(passage_id: str) -> str:
    """Deterministic UUID5 from the passage id -- same idempotent-upsert
    idiom as embedding.py's _point_id for user documents."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, passage_id))


def iter_sampled_documents(path: Path, max_documents: int, seed: int):
    """
    Reads every non-blank line of a jsonl split, takes a reproducible
    random sample of up to max_documents of them (max_documents <= 0
    means "no cap, use the whole split"), and yields the parsed JSON
    objects. Malformed lines are skipped rather than aborting the run.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    if max_documents and 0 < max_documents < len(lines):
        lines = random.Random(seed).sample(lines, max_documents)

    for line in lines:
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the regulatory reference collection from the real EURLEX57K dataset."
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Folder containing train.jsonl / dev.jsonl / test.jsonl",
    )
    parser.add_argument(
        "--splits", default="train",
        help="Comma-separated split names (train,dev,test) or 'all' (default: train)",
    )
    parser.add_argument(
        "--max-documents", type=int, default=500,
        help="Random sample size PER SPLIT; 0 = use the whole split, no cap (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Sampling seed, for a reproducible subset across reruns (default: 42)",
    )
    parser.add_argument(
        "--min-chars", type=int, default=_DEFAULT_MIN_CHARS,
        help=f"Drop passages shorter than this many characters (default: {_DEFAULT_MIN_CHARS})",
    )
    parser.add_argument(
        "--keep-existing", action="store_true",
        help="Upsert on top of whatever's already in the collection instead of replacing it. "
             "Default is to delete + recreate the collection first, so the 8-passage "
             "illustrative sample (from ingest_eurlex_sample.py) doesn't linger alongside "
             "the real corpus and skew retrieval.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    splits = list(_SPLIT_FILENAMES) if args.splits == "all" else [s.strip() for s in args.splits.split(",")]
    for s in splits:
        if s not in _SPLIT_FILENAMES:
            parser.error(f"Unknown split '{s}' -- expected one of {list(_SPLIT_FILENAMES)} or 'all'")

    for s in splits:
        path = data_dir / _SPLIT_FILENAMES[s]
        if not path.exists():
            parser.error(f"Expected {path} to exist -- check --data-dir")

    client = get_qdrant_client()
    if not args.keep_existing:
        try:
            client.delete_collection(collection_name=settings.eurlex_qdrant_collection)
        except Exception:
            pass  # fine if the collection didn't exist yet
    ensure_collection(client, settings.eurlex_qdrant_collection)

    total_docs = 0
    total_points = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def flush() -> None:
        nonlocal total_points
        if not batch_texts:
            return
        vectors = embed_texts(batch_texts)
        points = [
            qmodels.PointStruct(
                id=_point_id(pid),
                vector=vector,
                # No document_id -- this collection is document-agnostic
                # reference material (see compliance.py).
                payload={"clause_id": pid, "text": text, "label": pid},
            )
            for pid, text, vector in zip(batch_ids, batch_texts, vectors)
        ]
        client.upsert(collection_name=settings.eurlex_qdrant_collection, points=points)
        total_points += len(points)
        print(f"  ...indexed {total_points} passages so far")
        batch_ids.clear()
        batch_texts.clear()

    for split in splits:
        path = data_dir / _SPLIT_FILENAMES[split]
        print(f"Sampling '{split}' from {path} "
              f"(max_documents={args.max_documents or 'all'}, seed={args.seed})...")
        split_docs = 0
        for doc in iter_sampled_documents(path, args.max_documents, args.seed):
            split_docs += 1
            total_docs += 1
            for pid, text in passages_for_document(doc, args.min_chars):
                batch_ids.append(pid)
                batch_texts.append(text)
                if len(batch_texts) >= _EMBED_BATCH_SIZE:
                    flush()
        print(f"  sampled {split_docs} documents from '{split}'")

    flush()
    print(
        f"Done: indexed {total_points} regulatory reference passages from "
        f"{total_docs} EURLEX57K documents into '{settings.eurlex_qdrant_collection}'."
    )


if __name__ == "__main__":
    main()
