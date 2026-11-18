"""
Stage 7b: Compliance Assessment.

For each clause, retrieves the nearest passages from the EURLEX57K-
derived regulatory reference corpus (via the same dense retrieval used
for contradiction detection, just against a different Qdrant collection
with no document_id filter) and asks the LLM to judge whether the clause
is compliant with or falls short of that grounding passage -- the
retrieved text is passed into the prompt so the judgment is grounded
rather than relying on the model's own (unreliable, and out of scope for
a prompted baseline) legal knowledge.

Relevance gate: the top retrieval hit is always *something* -- Qdrant
returns the closest vector in the collection even when nothing in a small
reference corpus actually bears on the clause's subject matter. Originally
that irrelevant top hit was handed to the LLM anyway with an instruction
to self-report irrelevance as "unclear", but in practice the model would
often reason over the mismatched text literally and call it a "gap"
instead -- inflating compliance_gap_count with clauses that were never
checked against anything applicable. _RELEVANCE_MIN_SCORE makes that a
deterministic, pre-LLM decision instead of asking the model to judge its
own retrieval's relevance: below the threshold, the clause is marked
"not_applicable" and the LLM is never called for it at all (mirrors
contradiction.py's _RERANK_MIN_SCORE gate, and the same pattern: a
similarity threshold decides "unrelated", not the model). "unclear"
remains a possible LLM-returned status, but now only for cases where a
genuinely relevant reference was retrieved and the model still couldn't
make a clean compliant/gap call from it.

Mirrors deontic.py's structure: a pure prompt-builder, a pure response
parser, and a thin function that calls the LLM client -- kept consistent
so both LLM-touching stages are equally easy to test and to swap models on.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.core.config import settings
from app.pipeline.embedding import SearchHit, embed_texts, search_similar
from app.pipeline.llm_client import LLMClient

# Below this cosine similarity, the "best" retrieved reference passage is
# treated as unrelated to the clause, not as a weak match -- same role as
# contradiction.py's _RERANK_MIN_SCORE. Chosen conservatively: with the
# small illustrative EURLEX sample corpus, genuinely on-topic matches
# score well above this, while off-topic nearest-neighbors (an unrelated
# passage returned only because *something* has to be closest) typically
# fall below it. Revisit once the real EURLEX57K corpus is indexed and
# this can be tuned against labeled examples instead of by inspection.
_RELEVANCE_MIN_SCORE = 0.35


@dataclass
class ComplianceResult:
    clause_id: str
    # "compliant" | "gap" | "unclear" come from the LLM's own judgment on a
    # reference passage that cleared the relevance gate. "not_applicable"
    # is assigned by this module directly, before any LLM call, when no
    # reference passage was retrieved or the best one scored below
    # _RELEVANCE_MIN_SCORE -- i.e. no part of the reference corpus speaks
    # to this clause at all.
    status: str  # "compliant" | "gap" | "unclear" | "not_applicable"
    matched_reference: str | None
    rationale: str


_PROMPT_TEMPLATE = """You are a compliance analyst. Judge whether the CONTRACT CLAUSE below is
consistent with the REGULATORY REFERENCE passage, or whether it falls short of / conflicts
with what the reference requires. The reference passage has already been confirmed relevant
to this clause's subject matter, so assume it applies -- judge compliance, don't re-decide
relevance.

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"status": "<one of compliant|gap|unclear>", "rationale": "<one sentence>"}}

Use "unclear" only if you genuinely cannot tell whether the clause satisfies the reference
(e.g. the clause is ambiguously worded), not because the reference seems off-topic.

REGULATORY REFERENCE:
\"\"\"{reference_text}\"\"\"

CONTRACT CLAUSE:
\"\"\"{clause_text}\"\"\"
"""


def build_prompt(clause_text: str, reference_text: str) -> str:
    return _PROMPT_TEMPLATE.format(clause_text=clause_text.strip(), reference_text=reference_text.strip())


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_VALID_STATUSES = {"compliant", "gap", "unclear"}


def parse_llm_response(raw_text: str) -> dict | None:
    """Same defensive extraction strategy as deontic.parse_llm_response --
    duplicated rather than shared because the two prompts' expected keys
    differ and a shared helper would need to know both, which couples
    two independently-evolving stages for no real benefit."""
    if not raw_text:
        return None
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    match = _JSON_OBJECT_PATTERN.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or parsed.get("status") not in _VALID_STATUSES:
        return None
    return parsed


def retrieve_reference_passages(
    qdrant_client: QdrantClient,
    clause_text: str,
    top_k: int = 3,
) -> list[SearchHit]:
    query_vector = embed_texts([clause_text])[0]
    return search_similar(
        client=qdrant_client,
        collection_name=settings.eurlex_qdrant_collection,
        query_vector=query_vector,
        top_k=top_k,
        document_id=None,
    )


def assess_clause_compliance(
    clause_id: str,
    clause_text: str,
    reference_passages: list[SearchHit],
    llm_client: LLMClient | None = None,
) -> ComplianceResult:
    """
    Judges a single clause against its single best-matching reference
    passage. Returns "not_applicable" without calling the LLM when there's
    nothing relevant to ground a judgment in -- either no reference
    passages were retrieved at all (empty regulatory corpus), or the best
    one scored below _RELEVANCE_MIN_SCORE (see module docstring: relying
    on the LLM to self-report an irrelevant match as "unclear" rather than
    "gap" proved unreliable, so relevance is decided here instead).
    """
    if not reference_passages:
        return ComplianceResult(
            clause_id=clause_id,
            status="not_applicable",
            matched_reference=None,
            rationale="No regulatory reference passage was found for this clause.",
        )

    best_match = reference_passages[0]

    if best_match.score < _RELEVANCE_MIN_SCORE:
        return ComplianceResult(
            clause_id=clause_id,
            status="not_applicable",
            matched_reference=best_match.text,
            rationale=(
                f"The closest regulatory reference (similarity {best_match.score:.2f}) "
                f"falls below the relevance threshold ({_RELEVANCE_MIN_SCORE}) -- nothing "
                "in the reference corpus speaks to this clause's subject matter."
            ),
        )

    client = llm_client or LLMClient()

    raw_text = client.chat(build_prompt(clause_text, best_match.text))
    parsed = parse_llm_response(raw_text)

    if parsed is None:
        return ComplianceResult(
            clause_id=clause_id,
            status="unclear",
            matched_reference=best_match.text,
            rationale="Compliance judgment could not be parsed from the model response.",
        )

    return ComplianceResult(
        clause_id=clause_id,
        status=parsed["status"],
        matched_reference=best_match.text,
        rationale=parsed.get("rationale", ""),
    )


def assess_document_compliance(
    qdrant_client: QdrantClient,
    clause_texts_by_id: dict[str, str],
    llm_client: LLMClient | None = None,
) -> list[ComplianceResult]:
    """
    Retrieval (Qdrant) always happens per-clause in the main thread; only
    the LLM judgment call is candidate for concurrency, controlled by the
    same settings.llm_max_workers used in deontic.py (1 = sequential,
    right for Ollama; raise it for Groq).
    """
    client = llm_client or LLMClient()
    workers = max(1, settings.llm_max_workers)

    def _assess(item: tuple[str, str]) -> ComplianceResult:
        clause_id, clause_text = item
        references = retrieve_reference_passages(qdrant_client, clause_text)
        return assess_clause_compliance(clause_id, clause_text, references, client)

    items = list(clause_texts_by_id.items())
    if workers == 1:
        return [_assess(item) for item in items]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_assess, items))
