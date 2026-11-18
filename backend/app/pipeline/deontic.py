"""
Stage 3: Deontic Logic Classification.

Classifies each clause into one of four categories -- Obligation,
Permission, Prohibition, Penalty -- and identifies which entity the
clause binds. This is the module that was restructured in response to
faculty feedback (see the PDF's "Modifications Made Based on Abstract
Feedback" section): output feeds directly into the knowledge graph in
graph_builder.py.

Baseline approach: prompt a small local LLM via Ollama rather than
training a classifier first. This keeps the pipeline runnable with zero
labeled data on day one; LEDGAR-derived fine-tuning is the documented
upgrade path once the prompted baseline's error patterns are understood
(see README "Build order").

The prompt-response parsing (parse_llm_response) is kept as a pure
function deliberately -- it's the part most likely to break in practice
(local models wrap JSON in prose, use markdown fences, etc.) and it's
fully unit-testable without a running Ollama instance.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from app.core.config import settings
from app.pipeline.llm_client import LLMClient
from app.pipeline.segmentation import Clause


class DeonticCategory(str, Enum):
    OBLIGATION = "Obligation"
    PERMISSION = "Permission"
    PROHIBITION = "Prohibition"
    PENALTY = "Penalty"
    UNKNOWN = "Unknown"  # classifier couldn't confidently assign one of the four


@dataclass
class DeonticLabel:
    clause_id: str
    category: DeonticCategory
    entity: str | None       # who/what the clause binds, e.g. "the Contractor"
    confidence: str          # "high" | "medium" | "low" -- qualitative, from the prompt
    raw_response: str        # kept for debugging misclassifications


_PROMPT_TEMPLATE = """You are a legal analyst classifying a single contract clause.

Classify the clause into EXACTLY ONE of these four categories:
- Obligation: the clause requires a party to do something ("shall", "must", "is required to")
- Permission: the clause allows a party to do something optionally ("may", "is entitled to")
- Prohibition: the clause forbids a party from doing something ("shall not", "must not", "may not")
- Penalty: the clause specifies a consequence for breach or non-compliance ("in the event of breach", "shall be liable for")

Also identify the ENTITY the clause binds (e.g. "the Contractor", "the Client", "either party").
If no entity is clearly named, use "unspecified".

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"category": "<one of Obligation|Permission|Prohibition|Penalty>", "entity": "<entity>", "confidence": "<high|medium|low>"}}

Clause:
\"\"\"{clause_text}\"\"\"
"""


def build_prompt(clause_text: str) -> str:
    return _PROMPT_TEMPLATE.format(clause_text=clause_text.strip())


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_response(raw_text: str) -> dict | None:
    """
    Defensively extracts a JSON object from a local LLM's response.
    Handles the common failure modes: markdown code fences, leading/
    trailing prose ("Here is the classification: {...}"), and trailing
    commentary after the object. Returns None if no valid object with
    the expected keys can be recovered.
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    match = _JSON_OBJECT_PATTERN.search(text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict) or "category" not in parsed:
        return None

    return parsed


def _normalize_category(raw_category: str) -> DeonticCategory:
    normalized = (raw_category or "").strip().lower()
    for category in DeonticCategory:
        if category.value.lower() == normalized:
            return category
    return DeonticCategory.UNKNOWN


def classify_clause(clause: Clause, llm_client: LLMClient | None = None) -> DeonticLabel:
    """
    Classifies a single clause. Accepts an injected LLMClient so tests
    can pass a fake without hitting a real backend; defaults to a client
    built from settings.llm_provider otherwise.
    """
    client = llm_client or LLMClient()

    raw_text = client.chat(build_prompt(clause.text))

    parsed = parse_llm_response(raw_text)
    if parsed is None:
        return DeonticLabel(
            clause_id=clause.id,
            category=DeonticCategory.UNKNOWN,
            entity=None,
            confidence="low",
            raw_response=raw_text,
        )

    return DeonticLabel(
        clause_id=clause.id,
        category=_normalize_category(parsed.get("category", "")),
        entity=parsed.get("entity") or None,
        confidence=parsed.get("confidence", "low"),
        raw_response=raw_text,
    )


def classify_clauses(clauses: list[Clause], llm_client: LLMClient | None = None) -> list[DeonticLabel]:
    """
    Classifies a full document's clauses, optionally concurrently
    (settings.llm_max_workers). Defaults to 1 (sequential) for Ollama,
    since a local single-instance server just queues concurrent requests
    behind each other anyway; raise it in .env when llm_provider=groq,
    since Groq's hosted API is built to handle concurrent traffic and
    this is exactly the step that was slow under a serialized local model.
    """
    client = llm_client or LLMClient()
    workers = max(1, settings.llm_max_workers)

    if workers == 1:
        return [classify_clause(clause, client) for clause in clauses]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Preserve input order in the result even though completion order varies.
        return list(pool.map(lambda c: classify_clause(c, client), clauses))
