from app.pipeline.compliance import (
    _RELEVANCE_MIN_SCORE,
    assess_clause_compliance,
    build_prompt,
    parse_llm_response,
)
from app.pipeline.embedding import SearchHit


class _ExplodingLLMClient:
    """Fails the test if .chat() is ever called -- used to prove the
    relevance gate short-circuits before any LLM call is made."""

    def chat(self, prompt: str) -> str:
        raise AssertionError("LLM should not have been called -- relevance gate should have short-circuited")


class _StubLLMClient:
    def __init__(self, response: str):
        self._response = response

    def chat(self, prompt: str) -> str:
        return self._response


def _hit(score: float, text: str = "Some regulatory passage.") -> SearchHit:
    return SearchHit(clause_id="ref_0001", document_id="eurlex", text=text, label=None, score=score)


def test_parses_clean_json():
    raw = '{"status": "gap", "rationale": "The clause omits the 72-hour notification window."}'
    parsed = parse_llm_response(raw)
    assert parsed["status"] == "gap"


def test_parses_wrapped_in_fences():
    raw = '```json\n{"status": "compliant", "rationale": "Matches the reference requirement."}\n```'
    parsed = parse_llm_response(raw)
    assert parsed["status"] == "compliant"


def test_rejects_invalid_status_value():
    raw = '{"status": "maybe", "rationale": "not sure"}'
    assert parse_llm_response(raw) is None


def test_returns_none_for_non_json():
    assert parse_llm_response("The clause looks fine to me.") is None


def test_build_prompt_includes_both_texts():
    prompt = build_prompt("The Contractor shall retain data for 10 years.", "Data shall be kept no longer than necessary.")
    assert "The Contractor shall retain data for 10 years." in prompt
    assert "Data shall be kept no longer than necessary." in prompt


def test_no_reference_passages_returns_not_applicable_without_llm_call():
    result = assess_clause_compliance(
        clause_id="clause_0001",
        clause_text="The parties shall keep this agreement confidential.",
        reference_passages=[],
        llm_client=_ExplodingLLMClient(),
    )
    assert result.status == "not_applicable"
    assert result.matched_reference is None


def test_low_relevance_match_returns_not_applicable_without_llm_call():
    """
    This is the relevance-gate fix: an unrelated nearest-neighbor match
    (the only thing Qdrant has to return from a small reference corpus)
    must not be handed to the LLM and risk coming back labeled "gap" --
    it should be recognized as not applicable before the LLM is ever called.
    """
    irrelevant_hit = _hit(score=_RELEVANCE_MIN_SCORE - 0.05, text="Unrelated passage about fisheries quotas.")
    result = assess_clause_compliance(
        clause_id="clause_0002",
        clause_text="The vendor shall maintain product liability insurance of at least $1,000,000.",
        reference_passages=[irrelevant_hit],
        llm_client=_ExplodingLLMClient(),
    )
    assert result.status == "not_applicable"
    assert result.matched_reference == "Unrelated passage about fisheries quotas."
    assert "below the relevance threshold" in result.rationale


def test_relevant_match_still_goes_to_the_llm():
    relevant_hit = _hit(score=_RELEVANCE_MIN_SCORE + 0.1, text="Insurers must retain liability coverage records for 7 years.")
    stub = _StubLLMClient('{"status": "compliant", "rationale": "Matches the retention requirement."}')
    result = assess_clause_compliance(
        clause_id="clause_0003",
        clause_text="The vendor shall retain insurance records for 7 years.",
        reference_passages=[relevant_hit],
        llm_client=stub,
    )
    assert result.status == "compliant"
    assert result.matched_reference == "Insurers must retain liability coverage records for 7 years."


def test_score_exactly_at_threshold_is_treated_as_relevant():
    """Boundary check: the gate is `< _RELEVANCE_MIN_SCORE`, so a score
    exactly at the threshold should still go to the LLM, not be gated out."""
    hit = _hit(score=_RELEVANCE_MIN_SCORE)
    stub = _StubLLMClient('{"status": "gap", "rationale": "Missing the required clause."}')
    result = assess_clause_compliance(
        clause_id="clause_0004",
        clause_text="Some clause text.",
        reference_passages=[hit],
        llm_client=stub,
    )
    assert result.status == "gap"
