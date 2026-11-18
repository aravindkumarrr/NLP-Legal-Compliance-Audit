from app.pipeline.deontic import DeonticCategory, _normalize_category, build_prompt, parse_llm_response


def test_parses_clean_json():
    raw = '{"category": "Obligation", "entity": "the Contractor", "confidence": "high"}'
    parsed = parse_llm_response(raw)
    assert parsed["category"] == "Obligation"
    assert parsed["entity"] == "the Contractor"


def test_parses_json_wrapped_in_markdown_fences():
    raw = '```json\n{"category": "Prohibition", "entity": "the Client", "confidence": "medium"}\n```'
    parsed = parse_llm_response(raw)
    assert parsed["category"] == "Prohibition"


def test_parses_json_with_leading_prose():
    raw = 'Here is the classification:\n{"category": "Penalty", "entity": "the Vendor", "confidence": "low"}'
    parsed = parse_llm_response(raw)
    assert parsed["category"] == "Penalty"


def test_returns_none_for_garbage():
    assert parse_llm_response("I cannot classify this clause.") is None


def test_returns_none_for_empty_string():
    assert parse_llm_response("") is None


def test_returns_none_when_category_key_missing():
    raw = '{"entity": "the Contractor", "confidence": "high"}'
    assert parse_llm_response(raw) is None


def test_normalize_category_is_case_insensitive():
    assert _normalize_category("obligation") == DeonticCategory.OBLIGATION
    assert _normalize_category("PROHIBITION") == DeonticCategory.PROHIBITION


def test_normalize_category_falls_back_to_unknown():
    assert _normalize_category("something else") == DeonticCategory.UNKNOWN
    assert _normalize_category("") == DeonticCategory.UNKNOWN


def test_build_prompt_includes_clause_text():
    prompt = build_prompt("The Contractor shall deliver the goods within 30 days.")
    assert "The Contractor shall deliver the goods within 30 days." in prompt
    assert "Obligation" in prompt and "Permission" in prompt
