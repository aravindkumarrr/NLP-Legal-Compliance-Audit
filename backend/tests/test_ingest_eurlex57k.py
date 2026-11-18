from scripts.ingest_eurlex57k import extract_sections, passages_for_document


def _doc(**overrides):
    base = {
        "celex_id": "32014R0727",
        "title": "Some regulation",
        "header": "1.7.2014 EN Official Journal of the European Union L 192/42",
        "recitals": "Whereas the applicant lodged a request for review.",
        "main_body": [
            "Article 1. A review of the Regulation is hereby initiated.",
            "Article 2. This Regulation shall enter into force on the day of publication.",
        ],
        "eurovoc_concepts": ["1402", "2771"],
    }
    base.update(overrides)
    return base


def test_extract_sections_pulls_header_recitals_and_each_article():
    sections = extract_sections(_doc())
    labels = [label for label, _ in sections]
    assert labels == ["header", "recitals", "article_1", "article_2"]


def test_extract_sections_skips_blank_fields():
    doc = _doc(header="   ", recitals="")
    sections = extract_sections(doc)
    labels = [label for label, _ in sections]
    assert "header" not in labels
    assert "recitals" not in labels
    assert "article_1" in labels


def test_extract_sections_handles_string_main_body():
    """Some distributions might not pre-split main_body into a list."""
    doc = _doc(main_body="Article 1. A single fused main body string.")
    sections = extract_sections(doc)
    assert sections == [
        ("header", doc["header"]),
        ("recitals", doc["recitals"]),
        ("article_1", "Article 1. A single fused main body string."),
    ]


def test_extract_sections_includes_attachments_when_present():
    doc = _doc(attachments="ANNEX I List of affected products")
    sections = extract_sections(doc)
    assert ("attachments", "ANNEX I List of affected products") in sections


def test_extract_sections_falls_back_to_flat_text_field():
    doc = {"celex_id": "X", "text": "Some already-flattened document text."}
    sections = extract_sections(doc)
    assert sections == [("text", "Some already-flattened document text.")]


def test_extract_sections_empty_for_doc_with_nothing_usable():
    assert extract_sections({"celex_id": "X"}) == []


def test_passages_for_document_ids_are_stable_and_namespaced_by_celex_id():
    passages = passages_for_document(_doc(), min_chars=0)
    ids = [pid for pid, _ in passages]
    assert all(pid.startswith("32014R0727:") for pid in ids)
    assert len(ids) == len(set(ids))  # no collisions


def test_passages_for_document_drops_short_fragments():
    doc = _doc(recitals="Hi.", main_body=["Article 1. " + "A substantive clause. " * 5])
    passages = passages_for_document(doc, min_chars=40)
    texts = [text for _, text in passages]
    assert not any(t == "Hi." for t in texts)
    assert any("substantive clause" in t for t in texts)


def test_passages_for_document_resplits_multi_point_article():
    """An article with internal (a)/(b) structure should come back as
    separate passages, same as segment_clauses would do for a contract
    clause with lettered sub-points."""
    doc = _doc(main_body=[
        "Article 3. The following conditions shall apply: "
        "(a) the applicant must submit evidence within 30 days. "
        "(b) the review shall be concluded within nine months. "
        "(c) interested parties may request a hearing."
    ])
    passages = passages_for_document(doc, min_chars=0)
    labels = [pid.split(":")[1] for pid, _ in passages]
    assert len(labels) >= 3  # split on (a)/(b)/(c), not left as one fused blob
