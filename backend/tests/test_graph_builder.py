from app.pipeline.graph_builder import _find_referenced_labels


def test_finds_section_reference():
    text = "In the event of breach of Section 3.2, the Contractor shall pay damages."
    assert _find_referenced_labels(text) == {"3.2"}


def test_finds_clause_reference():
    text = "Failure to comply with Clause 5 shall result in immediate termination."
    assert _find_referenced_labels(text) == {"5"}


def test_finds_lettered_clause_reference():
    text = "Breach of clause (a) above shall incur a penalty of 2% per month."
    assert _find_referenced_labels(text) == {"a"}


def test_finds_multiple_references():
    text = "Violation of Section 2.1 or Clause 4 shall be treated as a material breach."
    assert _find_referenced_labels(text) == {"2.1", "4"}


def test_no_references_found():
    text = "The Contractor shall deliver the goods within 30 days."
    assert _find_referenced_labels(text) == set()
