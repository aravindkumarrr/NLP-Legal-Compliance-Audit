from app.pipeline.segmentation import segment_clauses


def test_numbered_segmentation_basic():
    text = (
        "1. The Contractor shall deliver the goods within 30 days.\n"
        "2. The Client shall pay within 15 days of delivery.\n"
        "3. Either party may terminate this agreement with 30 days notice.\n"
    )
    clauses = segment_clauses(text)
    assert len(clauses) == 3
    assert clauses[0].label == "1"
    assert "deliver the goods" in clauses[0].text
    assert clauses[1].label == "2"
    assert clauses[2].label == "3"


def test_numbered_segmentation_nested():
    text = (
        "1.1 The Contractor shall maintain confidentiality.\n"
        "1.2 The Contractor shall not disclose trade secrets.\n"
        "2.1 Payment shall be made in full within 30 days.\n"
        "2.2 Late payment shall incur a penalty of 2% per month.\n"
    )
    clauses = segment_clauses(text)
    assert len(clauses) == 4
    assert clauses[0].label == "1.1"
    assert clauses[3].label == "2.2"
    assert "penalty" in clauses[3].text


def test_lettered_subclauses():
    text = (
        "3. The following conditions shall apply:\n"
        "(a) The vendor must provide a warranty of one year.\n"
        "(b) The vendor must not subcontract without written consent.\n"
        "(c) Breach of this clause shall result in contract termination.\n"
    )
    clauses = segment_clauses(text)
    labels = [c.label for c in clauses]
    assert labels == ["3", "a", "b", "c"]


def test_falls_back_to_paragraph_when_no_numbering():
    text = (
        "This agreement is made between the parties for the purpose of "
        "establishing terms of engagement between the two entities involved.\n\n"
        "The parties agree that all disputes will be settled through binding "
        "arbitration in accordance with the rules established herein.\n\n"
        "This document shall be governed by the laws of the jurisdiction "
        "specified in the accompanying schedule of terms."
    )
    clauses = segment_clauses(text)
    assert len(clauses) == 3
    assert all(c.label is None for c in clauses)


def test_short_fragments_dropped_in_fallback():
    text = "Header\n\nThis is a real paragraph with enough content to count as a clause on its own.\n\nOK"
    clauses = segment_clauses(text)
    # "Header" and "OK" should both be dropped as too short
    assert len(clauses) == 1
    assert "real paragraph" in clauses[0].text


def test_clause_ids_are_sequential_and_stable():
    text = "1. First clause here.\n2. Second clause here.\n3. Third clause here.\n"
    clauses = segment_clauses(text)
    assert [c.id for c in clauses] == ["clause_0000", "clause_0001", "clause_0002"]
    assert [c.order for c in clauses] == [0, 1, 2]


def test_few_stray_numbers_dont_hijack_segmentation():
    """
    A document with only 1-2 incidental numbers (e.g. a date) shouldn't be
    treated as numbered — should fall back to paragraph segmentation.
    """
    text = (
        "This agreement is effective as of 2024 and shall remain in force "
        "until terminated by either party in writing at any time.\n\n"
        "Both parties acknowledge they have read and understood the terms "
        "set forth in this document prior to signing below."
    )
    clauses = segment_clauses(text)
    assert all(c.label is None for c in clauses)


def test_stray_measurement_number_not_treated_as_clause_label():
    """
    A line-wrapped measurement ("65 kms.") that happens to land at the
    start of a physical line — a real pypdf extraction artifact — must
    not be read as clause numbering. Previously this spawned a bogus
    4th clause labeled "65" containing only "kms. and includes...",
    splitting clause 2's text in half.
    """
    text = (
        "1. The Contractor shall deliver the goods within 30 days.\n"
        "2. The delivery location covers an area of approximately\n"
        "65 kms. and includes all access roads to the site.\n"
        "3. Either party may terminate this agreement with 30 days notice.\n"
    )
    clauses = segment_clauses(text)
    assert len(clauses) == 3
    assert [c.label for c in clauses] == ["1", "2", "3"]
    # the stray number stays folded into clause 2's text, not its own clause
    assert "65 kms." in clauses[1].text


def test_inline_numbering_without_newlines_is_not_one_merged_clause():
    """
    pypdf sometimes extracts a whole run of clauses onto one physical line
    with no newline (and no blank line) between them. Previously this
    numbering was invisible to the (line-start-anchored) detector, so the
    match count fell below the threshold and the whole passage collapsed
    into a single oversized paragraph-fallback clause.
    """
    text = (
        "1. The Contractor shall deliver the goods within 30 days. "
        "2. The Client shall pay within 15 days of delivery. "
        "3. Either party may terminate this agreement with 30 days notice."
    )
    clauses = segment_clauses(text)
    assert len(clauses) == 3
    assert [c.label for c in clauses] == ["1", "2", "3"]
    assert "deliver the goods" in clauses[0].text
    assert "Client shall pay" in clauses[1].text
    assert "terminate" in clauses[2].text
