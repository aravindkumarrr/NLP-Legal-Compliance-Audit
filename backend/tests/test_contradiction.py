from app.pipeline.contradiction import _categories_conflict, has_conflicting_modality
from app.pipeline.deontic import DeonticCategory


def test_obligation_vs_prohibition_conflicts():
    assert _categories_conflict(DeonticCategory.OBLIGATION, DeonticCategory.PROHIBITION)


def test_obligation_vs_obligation_conflicts():
    assert _categories_conflict(DeonticCategory.OBLIGATION, DeonticCategory.OBLIGATION)


def test_permission_vs_obligation_does_not_conflict():
    assert not _categories_conflict(DeonticCategory.PERMISSION, DeonticCategory.OBLIGATION)


def test_penalty_vs_obligation_does_not_conflict():
    assert not _categories_conflict(DeonticCategory.PENALTY, DeonticCategory.OBLIGATION)


def test_negation_mismatch_detected():
    a = "The Contractor shall disclose all subcontractor arrangements."
    b = "The Contractor shall not disclose subcontractor arrangements without consent."
    assert has_conflicting_modality(a, b)


def test_no_negation_mismatch_when_both_affirmative():
    a = "The Contractor shall deliver the goods within 30 days."
    b = "The Contractor shall provide monthly status reports."
    assert not has_conflicting_modality(a, b)


def test_no_negation_mismatch_when_both_negated():
    a = "The Contractor shall not subcontract without written consent."
    b = "The Contractor must not disclose confidential information."
    assert not has_conflicting_modality(a, b)
