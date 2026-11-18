"""
Stage 1b: Clause segmentation.

Splits normalized document text into individually addressable clauses.

Strategy (hybrid, in priority order):
  1. If the document has detectable numbered clauses (1., 1.1, (a), Section 3, etc.),
     split on that numbering — this best matches how contracts are actually
     structured and keeps cross-references intact.
  2. Otherwise, fall back to paragraph-based splitting (blank-line-separated blocks),
     since blind sentence-splitting tends to cut legal clauses mid-condition.

Each clause is returned as a Clause object carrying its detected label
(e.g. "3.2" or "(a)") so later stages (deontic graph, reporting) can cite it.

A numbering "candidate" has to clear two checks before it's trusted as a
clause boundary (see _find_numbering_candidates):
  - _is_boundary_acceptable: it has to sit somewhere a new clause could
    plausibly start -- the top of the document, right after a newline, or
    right after the previous sentence's closing punctuation. The last case
    matters because pypdf's extract_text() often runs clauses together on
    one physical line with no newline between them at all (and no blank
    line either), which used to mean a whole page -- sometimes a whole
    document -- collapsed into one giant paragraph-fallback clause.
  - _looks_like_stray_number (bare digit labels only): a number sitting at
    an otherwise-acceptable boundary still isn't clause numbering if it's
    actually a measurement, date, or count that happens to start a line or
    follow a period -- "...covers the depot.\n65 kms. and includes access
    roads..." should not spawn a clause "65". Lettered/"(a)"/"Section N"
    forms skip this check since they're unambiguous without it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Clause:
    id: str            # stable id, e.g. "clause_0007"
    label: str | None  # detected numbering label, e.g. "3.2", "(a)", or None if undetected
    text: str
    order: int          # position in document, 0-indexed


# Matches common legal numbering styles:
#   "1.", "1.1", "1.1.2", "(a)", "(iii)", "Section 4", "Article 2"
# Deliberately NOT anchored to line start here -- boundary plausibility is
# handled separately in _is_boundary_acceptable, since a numbering match
# can legitimately start a clause even mid-line (right after the prior
# clause's closing period) when the source document has no newline there.
_NUMBERING_PATTERN = re.compile(
    r"""(
        \d+(\.\d+)*\.?                # 1.  or 1.1  or 1.1.2
        |\([a-zA-Z]+\)                # (a) (iii)
        |\([0-9]+\)                   # (1)
        |Section\s+\d+[A-Za-z]?       # Section 4
        |Article\s+\d+[A-Za-z]?       # Article 2
        |Clause\s+\d+[A-Za-z]?        # Clause 5
    )\s+""",
    re.IGNORECASE | re.VERBOSE,
)

# A label is "bare digit" style (just "1" or "1.1", no letters/keywords) --
# only this style is ambiguous enough with ordinary numbers in running text
# to need the stray-number guard below.
_BARE_DIGIT_LABEL = re.compile(r"\d+(\.\d+)*\.?")

# Characters that plausibly end a sentence/clause, so a numbering match
# right after one of these (plus whitespace) can still be a real clause
# boundary even without an intervening newline.
_SENTENCE_END_CHARS = {".", ";", ":", "!", "?"}

# Words that commonly follow a bare number when it's actually a
# measurement, duration, or count rather than clause numbering. Checked
# against the word immediately after the match (case-insensitively, with
# trailing punctuation stripped).
_UNLIKELY_NEXT_WORDS = {
    "kms", "km", "kilometers", "kilometres",
    "kg", "kgs", "kilograms", "lb", "lbs", "pounds",
    "mm", "cm", "m", "meters", "metres", "ft", "feet",
    "miles", "mi", "percent", "pct",
    "years", "year", "months", "month", "weeks", "week", "days", "day",
    "hours", "hrs", "hour", "minutes", "mins", "minute", "seconds", "secs",
    "am", "pm", "employees", "people", "units", "times", "pages", "copies",
}

# Minimum number of accepted numbering candidates before we trust
# numbering-based segmentation over the paragraph fallback. Below this, a
# couple of stray numbers (e.g. a date or a page reference) shouldn't
# hijack segmentation.
_MIN_NUMBERED_MATCHES = 3


def segment_clauses(text: str) -> list[Clause]:
    candidates = _find_numbering_candidates(text)

    if len(candidates) >= _MIN_NUMBERED_MATCHES:
        return _segment_by_numbering(text, candidates)
    return _segment_by_paragraph(text)


def _find_numbering_candidates(text: str) -> list[re.Match]:
    candidates = []
    for match in _NUMBERING_PATTERN.finditer(text):
        if not _is_boundary_acceptable(text, match.start()):
            continue
        label = match.group(1)
        if _BARE_DIGIT_LABEL.fullmatch(label) and _looks_like_stray_number(text, match.end()):
            continue
        candidates.append(match)
    return candidates


def _is_boundary_acceptable(text: str, match_start: int) -> bool:
    """
    True if position `match_start` is somewhere a new clause could
    plausibly begin: the very start of the document, right after a
    newline (the classic "each clause on its own line" layout), or right
    after sentence-ending punctuation (handles clauses extracted onto one
    continuous line with no newline in between).
    """
    i = match_start
    while i > 0 and text[i - 1] in " \t":
        i -= 1
    if i == 0:
        return True
    return text[i - 1] == "\n" or text[i - 1] in _SENTENCE_END_CHARS


def _looks_like_stray_number(text: str, after: int) -> bool:
    """
    True if the word right after a bare-digit numbering match looks like
    it belongs to a measurement/date/count rather than a clause heading --
    either a known unit-ish word ("65 kms."), or lowercase where a clause
    heading would conventionally start a capitalized sentence.
    """
    word_match = re.match(r"\S+", text[after:])
    word = word_match.group(0) if word_match else ""
    if not word:
        return True
    cleaned = word.lower().rstrip(".,;:)")
    if cleaned in _UNLIKELY_NEXT_WORDS:
        return True
    return not word[0].isupper()


def _segment_by_numbering(text: str, matches: list[re.Match]) -> list[Clause]:
    clauses: list[Clause] = []
    for i, match in enumerate(matches):
        label = match.group(1).strip().rstrip(".")
        label = label.strip("()")  # normalize "(a)" -> "a", "(1)" -> "1"
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause_text = text[start:end].strip()

        if not clause_text:
            continue

        clauses.append(
            Clause(
                id=f"clause_{i:04d}",
                label=label,
                text=clause_text,
                order=i,
            )
        )
    return clauses


def _segment_by_paragraph(text: str) -> list[Clause]:
    """
    Fallback for documents without clear numbering: split on blank lines,
    then drop very short fragments (likely headers/whitespace artifacts).
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    clauses: list[Clause] = []
    order = 0
    for para in raw_paragraphs:
        cleaned = para.strip().replace("\n", " ")
        if len(cleaned) < 20:  # skip stray headers, short artifacts
            continue
        clauses.append(
            Clause(
                id=f"clause_{order:04d}",
                label=None,
                text=cleaned,
                order=order,
            )
        )
        order += 1
    return clauses
