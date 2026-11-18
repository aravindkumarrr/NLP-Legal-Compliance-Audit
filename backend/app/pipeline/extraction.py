"""
Stage 1a: Text extraction.

Pulls raw text out of an uploaded PDF or DOCX file. Kept deliberately dumb —
this module's only job is "bytes in, one clean text blob out". Clause
segmentation is a separate concern (see segmentation.py).
"""
from __future__ import annotations

import io
import re

from pypdf import PdfReader
from docx import Document as DocxDocument


class UnsupportedFileType(Exception):
    pass


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Dispatches to the right extractor based on file extension, then
    normalizes whitespace so downstream segmentation has consistent input.
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        raw = _extract_pdf(file_bytes)
    elif lower.endswith(".docx"):
        raw = _extract_docx(file_bytes)
    else:
        raise UnsupportedFileType(
            f"Unsupported file type for '{filename}'. Only .pdf and .docx are supported."
        )
    return _normalize_text(raw)


def _extract_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def _normalize_text(text: str) -> str:
    """
    Light cleanup: collapse repeated whitespace, drop common page-number-only
    lines (a frequent PDF-extraction artifact), and normalize line endings.
    Deliberately conservative — aggressive cleanup risks eating clause
    numbering that segmentation.py depends on.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Drop lines that are just a page number (e.g. "12", "Page 3 of 10")
        if re.fullmatch(r"(page\s+)?\d+(\s+of\s+\d+)?", stripped, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    # Collapse 3+ blank lines down to a double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
