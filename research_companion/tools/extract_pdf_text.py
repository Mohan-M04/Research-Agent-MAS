"""
tools/extract_pdf_text.py
─────────────────────────
Tool: extract_pdf_text

Purpose:
    Extract plain text, page-wise text, paper title, authors, publication year,
    and general metadata from an uploaded PDF research paper.

Input:
    pdf_path (str): Absolute or relative path to the PDF file.

Output (dict):
    {
        "success": bool,
        "pdf_path": str,
        "paper_title": str | None,
        "authors": list[str],
        "publication_year": str | None,
        "total_pages": int,
        "full_text": str,
        "pages": [
            {
                "page_number": int,
                "text": str,
                "char_count": int
            },
            ...
        ],
        "metadata": dict,
        "error": str | None
    }
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_import_pymupdf():
    """Prefer PyMuPDF (fitz) for best extraction quality."""
    try:
        import fitz  # pymupdf
        return fitz
    except ImportError:
        return None


def _try_import_pypdf2():
    """Fall back to PyPDF2 if PyMuPDF is not installed."""
    try:
        import PyPDF2
        return PyPDF2
    except ImportError:
        return None


def _extract_with_pymupdf(pdf_path: str) -> dict[str, Any]:
    """Extract text page-by-page using PyMuPDF."""
    fitz = _try_import_pymupdf()
    if fitz is None:
        raise ImportError("pymupdf is not installed. Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages = []
    full_text_parts = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append({
            "page_number": page_num,
            "text": text.strip(),
            "char_count": len(text.strip()),
        })
        full_text_parts.append(text)

    metadata = doc.metadata or {}
    doc.close()

    return {
        "pages": pages,
        "full_text": "\n".join(full_text_parts),
        "raw_metadata": metadata,
        "total_pages": len(pages),
    }


def _extract_with_pypdf2(pdf_path: str) -> dict[str, Any]:
    """Extract text page-by-page using PyPDF2 as fallback."""
    PyPDF2 = _try_import_pypdf2()
    if PyPDF2 is None:
        raise ImportError("Neither pymupdf nor PyPDF2 is installed.")

    pages = []
    full_text_parts = []

    with open(pdf_path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        raw_metadata = reader.metadata or {}

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({
                "page_number": page_num,
                "text": text.strip(),
                "char_count": len(text.strip()),
            })
            full_text_parts.append(text)

    return {
        "pages": pages,
        "full_text": "\n".join(full_text_parts),
        "raw_metadata": dict(raw_metadata),
        "total_pages": len(pages),
    }


def _infer_paper_title(full_text: str, raw_metadata: dict) -> str | None:
    """
    Attempt to infer the paper title from metadata or the first lines of text.
    Returns None if a confident title cannot be found.
    """
    # 1. Try PDF metadata title field
    for key in ("/Title", "title", "Title"):
        if raw_metadata.get(key):
            candidate = raw_metadata[key].strip()
            if len(candidate) > 5:
                return candidate

    # 2. Heuristic: first non-empty line of the first page that looks like a title
    #    (not a URL, not a number, not a common header like "Abstract")
    skip_patterns = re.compile(
        r"^(abstract|introduction|doi|http|www|arxiv|preprint|©|copyright|\d+)",
        re.IGNORECASE,
    )
    for line in full_text.split("\n")[:30]:
        line = line.strip()
        if 10 < len(line) < 200 and not skip_patterns.match(line):
            return line

    return None


def _infer_authors(full_text: str, raw_metadata: dict) -> list[str]:
    """
    Attempt to extract author names from metadata or text.
    Returns an empty list when no authors can be confidently identified.
    """
    # 1. Try PDF metadata author field
    for key in ("/Author", "author", "Author"):
        if raw_metadata.get(key):
            raw = raw_metadata[key].strip()
            # Split on semicolons first (always safe author separator).
            # Only fall back to comma-splitting when no semicolons are present
            # AND the comma is followed by a space and a capital letter/digit
            # (i.e. a new author name, not an initial like "Smith, J.").
            if ";" in raw:
                authors = [a.strip() for a in raw.split(";") if a.strip()]
            else:
                # Split on ", " only when followed by a full surname (capital +
                # at least one lowercase letter), NOT on initials like "A." or "J.".
                authors = [a.strip() for a in re.split(r",\s+(?=[A-Z][a-z]{1,})", raw) if a.strip()]
            if authors:
                return authors

    return []


def _infer_publication_year(full_text: str, raw_metadata: dict) -> str | None:
    """
    Attempt to extract a 4-digit publication year from metadata or text.
    """
    for key in ("/CreationDate", "creationDate", "CreationDate"):
        val = raw_metadata.get(key, "")
        m = re.search(r"(20\d{2}|19\d{2})", str(val))
        if m:
            return m.group(1)

    # Search the first 500 characters of text for a plausible year
    m = re.search(r"\b(20\d{2}|19\d{2})\b", full_text[:500])
    if m:
        return m.group(1)

    return None


def _clean_metadata(raw: dict) -> dict:
    """Return a plain-Python serialisable copy of the raw metadata dict."""
    cleaned = {}
    for k, v in raw.items():
        try:
            cleaned[str(k)] = str(v) if v is not None else None
        except Exception:
            pass
    return cleaned


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: str) -> dict[str, Any]:
    """
    Extract text, title, authors, year, and page-wise content from a PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to process.

    Returns
    -------
    dict
        Extraction result containing success flag, paper metadata, full text,
        page-level text blocks, and any error message.
    """
    result: dict[str, Any] = {
        "success": False,
        "pdf_path": pdf_path,
        "paper_title": None,
        "authors": [],
        "publication_year": None,
        "total_pages": 0,
        "full_text": "",
        "pages": [],
        "metadata": {},
        "error": None,
    }

    # Validate extension first (before the filesystem check) so the error message
    # is accurate regardless of whether the non-PDF file exists on disk.
    if not pdf_path or not pdf_path.lower().endswith(".pdf"):
        result["error"] = f"File does not appear to be a PDF: '{pdf_path}'"
        logger.error(result["error"])
        return result

    if not os.path.isfile(pdf_path):
        result["error"] = f"File not found or path is invalid: '{pdf_path}'"
        logger.error(result["error"])
        return result

    try:
        # Prefer PyMuPDF; fall back to PyPDF2
        fitz = _try_import_pymupdf()
        if fitz is not None:
            extracted = _extract_with_pymupdf(pdf_path)
            logger.debug("Extracted '%s' using PyMuPDF (%d pages).", pdf_path, extracted["total_pages"])
        else:
            extracted = _extract_with_pypdf2(pdf_path)
            logger.debug("Extracted '%s' using PyPDF2 (%d pages).", pdf_path, extracted["total_pages"])

        full_text: str = extracted["full_text"]
        raw_metadata: dict = extracted["raw_metadata"]

        result.update({
            "success": True,
            "total_pages": extracted["total_pages"],
            "full_text": full_text,
            "pages": extracted["pages"],
            "metadata": _clean_metadata(raw_metadata),
            "paper_title": _infer_paper_title(full_text, raw_metadata),
            "authors": _infer_authors(full_text, raw_metadata),
            "publication_year": _infer_publication_year(full_text, raw_metadata),
        })

    except Exception as exc:
        result["error"] = f"Extraction failed: {exc}"
        logger.exception("Failed to extract text from '%s'.", pdf_path)

    return result
