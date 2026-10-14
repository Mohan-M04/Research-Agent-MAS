"""
tests/test_extract_pdf_text.py
────────────────────────────────
Unit tests for tools/extract_pdf_text.py

Tests cover:
    - Missing / invalid file paths
    - Non-PDF file rejection
    - Successful extraction (mocked PDF library)
    - Title / author / year inference
    - Page-wise output structure
    - Graceful error handling
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# Ensure package root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.extract_pdf_text import (
    extract_pdf_text,
    _infer_paper_title,
    _infer_authors,
    _infer_publication_year,
    _clean_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Attention Is All You Need\n\n"
    "Vaswani, A.; Shazeer, N.; Parmar, N.\n\n"
    "Abstract\n\n"
    "We propose a new simple network architecture, the Transformer. "
    "Experiments on two machine translation tasks show these models to be superior. "
    "Published 2017.\n\n"
    "1. Introduction\n\n"
    "Recurrent neural networks have long been the dominant model for sequence transduction."
)

SAMPLE_METADATA = {
    "/Title": "Attention Is All You Need",
    "/Author": "Vaswani, A.; Shazeer, N.",
    "/CreationDate": "D:20170101000000",
}


# ---------------------------------------------------------------------------
# Tests: file-path validation
# ---------------------------------------------------------------------------

class TestExtractPdfTextFileValidation(unittest.TestCase):

    def test_missing_file_returns_error(self):
        result = extract_pdf_text("/nonexistent/path/paper.pdf")
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertIn("not found", result["error"].lower())

    def test_empty_path_returns_error(self):
        result = extract_pdf_text("")
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_non_pdf_extension_returns_error(self):
        result = extract_pdf_text("/tmp/document.docx")
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertIn("pdf", result["error"].lower())

    def test_result_has_expected_keys(self):
        result = extract_pdf_text("/nonexistent.pdf")
        required_keys = {
            "success", "pdf_path", "paper_title", "authors",
            "publication_year", "total_pages", "full_text", "pages",
            "metadata", "error",
        }
        self.assertTrue(required_keys.issubset(result.keys()))


# ---------------------------------------------------------------------------
# Tests: successful extraction (PyMuPDF mocked)
# ---------------------------------------------------------------------------

def _make_mock_fitz(text: str = SAMPLE_TEXT):
    """Build a minimal mock of the fitz (PyMuPDF) module."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = text

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
    mock_doc.metadata = SAMPLE_METADATA
    mock_doc.close = MagicMock()

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    return mock_fitz


class TestExtractPdfTextSuccess(unittest.TestCase):

    @patch("tools.extract_pdf_text.os.path.isfile", return_value=True)
    @patch("tools.extract_pdf_text._try_import_pymupdf")
    def test_success_with_pymupdf(self, mock_try_fitz, mock_isfile):
        mock_fitz = _make_mock_fitz()
        mock_try_fitz.return_value = mock_fitz

        # Patch _extract_with_pymupdf directly
        with patch("tools.extract_pdf_text._extract_with_pymupdf") as mock_extract:
            mock_extract.return_value = {
                "pages": [{"page_number": 1, "text": SAMPLE_TEXT, "char_count": len(SAMPLE_TEXT)}],
                "full_text": SAMPLE_TEXT,
                "raw_metadata": SAMPLE_METADATA,
                "total_pages": 1,
            }
            result = extract_pdf_text("fake_paper.pdf")

        self.assertTrue(result["success"])
        self.assertEqual(result["total_pages"], 1)
        self.assertGreater(len(result["full_text"]), 0)
        self.assertEqual(len(result["pages"]), 1)
        self.assertIsNone(result["error"])

    @patch("tools.extract_pdf_text.os.path.isfile", return_value=True)
    @patch("tools.extract_pdf_text._try_import_pymupdf", return_value=None)
    @patch("tools.extract_pdf_text._extract_with_pypdf2")
    def test_fallback_to_pypdf2(self, mock_pypdf2_extract, mock_fitz, mock_isfile):
        mock_pypdf2_extract.return_value = {
            "pages": [{"page_number": 1, "text": SAMPLE_TEXT, "char_count": len(SAMPLE_TEXT)}],
            "full_text": SAMPLE_TEXT,
            "raw_metadata": {},
            "total_pages": 1,
        }
        result = extract_pdf_text("fake_paper.pdf")
        self.assertTrue(result["success"])
        self.assertEqual(result["total_pages"], 1)

    @patch("tools.extract_pdf_text.os.path.isfile", return_value=True)
    @patch("tools.extract_pdf_text._try_import_pymupdf")
    @patch("tools.extract_pdf_text._extract_with_pymupdf")
    def test_extraction_exception_captured(self, mock_extract, mock_fitz, mock_isfile):
        mock_fitz.return_value = MagicMock()
        mock_extract.side_effect = RuntimeError("Corrupt PDF")
        result = extract_pdf_text("corrupt.pdf")
        self.assertFalse(result["success"])
        self.assertIn("Corrupt PDF", result["error"])


# ---------------------------------------------------------------------------
# Tests: _infer_paper_title
# ---------------------------------------------------------------------------

class TestInferPaperTitle(unittest.TestCase):

    def test_title_from_metadata(self):
        title = _infer_paper_title("", {"/Title": "Deep Learning"})
        self.assertEqual(title, "Deep Learning")

    def test_title_fallback_to_text(self):
        text = "Attention Is All You Need\n\nAbstract\nWe propose..."
        title = _infer_paper_title(text, {})
        self.assertEqual(title, "Attention Is All You Need")

    def test_title_skips_abstract(self):
        text = "Abstract\nWe propose a new model."
        title = _infer_paper_title(text, {})
        # "Abstract" should be skipped; next candidate is used
        self.assertNotEqual(title, "Abstract")

    def test_returns_none_for_empty(self):
        title = _infer_paper_title("", {})
        self.assertIsNone(title)

    def test_skips_short_lines(self):
        text = "Hi\nAttention Is All You Need\nAbstract"
        title = _infer_paper_title(text, {})
        # "Hi" is <10 chars; "Attention Is All You Need" should be selected
        self.assertIn("Attention", title)


# ---------------------------------------------------------------------------
# Tests: _infer_authors
# ---------------------------------------------------------------------------

class TestInferAuthors(unittest.TestCase):

    def test_authors_from_metadata(self):
        authors = _infer_authors("", {"/Author": "Smith, J.; Jones, K."})
        self.assertEqual(authors, ["Smith, J.", "Jones, K."])

    def test_returns_empty_when_no_metadata(self):
        authors = _infer_authors("Some paper text without author info.", {})
        self.assertEqual(authors, [])

    def test_single_author(self):
        authors = _infer_authors("", {"/Author": "Turing, A."})
        self.assertEqual(authors, ["Turing, A."])

    def test_author_split_by_semicolon(self):
        authors = _infer_authors("", {"/Author": "A; B; C"})
        self.assertEqual(len(authors), 3)

    def test_author_split_by_comma(self):
        # Realistic comma-separated authors where each name starts with capital+lowercase
        authors = _infer_authors("", {"/Author": "Smith, Jones, Brown"})
        self.assertEqual(len(authors), 3)


# ---------------------------------------------------------------------------
# Tests: _infer_publication_year
# ---------------------------------------------------------------------------

class TestInferPublicationYear(unittest.TestCase):

    def test_year_from_creation_date(self):
        year = _infer_publication_year("", {"/CreationDate": "D:20230515"})
        self.assertEqual(year, "2023")

    def test_year_from_text(self):
        year = _infer_publication_year("Published in 2021 by IEEE.", {})
        self.assertEqual(year, "2021")

    def test_returns_none_when_absent(self):
        year = _infer_publication_year("No year here.", {})
        self.assertIsNone(year)

    def test_prefers_metadata_over_text(self):
        year = _infer_publication_year("Paper from 2015.", {"/CreationDate": "D:20230101"})
        self.assertEqual(year, "2023")


# ---------------------------------------------------------------------------
# Tests: _clean_metadata
# ---------------------------------------------------------------------------

class TestCleanMetadata(unittest.TestCase):

    def test_returns_plain_dict(self):
        raw = {"/Title": "Test", "/Author": "Alice", "/CreationDate": None}
        cleaned = _clean_metadata(raw)
        self.assertIsInstance(cleaned, dict)
        self.assertEqual(cleaned["/Title"], "Test")
        self.assertIsNone(cleaned["/CreationDate"])

    def test_handles_non_string_keys(self):
        raw = {1: "numeric key", "normal": "value"}
        cleaned = _clean_metadata(raw)
        self.assertIn("1", cleaned)
        self.assertIn("normal", cleaned)


if __name__ == "__main__":
    unittest.main()
