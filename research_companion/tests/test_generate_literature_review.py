"""
tests/test_generate_literature_review.py
──────────────────────────────────────────
Unit tests for tools/generate_literature_review.py

Tests cover:
    - Empty chunks → graceful empty-state response
    - Unique paper collection
    - Section parsing from model output
    - Citation extraction
    - Mock review generation (dev/test mode)
    - watsonx.ai model call (mocked)
    - Section completeness
    - Exception handling
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.generate_literature_review import (
    generate_literature_review,
    _build_context_block,
    _get_unique_papers,
    _extract_citations,
    _parse_sections,
    SECTION_KEYS,
    NOT_AVAILABLE_MSG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "chunk_id": "a1",
        "paper_title": "Deep Learning",
        "authors": "LeCun et al.",
        "year": "2015",
        "page_ref": "Page 1",
        "source_text": "We present deep learning methods for feature extraction.",
        "similarity_score": 0.95,
    },
    {
        "chunk_id": "a2",
        "paper_title": "Deep Learning",
        "authors": "LeCun et al.",
        "year": "2015",
        "page_ref": "Page 5",
        "source_text": "Limitations include high computational cost and large data requirements.",
        "similarity_score": 0.87,
    },
    {
        "chunk_id": "b1",
        "paper_title": "Generative Adversarial Networks",
        "authors": "Goodfellow et al.",
        "year": "2014",
        "page_ref": "Page 2",
        "source_text": "GANs consist of a generator and a discriminator trained in opposition.",
        "similarity_score": 0.90,
    },
    {
        "chunk_id": "b2",
        "paper_title": "Generative Adversarial Networks",
        "authors": "Goodfellow et al.",
        "year": "2014",
        "page_ref": "Page 8",
        "source_text": "Future work should explore training stability improvements.",
        "similarity_score": 0.82,
    },
]

SAMPLE_MODEL_OUTPUT = """## 1. Background

Deep learning has transformed many fields [Deep Learning, Page 1]. Neural networks form the basis of modern AI [Generative Adversarial Networks, Page 2].

## 2. Methods and Approaches

Feature extraction is central to deep learning [Deep Learning, Page 1]. GANs use a generator and discriminator [Generative Adversarial Networks, Page 2].

## 3. Key Findings

Deep learning achieves high accuracy on benchmark tasks [Deep Learning, Page 5].

## 4. Limitations

High computational cost and data requirements are significant limitations [Deep Learning, Page 5].

## 5. Research Gaps

Training stability remains an open problem [Generative Adversarial Networks, Page 8].

## 6. Future Research Directions

Future work should address GAN stability [Generative Adversarial Networks, Page 8].
"""


# ---------------------------------------------------------------------------
# Tests: empty chunks
# ---------------------------------------------------------------------------

class TestGenerateLiteratureReviewEmptyChunks(unittest.TestCase):

    def test_empty_chunks_returns_success(self):
        result = generate_literature_review(chunks=[])
        self.assertTrue(result["success"])

    def test_empty_chunks_no_error(self):
        result = generate_literature_review(chunks=[])
        self.assertIsNone(result["error"])

    def test_empty_chunks_message_mentions_ingest(self):
        result = generate_literature_review(chunks=[])
        self.assertIn("ingest", result["literature_review"].lower())

    def test_result_has_expected_keys(self):
        result = generate_literature_review(chunks=[])
        required_keys = {
            "success", "literature_review", "sections",
            "citations", "papers_covered", "model_id", "error",
        }
        self.assertTrue(required_keys.issubset(result.keys()))


# ---------------------------------------------------------------------------
# Tests: _get_unique_papers
# ---------------------------------------------------------------------------

class TestGetUniquePapers(unittest.TestCase):

    def test_returns_unique_titles(self):
        papers = _get_unique_papers(SAMPLE_CHUNKS)
        self.assertEqual(len(papers), 2)
        self.assertIn("Deep Learning", papers)
        self.assertIn("Generative Adversarial Networks", papers)

    def test_preserves_insertion_order(self):
        papers = _get_unique_papers(SAMPLE_CHUNKS)
        self.assertEqual(papers[0], "Deep Learning")
        self.assertEqual(papers[1], "Generative Adversarial Networks")

    def test_single_paper(self):
        papers = _get_unique_papers(SAMPLE_CHUNKS[:2])
        self.assertEqual(papers, ["Deep Learning"])

    def test_empty_chunks(self):
        papers = _get_unique_papers([])
        self.assertEqual(papers, [])


# ---------------------------------------------------------------------------
# Tests: _extract_citations
# ---------------------------------------------------------------------------

class TestExtractCitations(unittest.TestCase):

    def test_extracts_all_citations(self):
        text = "[Deep Learning, Page 1] and [GANs, Page 2] and [BERT, Page 3]."
        cits = _extract_citations(text)
        self.assertEqual(len(cits), 3)

    def test_deduplicates(self):
        text = "[Paper A, Page 1] text [Paper A, Page 1] more text."
        cits = _extract_citations(text)
        self.assertEqual(len(cits), 1)

    def test_empty_text(self):
        self.assertEqual(_extract_citations(""), [])

    def test_no_citations(self):
        self.assertEqual(_extract_citations("No citations here."), [])


# ---------------------------------------------------------------------------
# Tests: _parse_sections
# ---------------------------------------------------------------------------

class TestParseSections(unittest.TestCase):

    def test_all_six_sections_present(self):
        sections = _parse_sections(SAMPLE_MODEL_OUTPUT)
        for key in SECTION_KEYS:
            self.assertIn(key, sections)

    def test_background_content_extracted(self):
        sections = _parse_sections(SAMPLE_MODEL_OUTPUT)
        self.assertIn("deep learning", sections["background"].lower())

    def test_methods_content_extracted(self):
        sections = _parse_sections(SAMPLE_MODEL_OUTPUT)
        self.assertIn("feature extraction", sections["methods_and_approaches"].lower())

    def test_limitations_content_extracted(self):
        sections = _parse_sections(SAMPLE_MODEL_OUTPUT)
        self.assertIn("computational", sections["limitations"].lower())

    def test_missing_section_gets_not_available(self):
        partial_output = "## 1. Background\n\nSome background text.\n\n## 3. Key Findings\n\nSome findings."
        sections = _parse_sections(partial_output)
        # Section 2 was missing — should default to NOT_AVAILABLE_MSG
        self.assertEqual(sections["methods_and_approaches"], NOT_AVAILABLE_MSG)


# ---------------------------------------------------------------------------
# Tests: full generate_literature_review function
# ---------------------------------------------------------------------------

class TestGenerateLiteratureReviewFull(unittest.TestCase):

    @patch("tools.generate_literature_review._call_watsonx_model", return_value=None)
    def test_dev_mode_generates_review(self, mock_model):
        result = generate_literature_review(chunks=SAMPLE_CHUNKS)
        self.assertTrue(result["success"])
        self.assertGreater(len(result["literature_review"]), 50)
        self.assertIsNone(result["error"])

    @patch("tools.generate_literature_review._call_watsonx_model")
    def test_model_output_used_when_available(self, mock_model):
        mock_model.return_value = SAMPLE_MODEL_OUTPUT
        result = generate_literature_review(chunks=SAMPLE_CHUNKS)
        self.assertTrue(result["success"])
        self.assertIn("Background", result["literature_review"])
        self.assertIn("Methods", result["literature_review"])
        self.assertIn("Findings", result["literature_review"])
        self.assertIn("Limitations", result["literature_review"])
        self.assertIn("Gaps", result["literature_review"])
        self.assertIn("Future", result["literature_review"])

    @patch("tools.generate_literature_review._call_watsonx_model")
    def test_papers_covered_correct(self, mock_model):
        mock_model.return_value = SAMPLE_MODEL_OUTPUT
        result = generate_literature_review(chunks=SAMPLE_CHUNKS)
        self.assertIn("Deep Learning", result["papers_covered"])
        self.assertIn("Generative Adversarial Networks", result["papers_covered"])

    @patch("tools.generate_literature_review._call_watsonx_model")
    def test_citations_extracted_from_review(self, mock_model):
        mock_model.return_value = SAMPLE_MODEL_OUTPUT
        result = generate_literature_review(chunks=SAMPLE_CHUNKS)
        self.assertGreater(len(result["citations"]), 0)

    @patch("tools.generate_literature_review._call_watsonx_model")
    def test_sections_dict_has_all_keys(self, mock_model):
        mock_model.return_value = SAMPLE_MODEL_OUTPUT
        result = generate_literature_review(chunks=SAMPLE_CHUNKS)
        for key in SECTION_KEYS:
            self.assertIn(key, result["sections"])

    @patch("tools.generate_literature_review._call_watsonx_model")
    def test_exception_captured(self, mock_model):
        mock_model.side_effect = RuntimeError("Model timeout")
        result = generate_literature_review(chunks=SAMPLE_CHUNKS)
        self.assertFalse(result["success"])
        self.assertIn("Model timeout", result["error"])

    def test_model_id_defaults_to_env_or_granite(self):
        result = generate_literature_review(chunks=[])
        self.assertIn("granite", result["model_id"].lower())

    @patch("tools.generate_literature_review._call_watsonx_model", return_value=None)
    def test_two_papers_covered_in_dev_mode(self, mock_model):
        result = generate_literature_review(chunks=SAMPLE_CHUNKS)
        self.assertEqual(len(result["papers_covered"]), 2)


if __name__ == "__main__":
    unittest.main()
