"""
tests/test_generate_cited_answer.py
─────────────────────────────────────
Unit tests for tools/generate_cited_answer.py

Tests cover:
    - Empty query validation
    - Empty chunks → NOT_AVAILABLE_MSG response
    - Citation extraction from answer text
    - Context block formatting
    - Mock answer synthesis in dev/test mode
    - watsonx.ai model call integration (mocked)
    - Exception handling
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.generate_cited_answer import (
    generate_cited_answer,
    _build_context_block,
    _extract_citations_from_answer,
    NOT_AVAILABLE_MSG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {
        "chunk_id": "abc001",
        "paper_title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": "2017",
        "page_ref": "Page 3",
        "source_text": (
            "We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence entirely."
        ),
        "similarity_score": 0.92,
    },
    {
        "chunk_id": "abc002",
        "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": "Devlin et al.",
        "year": "2019",
        "page_ref": "Page 2",
        "source_text": (
            "BERT is designed to pre-train deep bidirectional representations "
            "from unlabeled text by jointly conditioning on both left and right context."
        ),
        "similarity_score": 0.88,
    },
]


# ---------------------------------------------------------------------------
# Tests: input validation
# ---------------------------------------------------------------------------

class TestGenerateCitedAnswerValidation(unittest.TestCase):

    def test_empty_query_returns_error(self):
        result = generate_cited_answer(query="", chunks=SAMPLE_CHUNKS)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_whitespace_query_returns_error(self):
        result = generate_cited_answer(query="   ", chunks=SAMPLE_CHUNKS)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_result_has_expected_keys(self):
        result = generate_cited_answer(query="test", chunks=[])
        required_keys = {"success", "query", "answer", "citations", "sources_used", "model_id", "error"}
        self.assertTrue(required_keys.issubset(result.keys()))


# ---------------------------------------------------------------------------
# Tests: empty chunks → NOT_AVAILABLE_MSG
# ---------------------------------------------------------------------------

class TestGenerateCitedAnswerEmptyChunks(unittest.TestCase):

    def test_empty_chunks_returns_not_available(self):
        result = generate_cited_answer(query="What methods are used?", chunks=[])
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], NOT_AVAILABLE_MSG)
        self.assertEqual(result["sources_used"], 0)

    def test_empty_chunks_no_error(self):
        result = generate_cited_answer(query="test", chunks=[])
        self.assertIsNone(result["error"])


# ---------------------------------------------------------------------------
# Tests: _build_context_block
# ---------------------------------------------------------------------------

class TestBuildContextBlock(unittest.TestCase):

    def test_includes_paper_title(self):
        context = _build_context_block(SAMPLE_CHUNKS)
        self.assertIn("Attention Is All You Need", context)
        self.assertIn("BERT", context)

    def test_includes_page_ref(self):
        context = _build_context_block(SAMPLE_CHUNKS)
        self.assertIn("Page 3", context)
        self.assertIn("Page 2", context)

    def test_includes_source_text(self):
        context = _build_context_block(SAMPLE_CHUNKS)
        self.assertIn("Transformer", context)
        self.assertIn("bidirectional", context)

    def test_numbered_entries(self):
        context = _build_context_block(SAMPLE_CHUNKS)
        self.assertIn("[1]", context)
        self.assertIn("[2]", context)

    def test_empty_chunks_returns_empty_string(self):
        context = _build_context_block([])
        self.assertEqual(context, "")

    def test_chunk_without_page_ref(self):
        chunks = [{"paper_title": "Paper X", "page_ref": "", "source_text": "Some text."}]
        context = _build_context_block(chunks)
        self.assertIn("Paper X", context)
        self.assertIn("Some text.", context)


# ---------------------------------------------------------------------------
# Tests: _extract_citations_from_answer
# ---------------------------------------------------------------------------

class TestExtractCitationsFromAnswer(unittest.TestCase):

    def test_extracts_single_citation(self):
        answer = "Transformers are effective [Attention Is All You Need, Page 3]."
        cits = _extract_citations_from_answer(answer)
        self.assertIn("Attention Is All You Need, Page 3", cits)

    def test_extracts_multiple_citations(self):
        answer = "Method A [Paper A, Page 1] and method B [Paper B, Page 2]."
        cits = _extract_citations_from_answer(answer)
        self.assertEqual(len(cits), 2)

    def test_deduplicates_citations(self):
        answer = "[Paper A, Page 1] is useful. See also [Paper A, Page 1]."
        cits = _extract_citations_from_answer(answer)
        # Should be deduplicated
        self.assertEqual(len(cits), 1)

    def test_returns_empty_for_no_citations(self):
        cits = _extract_citations_from_answer("No citations here.")
        self.assertEqual(cits, [])


# ---------------------------------------------------------------------------
# Tests: model call and answer generation
# ---------------------------------------------------------------------------

class TestGenerateCitedAnswerModelCall(unittest.TestCase):

    @patch("tools.generate_cited_answer._call_watsonx_model", return_value=None)
    def test_dev_mode_returns_answer_with_citations(self, mock_model):
        result = generate_cited_answer(
            query="What methods are used?",
            chunks=SAMPLE_CHUNKS,
        )
        self.assertTrue(result["success"])
        self.assertNotEqual(result["answer"], "")
        self.assertGreater(result["sources_used"], 0)
        self.assertIsNone(result["error"])

    @patch("tools.generate_cited_answer._call_watsonx_model")
    def test_model_answer_used_when_available(self, mock_model):
        mock_answer = (
            "The Transformer uses attention mechanisms [Attention Is All You Need, Page 3]. "
            "BERT uses bidirectional pre-training [BERT: Pre-training of Deep Bidirectional Transformers, Page 2]."
        )
        mock_model.return_value = mock_answer

        result = generate_cited_answer(
            query="What methods are used?",
            chunks=SAMPLE_CHUNKS,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], mock_answer)
        self.assertGreaterEqual(len(result["citations"]), 1)

    @patch("tools.generate_cited_answer._call_watsonx_model", return_value="")
    def test_empty_model_response_falls_back_to_not_available(self, mock_model):
        result = generate_cited_answer(
            query="What are the findings?",
            chunks=SAMPLE_CHUNKS,
        )
        # Empty model response should be handled gracefully
        self.assertTrue(result["success"])

    @patch("tools.generate_cited_answer._call_watsonx_model")
    def test_model_exception_captured(self, mock_model):
        mock_model.side_effect = RuntimeError("API timeout")
        result = generate_cited_answer(
            query="test",
            chunks=SAMPLE_CHUNKS,
        )
        self.assertFalse(result["success"])
        self.assertIn("API timeout", result["error"])

    @patch("tools.generate_cited_answer._call_watsonx_model", return_value=None)
    def test_sources_used_count_matches_chunks(self, mock_model):
        result = generate_cited_answer(
            query="What are the methods?",
            chunks=SAMPLE_CHUNKS,
        )
        self.assertEqual(result["sources_used"], len(SAMPLE_CHUNKS))

    def test_model_id_defaults_to_env_or_granite(self):
        result = generate_cited_answer(query="test", chunks=[])
        self.assertIn("granite", result["model_id"].lower())


if __name__ == "__main__":
    unittest.main()
