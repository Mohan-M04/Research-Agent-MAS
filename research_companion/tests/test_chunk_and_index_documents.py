"""
tests/test_chunk_and_index_documents.py
─────────────────────────────────────────
Unit tests for tools/chunk_and_index_documents.py

Tests cover:
    - Text splitting into chunks
    - Empty / missing input handling
    - Chunk ID determinism
    - Page reference assignment
    - ChromaDB upsert (mocked)
    - Embedding fallback in dev/test mode
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.chunk_and_index_documents import (
    chunk_and_index_documents,
    _split_text_into_chunks,
    _make_chunk_id,
    _assign_page_refs,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_TEXT = " ".join(["word"] * 1000)  # 1000 words ≈ 4000 chars
SAMPLE_TITLE = "Attention Is All You Need"
SAMPLE_PAGES = [
    {"page_number": 1, "text": " ".join(["word"] * 500)},
    {"page_number": 2, "text": " ".join(["word"] * 500)},
]


# ---------------------------------------------------------------------------
# Tests: _split_text_into_chunks
# ---------------------------------------------------------------------------

class TestSplitTextIntoChunks(unittest.TestCase):

    def test_splits_long_text(self):
        chunks = _split_text_into_chunks(SAMPLE_TEXT, chunk_size=128, chunk_overlap=16)
        self.assertGreater(len(chunks), 1)

    def test_each_chunk_non_empty(self):
        chunks = _split_text_into_chunks(SAMPLE_TEXT, chunk_size=128, chunk_overlap=16)
        for chunk in chunks:
            self.assertTrue(len(chunk.strip()) > 0)

    def test_empty_text_returns_empty_list(self):
        chunks = _split_text_into_chunks("")
        self.assertEqual(chunks, [])

    def test_whitespace_only_returns_empty_list(self):
        chunks = _split_text_into_chunks("   \n\n\t  ")
        self.assertEqual(chunks, [])

    def test_short_text_returns_single_chunk(self):
        short = "A short sentence."
        chunks = _split_text_into_chunks(short, chunk_size=512, chunk_overlap=64)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short)

    def test_chunk_size_respected_approximately(self):
        chunks = _split_text_into_chunks(SAMPLE_TEXT, chunk_size=64, chunk_overlap=8)
        # Each chunk should be at most roughly chunk_size*4*1.2 chars (with tolerance)
        max_expected = 64 * 4 * 1.5
        for chunk in chunks:
            self.assertLessEqual(len(chunk), max_expected, f"Chunk too long: {len(chunk)}")

    def test_overlap_creates_more_chunks_than_no_overlap(self):
        chunks_no_overlap = _split_text_into_chunks(SAMPLE_TEXT, chunk_size=128, chunk_overlap=0)
        chunks_with_overlap = _split_text_into_chunks(SAMPLE_TEXT, chunk_size=128, chunk_overlap=32)
        self.assertGreaterEqual(len(chunks_with_overlap), len(chunks_no_overlap))


# ---------------------------------------------------------------------------
# Tests: _make_chunk_id
# ---------------------------------------------------------------------------

class TestMakeChunkId(unittest.TestCase):

    def test_returns_string(self):
        cid = _make_chunk_id("Test Paper", 0)
        self.assertIsInstance(cid, str)

    def test_is_deterministic(self):
        cid1 = _make_chunk_id("Paper A", 5)
        cid2 = _make_chunk_id("Paper A", 5)
        self.assertEqual(cid1, cid2)

    def test_different_titles_produce_different_ids(self):
        cid1 = _make_chunk_id("Paper A", 0)
        cid2 = _make_chunk_id("Paper B", 0)
        self.assertNotEqual(cid1, cid2)

    def test_different_indices_produce_different_ids(self):
        cid1 = _make_chunk_id("Paper A", 0)
        cid2 = _make_chunk_id("Paper A", 1)
        self.assertNotEqual(cid1, cid2)

    def test_length_is_24(self):
        cid = _make_chunk_id("Test", 0)
        self.assertEqual(len(cid), 24)


# ---------------------------------------------------------------------------
# Tests: _assign_page_refs
# ---------------------------------------------------------------------------

class TestAssignPageRefs(unittest.TestCase):

    def test_assigns_correct_page(self):
        pages = [
            {"page_number": 1, "text": "This is content from page one."},
            {"page_number": 2, "text": "This is content from page two."},
        ]
        chunks = ["This is content from page one."]
        refs = _assign_page_refs(chunks, pages)
        self.assertEqual(refs, ["Page 1"])

    def test_unknown_when_no_match(self):
        pages = [{"page_number": 1, "text": "Something else entirely."}]
        chunks = ["Non-matching content xyz."]
        refs = _assign_page_refs(chunks, pages)
        self.assertEqual(refs, ["Unknown"])

    def test_returns_one_ref_per_chunk(self):
        pages = [{"page_number": 1, "text": "alpha beta gamma"}]
        chunks = ["alpha", "beta", "gamma", "delta"]
        refs = _assign_page_refs(chunks, pages)
        self.assertEqual(len(refs), len(chunks))

    def test_empty_chunks(self):
        refs = _assign_page_refs([], [{"page_number": 1, "text": "text"}])
        self.assertEqual(refs, [])


# ---------------------------------------------------------------------------
# Tests: chunk_and_index_documents (with ChromaDB mocked)
# ---------------------------------------------------------------------------

class TestChunkAndIndexDocuments(unittest.TestCase):

    def _make_mock_collection(self):
        """Return a mock ChromaDB collection with an upsert method."""
        mock_col = MagicMock()
        mock_col.upsert = MagicMock()
        return mock_col

    @patch("tools.chunk_and_index_documents._get_or_create_collection")
    @patch("tools.chunk_and_index_documents._get_watsonx_embeddings", return_value=None)
    def test_successful_indexing(self, mock_embed, mock_get_col):
        mock_col = self._make_mock_collection()
        mock_get_col.return_value = mock_col

        result = chunk_and_index_documents(
            full_text=SAMPLE_TEXT,
            paper_title=SAMPLE_TITLE,
        )

        self.assertTrue(result["success"])
        self.assertGreater(result["total_chunks"], 0)
        self.assertGreater(len(result["chunk_ids"]), 0)
        self.assertIsNone(result["error"])
        mock_col.upsert.assert_called_once()

    @patch("tools.chunk_and_index_documents._get_or_create_collection")
    @patch("tools.chunk_and_index_documents._get_watsonx_embeddings", return_value=None)
    def test_chunk_ids_match_total_chunks(self, mock_embed, mock_get_col):
        mock_col = self._make_mock_collection()
        mock_get_col.return_value = mock_col

        result = chunk_and_index_documents(
            full_text=SAMPLE_TEXT,
            paper_title=SAMPLE_TITLE,
        )
        self.assertEqual(result["total_chunks"], len(result["chunk_ids"]))

    def test_empty_text_returns_error(self):
        result = chunk_and_index_documents(full_text="", paper_title=SAMPLE_TITLE)
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_missing_title_returns_error(self):
        result = chunk_and_index_documents(full_text=SAMPLE_TEXT, paper_title="")
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_whitespace_title_returns_error(self):
        result = chunk_and_index_documents(full_text=SAMPLE_TEXT, paper_title="   ")
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    @patch("tools.chunk_and_index_documents._get_or_create_collection")
    @patch("tools.chunk_and_index_documents._get_watsonx_embeddings", return_value=None)
    def test_metadata_contains_paper_title(self, mock_embed, mock_get_col):
        mock_col = self._make_mock_collection()
        mock_get_col.return_value = mock_col

        chunk_and_index_documents(full_text=SAMPLE_TEXT, paper_title=SAMPLE_TITLE)

        call_kwargs = mock_col.upsert.call_args.kwargs
        metadatas = call_kwargs.get("metadatas", [])
        for meta in metadatas:
            self.assertEqual(meta["paper_title"], SAMPLE_TITLE)

    @patch("tools.chunk_and_index_documents._get_or_create_collection")
    @patch("tools.chunk_and_index_documents._get_watsonx_embeddings", return_value=None)
    def test_authors_and_year_stored_in_metadata(self, mock_embed, mock_get_col):
        mock_col = self._make_mock_collection()
        mock_get_col.return_value = mock_col

        chunk_and_index_documents(
            full_text=SAMPLE_TEXT,
            paper_title=SAMPLE_TITLE,
            authors=["Vaswani, A."],
            year="2017",
        )

        call_kwargs = mock_col.upsert.call_args.kwargs
        metadatas = call_kwargs.get("metadatas", [])
        for meta in metadatas:
            self.assertIn("Vaswani", meta["authors"])
            self.assertEqual(meta["year"], "2017")

    @patch("tools.chunk_and_index_documents._get_or_create_collection")
    @patch("tools.chunk_and_index_documents._get_watsonx_embeddings", return_value=None)
    def test_exception_during_upsert_captured(self, mock_embed, mock_get_col):
        mock_col = self._make_mock_collection()
        mock_col.upsert.side_effect = RuntimeError("DB write error")
        mock_get_col.return_value = mock_col

        result = chunk_and_index_documents(full_text=SAMPLE_TEXT, paper_title=SAMPLE_TITLE)
        self.assertFalse(result["success"])
        self.assertIn("DB write error", result["error"])

    @patch("tools.chunk_and_index_documents._get_or_create_collection")
    @patch("tools.chunk_and_index_documents._get_watsonx_embeddings")
    def test_real_embeddings_used_when_available(self, mock_embed, mock_get_col):
        n_chunks = len(_split_text_into_chunks(SAMPLE_TEXT))
        mock_embed.return_value = [[0.1] * 768 for _ in range(n_chunks)]
        mock_col = self._make_mock_collection()
        mock_get_col.return_value = mock_col

        result = chunk_and_index_documents(full_text=SAMPLE_TEXT, paper_title=SAMPLE_TITLE)
        self.assertTrue(result["success"])
        # Verify that the real embedding vectors were used (not zero-vectors)
        call_kwargs = mock_col.upsert.call_args.kwargs
        embs = call_kwargs.get("embeddings", [])
        self.assertTrue(any(v != 0.0 for vec in embs for v in vec))


if __name__ == "__main__":
    unittest.main()
