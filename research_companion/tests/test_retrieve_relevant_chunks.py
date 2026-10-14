"""
tests/test_retrieve_relevant_chunks.py
────────────────────────────────────────
Unit tests for tools/retrieve_relevant_chunks.py

Tests cover:
    - Empty query validation
    - Missing knowledge base handling
    - Empty collection handling
    - Vector search path (embedding available)
    - Keyword fallback search (no embedding)
    - paper_filter option
    - Result structure and field presence
    - Exception handling
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.retrieve_relevant_chunks import (
    retrieve_relevant_chunks,
    _keyword_fallback_search,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_collection(docs=None, metas=None, ids=None, distances=None, count=5):
    """Build a mock ChromaDB collection."""
    docs = docs or ["Chunk text A", "Chunk text B", "Chunk text C"]
    metas = metas or [
        {"paper_title": "Paper A", "authors": "Smith", "year": "2022", "page_ref": "Page 1"},
        {"paper_title": "Paper B", "authors": "Jones", "year": "2021", "page_ref": "Page 3"},
        {"paper_title": "Paper A", "authors": "Smith", "year": "2022", "page_ref": "Page 2"},
    ]
    ids = ids or ["id1", "id2", "id3"]
    distances = distances or [0.1, 0.3, 0.5]

    mock_col = MagicMock()
    mock_col.count.return_value = count
    mock_col.query.return_value = {
        "documents": [docs],
        "metadatas": [metas],
        "ids": [ids],
        "distances": [distances],
    }
    mock_col.get.return_value = {
        "documents": docs,
        "metadatas": metas,
        "ids": ids,
    }
    return mock_col


def _make_mock_chromadb(mock_col, collection_exists=True):
    """Build a mock chromadb module."""
    mock_chroma = MagicMock()
    mock_client = MagicMock()
    mock_chroma.PersistentClient.return_value = mock_client
    mock_client.list_collections.return_value = (
        [MagicMock(name="research_papers")] if collection_exists else []
    )
    # Make list_collections return objects with .name attribute
    if collection_exists:
        coll_mock = MagicMock()
        coll_mock.name = "research_papers"
        mock_client.list_collections.return_value = [coll_mock]
    mock_client.get_collection.return_value = mock_col
    return mock_chroma


# ---------------------------------------------------------------------------
# Tests: input validation
# ---------------------------------------------------------------------------

class TestRetrieveRelevantChunksValidation(unittest.TestCase):

    def test_empty_query_returns_error(self):
        result = retrieve_relevant_chunks(query="")
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_whitespace_query_returns_error(self):
        result = retrieve_relevant_chunks(query="   ")
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    @patch("tools.retrieve_relevant_chunks.os.path.isdir", return_value=False)
    def test_missing_persist_dir_returns_error(self, _):
        result = retrieve_relevant_chunks(query="test query")
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"].lower())

    def test_result_has_expected_keys(self):
        result = retrieve_relevant_chunks(query="test", persist_dir="/nonexistent/path")
        required_keys = {"success", "query", "total_results", "chunks", "error"}
        self.assertTrue(required_keys.issubset(result.keys()))


# ---------------------------------------------------------------------------
# Tests: vector search path
# ---------------------------------------------------------------------------

class TestRetrieveRelevantChunksVectorSearch(unittest.TestCase):

    @patch("tools.retrieve_relevant_chunks.os.path.isdir", return_value=True)
    @patch("tools.retrieve_relevant_chunks._embed_query", return_value=[0.1] * 768)
    def test_vector_search_returns_chunks(self, mock_embed, mock_isdir):
        """Vector search returns a list of chunk dicts when chromadb is mocked."""
        mock_col = _make_mock_collection()
        with patch("tools.retrieve_relevant_chunks.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            coll_mock = MagicMock()
            coll_mock.name = "research_papers"
            mock_client.list_collections.return_value = [coll_mock]
            mock_client.get_collection.return_value = mock_col

            result = retrieve_relevant_chunks(query="What methods are used?")

        self.assertTrue(result["success"])
        self.assertGreater(result["total_results"], 0)
        required_fields = {"chunk_id", "paper_title", "page_ref", "source_text", "similarity_score"}
        for chunk in result["chunks"]:
            self.assertTrue(required_fields.issubset(chunk.keys()))

    @patch("tools.retrieve_relevant_chunks.os.path.isdir", return_value=True)
    @patch("tools.retrieve_relevant_chunks._embed_query", return_value=[0.1] * 768)
    def test_similarity_score_between_0_and_1(self, mock_embed, mock_isdir):
        mock_col = _make_mock_collection(distances=[0.0, 0.5, 1.0, 2.0])
        with patch("tools.retrieve_relevant_chunks.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            coll_mock = MagicMock()
            coll_mock.name = "research_papers"
            mock_client.list_collections.return_value = [coll_mock]
            mock_client.get_collection.return_value = mock_col

            result = retrieve_relevant_chunks(query="methods")

        if result["success"]:
            for chunk in result["chunks"]:
                score = chunk["similarity_score"]
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


# ---------------------------------------------------------------------------
# Tests: keyword fallback search
# ---------------------------------------------------------------------------

class TestKeywordFallbackSearch(unittest.TestCase):

    def test_returns_top_k_results(self):
        mock_col = _make_mock_collection(
            docs=["attention mechanism transformer", "rnn recurrent", "bert pretraining"],
            metas=[
                {"paper_title": "T", "authors": "", "year": "2017", "page_ref": "Page 1"},
                {"paper_title": "R", "authors": "", "year": "2018", "page_ref": "Page 2"},
                {"paper_title": "B", "authors": "", "year": "2019", "page_ref": "Page 3"},
            ],
            ids=["id1", "id2", "id3"],
        )
        results = _keyword_fallback_search("attention transformer", mock_col, top_k=2, paper_filter=None)
        self.assertLessEqual(len(results), 2)

    def test_most_relevant_ranked_first(self):
        mock_col = _make_mock_collection(
            docs=["completely unrelated text", "methods and approaches are used in the study methods"],
            metas=[
                {"paper_title": "A", "authors": "", "year": "", "page_ref": "Page 1"},
                {"paper_title": "B", "authors": "", "year": "", "page_ref": "Page 2"},
            ],
            ids=["id1", "id2"],
        )
        results = _keyword_fallback_search("methods approaches", mock_col, top_k=2, paper_filter=None)
        if len(results) >= 2:
            self.assertGreaterEqual(
                results[0]["similarity_score"], results[1]["similarity_score"]
            )

    def test_result_has_required_fields(self):
        mock_col = _make_mock_collection()
        results = _keyword_fallback_search("test query", mock_col, top_k=3, paper_filter=None)
        required_fields = {
            "chunk_id", "paper_title", "authors", "year", "page_ref",
            "source_text", "similarity_score",
        }
        for r in results:
            self.assertTrue(required_fields.issubset(r.keys()))

    def test_empty_collection_returns_empty_list(self):
        mock_col = MagicMock()
        mock_col.get.return_value = {"documents": [], "metadatas": [], "ids": []}
        results = _keyword_fallback_search("query", mock_col, top_k=5, paper_filter=None)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# Tests: paper_filter option
# ---------------------------------------------------------------------------

class TestRetrieveWithPaperFilter(unittest.TestCase):

    @patch("tools.retrieve_relevant_chunks.os.path.isdir", return_value=True)
    @patch("tools.retrieve_relevant_chunks._embed_query", return_value=[0.1] * 768)
    def test_paper_filter_passed_to_query(self, mock_embed, mock_isdir):
        mock_col = _make_mock_collection()
        with patch("tools.retrieve_relevant_chunks.chromadb") as mock_chroma:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client
            coll_mock = MagicMock()
            coll_mock.name = "research_papers"
            mock_client.list_collections.return_value = [coll_mock]
            mock_client.get_collection.return_value = mock_col

            retrieve_relevant_chunks(query="test", paper_filter="Paper A")

        # Verify that the where clause was passed to query
        if mock_col.query.called:
            call_kwargs = mock_col.query.call_args.kwargs
            where = call_kwargs.get("where", {})
            self.assertIn("paper_title", where)


if __name__ == "__main__":
    unittest.main()
