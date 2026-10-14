"""
knowledge_base/research_papers_kb.py
──────────────────────────────────────
Knowledge Base: research_papers_kb

Schema stored per chunk:
    - paper_title      (str) : Canonical paper title for citations
    - authors          (str) : Semicolon-delimited author names
    - year             (str) : Publication year
    - page_ref         (str) : Page or section reference (e.g. "Page 3")
    - chunk_index      (int) : Position of this chunk within the paper
    - source_text      (str) : The chunk text (stored as ChromaDB document)
    - embedding        (vec) : watsonx.ai slate embedding vector

Backend: ChromaDB PersistentClient (local disk).
For production, swap _get_client() to use a managed vector store
(e.g. IBM Watson Discovery, Elasticsearch, or a hosted Chroma instance).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

KB_NAME = "research_papers_kb"
COLLECTION_NAME = "research_papers"
DEFAULT_PERSIST_DIR = os.getenv("KB_PERSIST_DIRECTORY", "./knowledge_base/chroma_db")


class ResearchPapersKB:
    """
    Thin wrapper around ChromaDB that provides domain-oriented methods
    for the research_papers knowledge base.

    All reads/writes go through this class so switching backends requires
    changing only this file.
    """

    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _get_collection(self):
        """Return (or lazily create) the ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("Connected to ChromaDB collection '%s' at '%s'.", COLLECTION_NAME, self.persist_dir)
        return self._collection

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """
        Insert or update chunks in the knowledge base.

        Parameters
        ----------
        ids : list[str]
            Unique chunk identifiers.
        texts : list[str]
            Raw text for each chunk.
        embeddings : list[list[float]]
            Embedding vectors (must match ids in length).
        metadatas : list[dict]
            Per-chunk metadata (paper_title, authors, year, page_ref, chunk_index).
        """
        col = self._get_collection()
        col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks into '%s'.", len(ids), COLLECTION_NAME)

    def delete_paper(self, paper_title: str) -> int:
        """
        Remove all chunks for a given paper title.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        col = self._get_collection()
        results = col.get(where={"paper_title": {"$eq": paper_title}})
        ids = results.get("ids", [])
        if ids:
            col.delete(ids=ids)
            logger.info("Deleted %d chunks for paper '%s'.", len(ids), paper_title)
        return len(ids)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        paper_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Run a vector similarity query against the knowledge base.

        Parameters
        ----------
        query_embedding : list[float]
            The embedded user query.
        top_k : int
            Number of results to return.
        paper_filter : str, optional
            Restrict results to a specific paper title.

        Returns
        -------
        dict
            ChromaDB query result (documents, metadatas, ids, distances).
        """
        col = self._get_collection()
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, max(1, col.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if paper_filter:
            kwargs["where"] = {"paper_title": {"$eq": paper_filter}}
        return col.query(**kwargs)

    def get_all_papers(self) -> list[str]:
        """Return a deduplicated list of all indexed paper titles."""
        col = self._get_collection()
        results = col.get(include=["metadatas"])
        titles = []
        seen: set[str] = set()
        for meta in results.get("metadatas", []):
            title = meta.get("paper_title", "")
            if title and title not in seen:
                titles.append(title)
                seen.add(title)
        return titles

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return self._get_collection().count()

    def is_empty(self) -> bool:
        """Return True if the knowledge base contains no chunks."""
        return self.count() == 0

    # ------------------------------------------------------------------
    # Admin operations
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Delete all chunks from the knowledge base.
        WARNING: This is irreversible.
        """
        try:
            import chromadb

            if self._client is not None:
                self._client.delete_collection(COLLECTION_NAME)
                self._collection = None
                logger.warning("Knowledge base '%s' has been reset (all data deleted).", KB_NAME)
        except Exception as exc:
            logger.error("Failed to reset knowledge base: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def get_kb_definition() -> dict[str, Any]:
        """Return the ADK-compatible knowledge base definition dict."""
        return {
            "name": KB_NAME,
            "description": (
                "Vector store for Research Companion. Stores text chunks, embeddings, "
                "and citation metadata (title, authors, year, page reference) for "
                "all uploaded research papers."
            ),
            "backend": "chromadb",
            "collection": COLLECTION_NAME,
            "schema": {
                "paper_title": "string — canonical paper title for citations",
                "authors": "string — semicolon-delimited author names",
                "year": "string — publication year",
                "page_ref": "string — page or section reference (e.g. 'Page 3')",
                "chunk_index": "integer — position of chunk within the paper",
                "source_text": "string — raw chunk text (ChromaDB document field)",
                "embedding": "float[] — watsonx.ai slate embedding vector",
            },
            "embedding_model": os.getenv(
                "WX_EMBEDDING_MODEL", "ibm/slate-125m-english-rtrvr"
            ),
        }
