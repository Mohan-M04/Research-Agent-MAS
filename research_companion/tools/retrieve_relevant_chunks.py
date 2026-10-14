"""
tools/retrieve_relevant_chunks.py
──────────────────────────────────
Tool: retrieve_relevant_chunks

Purpose:
    Perform semantic retrieval over the indexed research-paper chunks stored
    in the ChromaDB knowledge base.  Returns the top-k most relevant chunks
    for a given user query, including citation metadata.

Input:
    query      (str) : The user's research question or search string.
    top_k      (int) : Number of top chunks to return (default 8).
    paper_filter (str | None): Optional paper title to restrict search scope.

Output (dict):
    {
        "success": bool,
        "query": str,
        "total_results": int,
        "chunks": [
            {
                "chunk_id": str,
                "paper_title": str,
                "authors": str,
                "year": str,
                "page_ref": str,
                "source_text": str,
                "similarity_score": float
            },
            ...
        ],
        "error": str | None
    }
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = os.getenv("KB_PERSIST_DIRECTORY", "./knowledge_base/chroma_db")
COLLECTION_NAME = "research_papers"
DEFAULT_TOP_K = int(os.getenv("TOP_K_CHUNKS", "8"))

# Module-level import so tests can patch `tools.retrieve_relevant_chunks.chromadb`.
# Silently set to None when the package is not installed; the runtime function
# will raise a friendly ImportError with install instructions.
try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embed_query(query: str) -> list[float] | None:
    """
    Generate a query embedding via watsonx.ai.
    Returns None in dev/test mode when credentials are absent.
    """
    model_id = os.getenv("WX_EMBEDDING_MODEL", "ibm/slate-125m-english-rtrvr")
    api_key = os.getenv("WATSONX_APIKEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    if not api_key or not project_id:
        logger.warning("Credentials not set; using keyword-based fallback for retrieval.")
        return None

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import Embeddings

        credentials = Credentials(url=url, api_key=api_key)
        embed_client = Embeddings(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
        )
        result = embed_client.embed_documents(texts=[query])
        if result and len(result) > 0:
            return result[0]
    except Exception as exc:
        logger.warning("Query embedding failed (%s); falling back to keyword search.", exc)

    return None


def _keyword_fallback_search(
    query: str,
    collection,
    top_k: int,
    paper_filter: str | None,
) -> dict:
    """
    Simple keyword-based retrieval when embeddings are unavailable.
    Scores chunks by counting query term occurrences (case-insensitive).
    """
    import re

    where = {"paper_title": {"$eq": paper_filter}} if paper_filter else None
    get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
    if where:
        get_kwargs["where"] = where

    all_data = collection.get(**get_kwargs)
    docs: list[str] = all_data.get("documents") or []
    metas: list[dict] = all_data.get("metadatas") or []
    ids: list[str] = all_data.get("ids") or []

    query_terms = re.findall(r"\w+", query.lower())

    scored = []
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        score = sum(doc_lower.count(term) for term in query_terms)
        scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    chunks = []
    for score, idx in top:
        meta = metas[idx] if idx < len(metas) else {}
        chunks.append({
            "chunk_id": ids[idx] if idx < len(ids) else "",
            "paper_title": meta.get("paper_title", "Unknown"),
            "authors": meta.get("authors", ""),
            "year": meta.get("year", ""),
            "page_ref": meta.get("page_ref", ""),
            "source_text": docs[idx],
            "similarity_score": round(score / max(1, len(query_terms) * 10), 4),
        })

    return chunks


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    paper_filter: str | None = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> dict[str, Any]:
    """
    Retrieve top-k semantically relevant chunks from the knowledge base.

    Parameters
    ----------
    query : str
        The user's research question.
    top_k : int
        Number of chunks to return.
    paper_filter : str, optional
        Restrict results to a specific paper title.
    persist_dir : str
        ChromaDB persistence directory.

    Returns
    -------
    dict
        Retrieved chunks with citation metadata and similarity scores.
    """
    result: dict[str, Any] = {
        "success": False,
        "query": query,
        "total_results": 0,
        "chunks": [],
        "error": None,
    }

    if not query or not query.strip():
        result["error"] = "Query string is empty."
        return result

    # Check directory before attempting any chromadb operation so the error
    # message is accurate even when chromadb is not installed.
    if not os.path.isdir(persist_dir):
        result["error"] = (
            f"Knowledge base not found at '{persist_dir}'. "
            "Please ingest PDFs first using chunk_and_index_documents."
        )
        return result

    try:
        if chromadb is None:
            raise ImportError("chromadb is not installed. Run: pip install chromadb")

        client = chromadb.PersistentClient(path=persist_dir)

        # Check collection exists
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME not in existing:
            result["error"] = (
                f"Collection '{COLLECTION_NAME}' does not exist. "
                "Please ingest PDFs first."
            )
            return result

        collection = client.get_collection(name=COLLECTION_NAME)
        total_docs = collection.count()

        if total_docs == 0:
            result["error"] = "Knowledge base is empty. Please ingest PDFs first."
            return result

        actual_top_k = min(top_k, total_docs)

        # Build optional where filter
        where = {"paper_title": {"$eq": paper_filter}} if paper_filter else None

        # Try vector search first
        query_embedding = _embed_query(query)

        if query_embedding is not None:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": actual_top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_kwargs["where"] = where

            qr = collection.query(**query_kwargs)

            raw_docs = qr.get("documents", [[]])[0]
            raw_metas = qr.get("metadatas", [[]])[0]
            raw_ids = qr.get("ids", [[]])[0]
            raw_dists = qr.get("distances", [[]])[0]

            chunks = []
            for i, doc in enumerate(raw_docs):
                meta = raw_metas[i] if i < len(raw_metas) else {}
                dist = raw_dists[i] if i < len(raw_dists) else 1.0
                # Convert cosine distance (0–2) → similarity (0–1)
                similarity = round(1.0 - dist / 2.0, 4)
                chunks.append({
                    "chunk_id": raw_ids[i] if i < len(raw_ids) else "",
                    "paper_title": meta.get("paper_title", "Unknown"),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year", ""),
                    "page_ref": meta.get("page_ref", ""),
                    "source_text": doc,
                    "similarity_score": max(0.0, similarity),
                })
        else:
            # Keyword fallback
            chunks = _keyword_fallback_search(query, collection, actual_top_k, paper_filter)

        result.update({
            "success": True,
            "total_results": len(chunks),
            "chunks": chunks,
        })

        logger.info(
            "Retrieved %d chunks for query: '%s'.",
            len(chunks), query[:80]
        )

    except ImportError:
        result["error"] = "chromadb is not installed. Run: pip install chromadb"
        logger.exception("chromadb import error.")
    except Exception as exc:
        result["error"] = f"Retrieval failed: {exc}"
        logger.exception("retrieve_relevant_chunks failed.")

    return result
