"""
tools/chunk_and_index_documents.py
───────────────────────────────────
Tool: chunk_and_index_documents

Purpose:
    Split extracted paper text into overlapping chunks, generate embeddings
    using IBM watsonx.ai (slate embedding model), and store them in the
    ChromaDB vector store that backs the research_papers_kb knowledge base.

Input:
    full_text    (str)  : Full extracted text from a single paper.
    paper_title  (str)  : Canonical title for citation references.
    authors      (list) : Author name strings (may be empty).
    year         (str)  : Publication year string (may be None).
    pages        (list) : Page-wise dicts from extract_pdf_text (optional).
    chunk_size   (int)  : Target token count per chunk  (default 512).
    chunk_overlap(int)  : Token overlap between chunks  (default 64).

Output (dict):
    {
        "success": bool,
        "paper_title": str,
        "total_chunks": int,
        "chunk_ids": list[str],
        "embedding_model": str,
        "index_name": str,
        "error": str | None
    }
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (can be overridden via environment or explicit args)
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
DEFAULT_PERSIST_DIR = os.getenv("KB_PERSIST_DIRECTORY", "./knowledge_base/chroma_db")
COLLECTION_NAME = "research_papers"


# ---------------------------------------------------------------------------
# Internal: text splitter (character-based, no LangChain dependency required)
# ---------------------------------------------------------------------------

def _split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping chunks.

    Uses a simple character-level sliding window.  Characters-per-token ratio
    is approximated at 4 (typical for English prose).  This avoids a hard
    dependency on tiktoken at runtime while remaining accurate enough for
    retrieval purposes.
    """
    # Convert token counts to character counts (≈ 4 chars / token)
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4

    if not text or len(text.strip()) == 0:
        return []

    # Short-circuit: the entire text fits inside a single chunk.
    if len(text) <= char_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Prefer splitting at paragraph / sentence boundaries
    separators = ["\n\n", "\n", ". ", "! ", "? ", " "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + char_size, len(text))

        # If we haven't reached the end of the document, try to find a clean break
        if end < len(text):
            best_break = end
            for sep in separators:
                idx = text.rfind(sep, start, end)
                if idx != -1 and idx > start:
                    best_break = idx + len(sep)
                    break
            end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start, keeping overlap
        start = max(end - char_overlap, start + 1)

    return chunks


# ---------------------------------------------------------------------------
# Internal: embed with watsonx.ai
# ---------------------------------------------------------------------------

def _get_watsonx_embeddings(
    texts: list[str],
    model_id: str | None = None,
) -> list[list[float]] | None:
    """
    Generate embeddings via the IBM watsonx.ai Python SDK.

    Returns None if the SDK / credentials are unavailable (so the rest of
    indexing can still proceed with empty vectors in dev/test mode).
    """
    if not texts:
        return []

    model_id = model_id or os.getenv("WX_EMBEDDING_MODEL", "ibm/slate-125m-english-rtrvr")
    api_key = os.getenv("WATSONX_APIKEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    if not api_key or not project_id:
        logger.warning(
            "WATSONX_APIKEY or WATSONX_PROJECT_ID not set — skipping real embeddings; "
            "zero vectors will be stored (dev/test mode)."
        )
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
        response = embed_client.embed_documents(texts=texts)
        # ibm-watsonx-ai SDK returns a list of embedding vectors
        return response
    except Exception as exc:
        logger.warning("watsonx.ai embedding call failed (%s); falling back to zero vectors.", exc)
        return None


# ---------------------------------------------------------------------------
# Internal: ChromaDB persistence
# ---------------------------------------------------------------------------

def _get_or_create_collection(persist_dir: str):
    """Return (or create) a ChromaDB PersistentClient collection."""
    try:
        import chromadb

        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return collection
    except ImportError:
        raise ImportError("chromadb is not installed. Run: pip install chromadb")


def _make_chunk_id(paper_title: str, chunk_index: int) -> str:
    """Generate a deterministic, collision-resistant chunk ID."""
    raw = f"{paper_title}::chunk::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Internal: assign page / section reference to a chunk
# ---------------------------------------------------------------------------

def _assign_page_refs(
    chunks: list[str],
    pages: list[dict],
) -> list[str]:
    """
    Map each chunk back to the most likely page number by matching substring.
    Returns a list of page-reference strings (e.g. "Page 3").
    """
    refs = []
    for chunk in chunks:
        # Take first 120 chars of chunk as anchor
        anchor = chunk[:120].strip()
        matched_page = "Unknown"
        for page in pages:
            if anchor in page.get("text", ""):
                matched_page = f"Page {page['page_number']}"
                break
        refs.append(matched_page)
    return refs


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def chunk_and_index_documents(
    full_text: str,
    paper_title: str,
    authors: list[str] | None = None,
    year: str | None = None,
    pages: list[dict] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> dict[str, Any]:
    """
    Chunk, embed, and index a research paper into the knowledge base.

    Parameters
    ----------
    full_text : str
        Complete extracted text of the paper.
    paper_title : str
        Canonical paper title for citation and retrieval metadata.
    authors : list[str], optional
        List of author name strings.
    year : str, optional
        Publication year.
    pages : list[dict], optional
        Page-wise dicts from extract_pdf_text for page-ref assignment.
    chunk_size : int
        Target chunk size in tokens (≈ 4 chars/token).
    chunk_overlap : int
        Overlap in tokens between consecutive chunks.
    persist_dir : str
        Directory where ChromaDB stores its data.

    Returns
    -------
    dict
        Indexing result with chunk IDs, counts, and status flags.
    """
    result: dict[str, Any] = {
        "success": False,
        "paper_title": paper_title,
        "total_chunks": 0,
        "chunk_ids": [],
        "embedding_model": os.getenv("WX_EMBEDDING_MODEL", "ibm/slate-125m-english-rtrvr"),
        "index_name": COLLECTION_NAME,
        "error": None,
    }

    if not full_text or not full_text.strip():
        result["error"] = "full_text is empty; nothing to index."
        logger.error(result["error"])
        return result

    if not paper_title or not paper_title.strip():
        result["error"] = "paper_title is required for indexing."
        logger.error(result["error"])
        return result

    authors = authors or []
    pages = pages or []

    try:
        # 1. Split text into chunks
        chunks = _split_text_into_chunks(full_text, chunk_size, chunk_overlap)
        if not chunks:
            result["error"] = "Text splitting produced no chunks."
            return result

        logger.info("Split '%s' into %d chunks.", paper_title, len(chunks))

        # 2. Assign page references
        page_refs = _assign_page_refs(chunks, pages) if pages else [f"Chunk {i+1}" for i in range(len(chunks))]

        # 3. Generate embeddings
        embeddings = _get_watsonx_embeddings(chunks)
        embedding_available = embeddings is not None and len(embeddings) == len(chunks)

        if not embedding_available:
            # Use zero-vectors as placeholder (dev/test mode)
            dummy_dim = 768
            embeddings = [[0.0] * dummy_dim for _ in chunks]
            logger.warning("Using zero-vector placeholders for embeddings.")

        # 4. Build metadata for each chunk
        chunk_ids = [_make_chunk_id(paper_title, i) for i in range(len(chunks))]
        metadatas = [
            {
                "paper_title": paper_title,
                "authors": "; ".join(authors) if authors else "",
                "year": year or "",
                "page_ref": page_refs[i],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # 5. Upsert into ChromaDB
        collection = _get_or_create_collection(persist_dir)
        collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Indexed %d chunks for '%s' into collection '%s'.",
            len(chunks), paper_title, COLLECTION_NAME,
        )

        result.update({
            "success": True,
            "total_chunks": len(chunks),
            "chunk_ids": chunk_ids,
        })

    except Exception as exc:
        result["error"] = f"Indexing failed: {exc}"
        logger.exception("chunk_and_index_documents failed for '%s'.", paper_title)

    return result
