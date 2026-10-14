"""
adk/tools/chunk_and_index_documents_tool.py
─────────────────────────────────────────────
ADK @tool wrapper for chunk_and_index_documents.

Import command (run from research_companion/):
  orchestrate tools import --kind python \\
    --file adk/tools/chunk_and_index_documents_tool.py \\
    --requirements-file adk/tools/requirements.txt \\
    --package-root .
"""

from __future__ import annotations

from ibm_watsonx_orchestrate.agent_builder.tools import tool
from tools.chunk_and_index_documents import chunk_and_index_documents as _chunk_and_index


@tool
def chunk_and_index_documents(
    full_text: str,
    paper_title: str,
    authors: list | None = None,
    year: str | None = None,
) -> dict:
    """
    Split a research paper's extracted text into overlapping chunks, generate
    semantic embeddings using IBM watsonx.ai (slate model), and index the chunks
    into the research_papers knowledge base.

    Call this tool after extract_pdf_text to make the paper searchable.
    The paper_title is used for all future citation references.

    Parameters
    ----------
    full_text : str
        Complete extracted text of the paper (from extract_pdf_text).
    paper_title : str
        Canonical paper title used for all citation references.
    authors : list of str, optional
        Author names. Pass an empty list if unavailable.
    year : str, optional
        Publication year (e.g. "2023"). Pass null if unavailable.

    Returns
    -------
    dict
        success (bool), paper_title (str), total_chunks (int),
        chunk_ids (list[str]), embedding_model (str),
        index_name (str), error (str|null)
    """
    return _chunk_and_index(
        full_text=full_text,
        paper_title=paper_title,
        authors=authors or [],
        year=year,
    )
