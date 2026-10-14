"""
adk/tools/retrieve_relevant_chunks_tool.py
────────────────────────────────────────────
ADK @tool wrapper for retrieve_relevant_chunks.

Import command (run from research_companion/):
  orchestrate tools import --kind python \\
    --file adk/tools/retrieve_relevant_chunks_tool.py \\
    --requirements-file adk/tools/requirements.txt \\
    --package-root .
"""

from __future__ import annotations

from ibm_watsonx_orchestrate.agent_builder.tools import tool
from tools.retrieve_relevant_chunks import retrieve_relevant_chunks as _retrieve


@tool
def retrieve_relevant_chunks(
    query: str,
    top_k: int = 8,
    paper_filter: str | None = None,
) -> dict:
    """
    Perform semantic retrieval over all indexed research paper chunks and return
    the most relevant passages for the given query.

    Always call this tool before answering any research question or generating
    a literature review. Use top_k=20 for literature reviews to get broad coverage.

    Parameters
    ----------
    query : str
        The user's research question or topic
        (e.g. "methods used in the papers").
    top_k : int, optional
        Number of top chunks to return. Default 8 for Q&A; use 20 for reviews.
    paper_filter : str, optional
        Restrict results to a specific paper title. Null searches all papers.

    Returns
    -------
    dict
        success (bool), query (str), total_results (int),
        chunks (list[{chunk_id, paper_title, authors, year,
                       page_ref, source_text, similarity_score}]),
        error (str|null)
    """
    return _retrieve(query=query, top_k=top_k, paper_filter=paper_filter)
