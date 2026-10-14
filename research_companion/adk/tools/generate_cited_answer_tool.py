"""
adk/tools/generate_cited_answer_tool.py
─────────────────────────────────────────
ADK @tool wrapper for generate_cited_answer.

Import command (run from research_companion/):
  orchestrate tools import --kind python \\
    --file adk/tools/generate_cited_answer_tool.py \\
    --requirements-file adk/tools/requirements.txt \\
    --package-root .
"""

from __future__ import annotations

from ibm_watsonx_orchestrate.agent_builder.tools import tool
from tools.generate_cited_answer import generate_cited_answer as _generate_cited_answer


@tool
def generate_cited_answer(query: str, chunks: list) -> dict:
    """
    Generate a concise, accurate, citation-backed answer to a research question
    using ONLY the retrieved paper chunks provided.

    Every factual claim in the answer includes an inline citation in the format
    [Paper Title, Page/Section]. If the chunks do not contain sufficient
    information, the answer will be exactly:
    "This information is not available in the uploaded research papers."

    IMPORTANT: Never answer from general knowledge. Use only the provided chunks.

    Parameters
    ----------
    query : str
        The research question to answer (e.g. "What methods are used?").
    chunks : list of dict
        Retrieved chunks from retrieve_relevant_chunks. Each chunk must have:
        paper_title (str), page_ref (str), source_text (str).

    Returns
    -------
    dict
        success (bool), query (str), answer (str),
        citations (list[str]), sources_used (int),
        model_id (str), error (str|null)
    """
    return _generate_cited_answer(query=query, chunks=chunks)
