"""
adk/tools/generate_literature_review_tool.py
──────────────────────────────────────────────
ADK @tool wrapper for generate_literature_review.

Import command (run from research_companion/):
  orchestrate tools import --kind python \\
    --file adk/tools/generate_literature_review_tool.py \\
    --requirements-file adk/tools/requirements.txt \\
    --package-root .
"""

from __future__ import annotations

from ibm_watsonx_orchestrate.agent_builder.tools import tool
from tools.generate_literature_review import generate_literature_review as _generate_review


@tool
def generate_literature_review(chunks: list) -> dict:
    """
    Generate a structured, academic literature review from retrieved research
    paper chunks covering all six required sections.

    The review is organised into:
      1. Background
      2. Methods and Approaches
      3. Key Findings
      4. Limitations
      5. Research Gaps
      6. Future Research Directions

    Every major claim includes at least one inline citation [Paper Title, Page/Section].
    If a section has no supporting evidence, it will note that explicitly.

    IMPORTANT: Use ONLY the provided chunks. Do NOT add general knowledge.

    Parameters
    ----------
    chunks : list of dict
        Retrieved chunks from retrieve_relevant_chunks across all uploaded papers.
        Each chunk must have: paper_title (str), page_ref (str), source_text (str).

    Returns
    -------
    dict
        success (bool), literature_review (str),
        sections ({background, methods_and_approaches, key_findings,
                   limitations, research_gaps, future_research_directions}),
        citations (list[str]), papers_covered (list[str]),
        model_id (str), error (str|null)
    """
    return _generate_review(chunks=chunks)
