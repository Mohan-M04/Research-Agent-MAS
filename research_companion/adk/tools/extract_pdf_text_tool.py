"""
adk/tools/extract_pdf_text_tool.py
───────────────────────────────────
ADK @tool wrapper for extract_pdf_text.

Import command (run from research_companion/):
  orchestrate tools import --kind python \\
    --file adk/tools/extract_pdf_text_tool.py \\
    --requirements-file adk/tools/requirements.txt \\
    --package-root .
"""

from __future__ import annotations

from ibm_watsonx_orchestrate.agent_builder.tools import tool
from tools.extract_pdf_text import extract_pdf_text as _extract_pdf_text


@tool
def extract_pdf_text(pdf_path: str) -> dict:
    """
    Extract plain text, title, authors, publication year, and page-wise content
    from a research paper PDF file.

    Use this tool when the user uploads a PDF research paper and you need to
    extract its contents before indexing. Always call this before
    chunk_and_index_documents.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file on disk.

    Returns
    -------
    dict
        success (bool), pdf_path (str), paper_title (str|null),
        authors (list[str]), publication_year (str|null),
        total_pages (int), full_text (str),
        pages (list[{page_number, text, char_count}]),
        metadata (dict), error (str|null)
    """
    return _extract_pdf_text(pdf_path=pdf_path)
