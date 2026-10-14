"""
agents/pdf_ingestion_agent.py
──────────────────────────────
Agent: pdf_ingestion_agent

Role:
    Handles uploaded PDF files.  Calls extract_pdf_text for each file,
    then calls chunk_and_index_documents to index the extracted content
    into the research_papers_kb knowledge base.

Returns ingestion status and the list of processed paper titles.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

AGENT_NAME = "pdf_ingestion_agent"
AGENT_DESCRIPTION = (
    "Processes uploaded PDF research papers: extracts text and metadata, "
    "cleans and chunks the content, and indexes it into the knowledge base."
)

SYSTEM_PROMPT = """You are the PDF ingestion specialist for Research Companion.

Your responsibilities:
1. Accept one or more PDF file paths from the coordinator.
2. For each file, extract the full text, paper title, authors, year, and page-wise content.
3. Clean the text and split it into overlapping chunks.
4. Generate embeddings and index all chunks into the research_papers_kb knowledge base.
5. Return a clear status report: which papers succeeded, which failed, and why.

Rules:
- Never skip a valid PDF.
- Report errors clearly without crashing the pipeline.
- Do not store any content outside the knowledge base.
"""

TOOLS = [
    {
        "name": "extract_pdf_text",
        "description": "Extract text and metadata from a single PDF file.",
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Path to the PDF file.",
                }
            },
            "required": ["pdf_path"],
        },
    },
    {
        "name": "chunk_and_index_documents",
        "description": "Chunk extracted text and index it into the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "full_text": {"type": "string"},
                "paper_title": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "year": {"type": "string"},
                "pages": {"type": "array"},
            },
            "required": ["full_text", "paper_title"],
        },
    },
]


class PDFIngestionAgent:
    """
    Orchestrates PDF extraction and indexing for all uploaded files.
    """

    def __init__(self) -> None:
        try:
            from ..tools.extract_pdf_text import extract_pdf_text
            from ..tools.chunk_and_index_documents import chunk_and_index_documents
        except ImportError:
            from tools.extract_pdf_text import extract_pdf_text
            from tools.chunk_and_index_documents import chunk_and_index_documents

        self._extract = extract_pdf_text
        self._index = chunk_and_index_documents

    def run(self, pdf_paths: list[str]) -> dict[str, Any]:
        """
        Process a list of PDF file paths.

        Parameters
        ----------
        pdf_paths : list[str]
            Paths to the PDF files to ingest.

        Returns
        -------
        dict
            {
                "success": bool,
                "processed_papers": list[str],
                "failed_files": list[dict],
                "total_chunks_indexed": int,
                "details": list[dict],
                "error": str | None
            }
        """
        result: dict[str, Any] = {
            "success": False,
            "processed_papers": [],
            "failed_files": [],
            "total_chunks_indexed": 0,
            "details": [],
            "error": None,
        }

        if not pdf_paths:
            result["error"] = "No PDF paths provided."
            return result

        total_chunks = 0
        processed = []
        failed = []
        details = []

        for pdf_path in pdf_paths:
            logger.info("Ingesting: %s", pdf_path)
            detail: dict[str, Any] = {"pdf_path": pdf_path}

            # Step 1: Extract
            extraction = self._extract(pdf_path)
            if not extraction.get("success"):
                err = extraction.get("error", "Extraction failed.")
                logger.warning("Extraction failed for '%s': %s", pdf_path, err)
                failed.append({"pdf_path": pdf_path, "error": err})
                detail["extraction_error"] = err
                details.append(detail)
                continue

            paper_title = extraction.get("paper_title") or os.path.basename(pdf_path)
            authors = extraction.get("authors", [])
            year = extraction.get("publication_year")
            full_text = extraction.get("full_text", "")
            pages = extraction.get("pages", [])

            detail.update({
                "paper_title": paper_title,
                "authors": authors,
                "year": year,
                "total_pages": extraction.get("total_pages", 0),
            })

            # Step 2: Chunk & Index
            indexing = self._index(
                full_text=full_text,
                paper_title=paper_title,
                authors=authors,
                year=year,
                pages=pages,
            )

            if not indexing.get("success"):
                err = indexing.get("error", "Indexing failed.")
                logger.warning("Indexing failed for '%s': %s", paper_title, err)
                failed.append({"pdf_path": pdf_path, "paper_title": paper_title, "error": err})
                detail["indexing_error"] = err
            else:
                chunks_added = indexing.get("total_chunks", 0)
                total_chunks += chunks_added
                processed.append(paper_title)
                detail["chunks_indexed"] = chunks_added
                detail["chunk_ids_sample"] = indexing.get("chunk_ids", [])[:3]
                logger.info("Indexed '%s': %d chunks.", paper_title, chunks_added)

            details.append(detail)

        result.update({
            "success": len(processed) > 0,
            "processed_papers": processed,
            "failed_files": failed,
            "total_chunks_indexed": total_chunks,
            "details": details,
        })

        if len(failed) > 0 and len(processed) == 0:
            result["error"] = f"All {len(failed)} file(s) failed to ingest."

        return result

    @staticmethod
    def get_agent_definition() -> dict[str, Any]:
        return {
            "name": AGENT_NAME,
            "description": AGENT_DESCRIPTION,
            "model_id": os.getenv("WX_MODEL_QA", "ibm/granite-3-3-8b-instruct"),
            "system_prompt": SYSTEM_PROMPT,
            "tools": TOOLS,
        }
