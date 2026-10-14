"""
tools/generate_literature_review.py
─────────────────────────────────────
Tool: generate_literature_review

Purpose:
    Generate a structured, multi-section literature review from retrieved
    research-paper chunks.  Every major claim must include at least one
    inline citation in the format [Paper Title, Page/Section].

    Literature review sections:
        1. Background
        2. Methods and Approaches
        3. Key Findings
        4. Limitations
        5. Research Gaps
        6. Future Research Directions

Input:
    chunks  (list) : Retrieved chunks from all uploaded papers.

Output (dict):
    {
        "success": bool,
        "literature_review": str,
        "sections": {
            "background": str,
            "methods_and_approaches": str,
            "key_findings": str,
            "limitations": str,
            "research_gaps": str,
            "future_research_directions": str
        },
        "citations": list[str],
        "papers_covered": list[str],
        "model_id": str,
        "error": str | None
    }
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

NOT_AVAILABLE_MSG = (
    "This information is not available in the uploaded research papers."
)

SECTION_KEYS = [
    "background",
    "methods_and_approaches",
    "key_findings",
    "limitations",
    "research_gaps",
    "future_research_directions",
]

SECTION_LABELS = {
    "background": "1. Background",
    "methods_and_approaches": "2. Methods and Approaches",
    "key_findings": "3. Key Findings",
    "limitations": "4. Limitations",
    "research_gaps": "5. Research Gaps",
    "future_research_directions": "6. Future Research Directions",
}

_LIT_REVIEW_SYSTEM_PROMPT = """You are an expert academic research writer. Your task is to write a structured literature review using ONLY the research paper excerpts provided in the context below.

Rules you MUST follow:
1. Use ONLY information from the provided context. Do NOT add general knowledge.
2. Every major claim must include at least one inline citation: [Paper Title, Page/Section].
3. Structure your review with exactly these six numbered sections:
   1. Background
   2. Methods and Approaches
   3. Key Findings
   4. Limitations
   5. Research Gaps
   6. Future Research Directions
4. If a section has no supporting evidence in the context, write: "No information available in the uploaded papers for this section."
5. Do NOT fabricate paper titles, author names, findings, page numbers, or references.
6. Write in clear, academic prose suitable for students and researchers.
7. Each section should be 2–5 sentences and include at least one citation when evidence is available.
"""

_LIT_REVIEW_USER_TEMPLATE = """Context (retrieved research paper excerpts from all uploaded papers):
{context}

---
Write a structured literature review based solely on the context above.
Include every section listed below, with inline citations after each major claim.

Format:
## 1. Background
...

## 2. Methods and Approaches
...

## 3. Key Findings
...

## 4. Limitations
...

## 5. Research Gaps
...

## 6. Future Research Directions
...
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("paper_title", "Unknown Paper")
        page_ref = chunk.get("page_ref", "")
        text = chunk.get("source_text", "").strip()
        ref_label = f"[{title}, {page_ref}]" if page_ref else f"[{title}]"
        lines.append(f"[{i}] {ref_label}\n{text}")
    return "\n\n".join(lines)


def _get_unique_papers(chunks: list[dict]) -> list[str]:
    """Return a deduplicated list of paper titles present in the chunks."""
    seen = {}
    for chunk in chunks:
        title = chunk.get("paper_title", "Unknown")
        if title not in seen:
            seen[title] = True
    return list(seen.keys())


def _extract_citations(text: str) -> list[str]:
    """Extract all [Citation] markers from a text."""
    return list(set(re.findall(r"\[([^\[\]]+)\]", text)))


def _call_watsonx_model(prompt: str, model_id: str) -> str | None:
    """
    Call the IBM watsonx.ai foundation model for literature review generation.
    Returns None when credentials are absent (dev/test mode).
    """
    api_key = os.getenv("WATSONX_APIKEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    if not api_key or not project_id:
        logger.warning("Credentials not set; returning mock literature review in dev/test mode.")
        return None

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        credentials = Credentials(url=url, api_key=api_key)
        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params={
                "max_new_tokens": 2048,
                "temperature": 0.2,
                "repetition_penalty": 1.05,
            },
        )
        response = model.generate_text(prompt=prompt)
        return response.strip() if response else None
    except Exception as exc:
        logger.exception("watsonx.ai model call failed: %s", exc)
        return None


def _parse_sections(review_text: str) -> dict[str, str]:
    """
    Parse the model output into individual section strings.
    Falls back to NOT_AVAILABLE_MSG for any missing section.
    """
    section_patterns = {
        "background": r"(?:##\s*1\.\s*Background)(.*?)(?=##\s*2\.|$)",
        "methods_and_approaches": r"(?:##\s*2\.\s*Methods and Approaches)(.*?)(?=##\s*3\.|$)",
        "key_findings": r"(?:##\s*3\.\s*Key Findings)(.*?)(?=##\s*4\.|$)",
        "limitations": r"(?:##\s*4\.\s*Limitations)(.*?)(?=##\s*5\.|$)",
        "research_gaps": r"(?:##\s*5\.\s*Research Gaps)(.*?)(?=##\s*6\.|$)",
        "future_research_directions": r"(?:##\s*6\.\s*Future Research Directions)(.*?)$",
    }

    sections = {}
    for key, pattern in section_patterns.items():
        match = re.search(pattern, review_text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()
        else:
            sections[key] = NOT_AVAILABLE_MSG

    return sections


def _build_mock_review(chunks: list[dict]) -> str:
    """
    Construct a minimal mock literature review for dev/test mode.
    Uses chunk snippets as synthetic content.
    """
    by_paper: dict[str, list[str]] = {}
    for chunk in chunks:
        title = chunk.get("paper_title", "Unknown")
        page = chunk.get("page_ref", "")
        text = chunk.get("source_text", "")[:180].strip()
        by_paper.setdefault(title, []).append(f"{text} [{title}, {page}]")

    def sample(paper_chunks: list[str]) -> str:
        return " ".join(paper_chunks[:2]) if paper_chunks else NOT_AVAILABLE_MSG

    all_chunks_flat = [t for ts in by_paper.values() for t in ts]
    general = sample(all_chunks_flat)

    lines = [
        "## 1. Background",
        general,
        "",
        "## 2. Methods and Approaches",
        sample(all_chunks_flat[1:]),
        "",
        "## 3. Key Findings",
        sample(all_chunks_flat[2:]),
        "",
        "## 4. Limitations",
        NOT_AVAILABLE_MSG if not all_chunks_flat else sample(all_chunks_flat),
        "",
        "## 5. Research Gaps",
        NOT_AVAILABLE_MSG,
        "",
        "## 6. Future Research Directions",
        NOT_AVAILABLE_MSG,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def generate_literature_review(
    chunks: list[dict],
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Generate a structured literature review from retrieved research-paper chunks.

    Parameters
    ----------
    chunks : list[dict]
        Retrieved chunks from retrieve_relevant_chunks across all papers.
    model_id : str, optional
        watsonx.ai model ID (defaults to WX_MODEL_SUMMARIZE env var).

    Returns
    -------
    dict
        Structured literature review with per-section text, citations, and metadata.
    """
    model_id = model_id or os.getenv("WX_MODEL_SUMMARIZE", "ibm/granite-3-3-8b-instruct")

    result: dict[str, Any] = {
        "success": False,
        "literature_review": NOT_AVAILABLE_MSG,
        "sections": {key: NOT_AVAILABLE_MSG for key in SECTION_KEYS},
        "citations": [],
        "papers_covered": [],
        "model_id": model_id,
        "error": None,
    }

    if not chunks:
        result["success"] = True
        result["literature_review"] = (
            "No research papers have been indexed. "
            "Please upload and ingest PDFs before requesting a literature review."
        )
        return result

    try:
        papers_covered = _get_unique_papers(chunks)
        context = _build_context_block(chunks)
        prompt = (
            _LIT_REVIEW_SYSTEM_PROMPT
            + "\n\n"
            + _LIT_REVIEW_USER_TEMPLATE.format(context=context)
        )

        raw_review = _call_watsonx_model(prompt, model_id)

        if raw_review is None:
            # Dev/test fallback
            logger.info("Using mock literature review for dev/test mode.")
            raw_review = _build_mock_review(chunks)

        if not raw_review.strip():
            raw_review = NOT_AVAILABLE_MSG

        sections = _parse_sections(raw_review)
        citations = _extract_citations(raw_review)

        # Assemble formatted full-text review
        formatted_sections = []
        for key in SECTION_KEYS:
            label = SECTION_LABELS[key]
            formatted_sections.append(f"## {label}\n\n{sections.get(key, NOT_AVAILABLE_MSG)}")

        full_review = "\n\n".join(formatted_sections)

        result.update({
            "success": True,
            "literature_review": full_review,
            "sections": sections,
            "citations": citations,
            "papers_covered": papers_covered,
        })

        logger.info(
            "Generated literature review covering %d papers (%d citations).",
            len(papers_covered), len(citations)
        )

    except Exception as exc:
        result["error"] = f"Literature review generation failed: {exc}"
        logger.exception("generate_literature_review failed.")

    return result
