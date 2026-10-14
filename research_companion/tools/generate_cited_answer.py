"""
tools/generate_cited_answer.py
───────────────────────────────
Tool: generate_cited_answer

Purpose:
    Generate a concise, accurate, well-structured answer to a user research
    question using ONLY the retrieved paper chunks.  Every factual claim
    includes an inline citation in the format [Paper Title, Page/Section].

    If the retrieved context does not contain the requested information,
    responds: "This information is not available in the uploaded research papers."

Input:
    query   (str)  : The user's research question.
    chunks  (list) : Top retrieved chunks from retrieve_relevant_chunks.

Output (dict):
    {
        "success": bool,
        "query": str,
        "answer": str,
        "citations": list[str],
        "sources_used": int,
        "model_id": str,
        "error": str | None
    }
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

NOT_AVAILABLE_MSG = (
    "This information is not available in the uploaded research papers."
)

_QA_SYSTEM_PROMPT = """You are a precise academic research assistant. Your task is to answer the user's question using ONLY the research paper excerpts provided in the context below.

Rules you MUST follow:
1. Base every factual claim on the provided context. Do NOT use general knowledge.
2. After each factual claim, add an inline citation in the format: [Paper Title, Page/Section].
3. If the context does not contain enough information to answer the question, respond with exactly: "This information is not available in the uploaded research papers."
4. Do NOT fabricate paper titles, author names, findings, page numbers, or any other content.
5. Keep the answer concise, well-structured, and easy for students to read.
6. Do NOT provide medical, legal, financial, or safety-critical advice.
7. Do NOT answer questions unrelated to the provided research papers.
"""

_QA_USER_TEMPLATE = """Context (retrieved research paper excerpts):
{context}

---
Question: {question}

Answer (with inline citations):"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("paper_title", "Unknown Paper")
        page_ref = chunk.get("page_ref", "")
        text = chunk.get("source_text", "").strip()
        ref_label = f"[{title}, {page_ref}]" if page_ref else f"[{title}]"
        lines.append(f"[{i}] {ref_label}\n{text}")
    return "\n\n".join(lines)


def _extract_citations_from_answer(answer: str) -> list[str]:
    """Extract all citation markers [ ... ] from the generated answer (deduplicated)."""
    import re
    return list(set(re.findall(r"\[([^\[\]]+)\]", answer)))


def _call_watsonx_model(prompt: str, model_id: str) -> str | None:
    """
    Call the IBM watsonx.ai foundation model for text generation.
    Returns None when credentials are absent (dev/test mode).
    """
    api_key = os.getenv("WATSONX_APIKEY")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    if not api_key or not project_id:
        logger.warning("Credentials not set; returning mock answer in dev/test mode.")
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
                "max_new_tokens": 1024,
                "temperature": 0.1,
                "repetition_penalty": 1.1,
            },
        )
        response = model.generate_text(prompt=prompt)
        return response.strip() if response else None
    except Exception as exc:
        logger.exception("watsonx.ai model call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def generate_cited_answer(
    query: str,
    chunks: list[dict],
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Generate a cited answer to the user query using retrieved paper chunks.

    Parameters
    ----------
    query : str
        The user's research question.
    chunks : list[dict]
        Retrieved chunks from retrieve_relevant_chunks (may be empty).
    model_id : str, optional
        watsonx.ai model ID (defaults to WX_MODEL_QA env var).

    Returns
    -------
    dict
        Answer text with inline citations, citation list, and metadata.
    """
    model_id = model_id or os.getenv("WX_MODEL_QA", "ibm/granite-3-3-8b-instruct")

    result: dict[str, Any] = {
        "success": False,
        "query": query,
        "answer": NOT_AVAILABLE_MSG,
        "citations": [],
        "sources_used": 0,
        "model_id": model_id,
        "error": None,
    }

    if not query or not query.strip():
        result["error"] = "Query is empty."
        return result

    # No chunks → nothing to answer from
    if not chunks:
        result["success"] = True
        result["answer"] = NOT_AVAILABLE_MSG
        return result

    try:
        context = _build_context_block(chunks)
        prompt = (
            _QA_SYSTEM_PROMPT
            + "\n\n"
            + _QA_USER_TEMPLATE.format(context=context, question=query)
        )

        raw_answer = _call_watsonx_model(prompt, model_id)

        if raw_answer is None:
            # Dev/test fallback: synthesise an answer from chunk text
            logger.info("Using mock answer synthesis for dev/test mode.")
            snippet_parts = []
            for chunk in chunks[:3]:
                title = chunk.get("paper_title", "Paper")
                page = chunk.get("page_ref", "")
                text = chunk.get("source_text", "")[:200].strip()
                ref = f"[{title}, {page}]" if page else f"[{title}]"
                snippet_parts.append(f"{text} {ref}")
            raw_answer = " ".join(snippet_parts) if snippet_parts else NOT_AVAILABLE_MSG

        # Safety guard: if model returns empty text, fall back
        if not raw_answer.strip():
            raw_answer = NOT_AVAILABLE_MSG

        citations = _extract_citations_from_answer(raw_answer)

        result.update({
            "success": True,
            "answer": raw_answer.strip(),
            "citations": list(set(citations)),
            "sources_used": len(chunks),
        })

    except Exception as exc:
        result["error"] = f"Answer generation failed: {exc}"
        logger.exception("generate_cited_answer failed.")

    return result
