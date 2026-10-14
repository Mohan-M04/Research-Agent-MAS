"""
agents/literature_review_agent.py
───────────────────────────────────
Agent: literature_review_agent

Role:
    Generates a structured literature review from retrieved content across
    all uploaded research papers.  Every major claim includes at least one
    citation in the format [Paper Title, Page/Section].

Literature review sections:
    1. Background
    2. Methods and Approaches
    3. Key Findings
    4. Limitations
    5. Research Gaps
    6. Future Research Directions
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

AGENT_NAME = "literature_review_agent"
AGENT_DESCRIPTION = (
    "Generates a structured, six-section literature review with citations "
    "from retrieved research paper chunks across all uploaded papers."
)

SYSTEM_PROMPT = """You are the literature review writer for Research Companion.

Your responsibilities:
1. Accept retrieved chunks from all uploaded research papers.
2. Synthesise the information into a structured six-section literature review:
   1. Background
   2. Methods and Approaches
   3. Key Findings
   4. Limitations
   5. Research Gaps
   6. Future Research Directions
3. Every major claim must include at least one inline citation: [Paper Title, Page/Section].
4. If a section has no supporting evidence, write:
   "No information available in the uploaded papers for this section."

Rules:
- Synthesise across papers; do not just repeat chunks verbatim.
- Every factual claim must be grounded in the retrieved context.
- Never fabricate content, titles, findings, or references.
- Write in clear academic prose suitable for students and researchers.
"""

TOOLS = [
    {
        "name": "generate_literature_review",
        "description": "Generate a structured literature review with citations.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "description": "Retrieved chunks from all uploaded research papers.",
                }
            },
            "required": ["chunks"],
        },
    }
]


class LiteratureReviewAgent:
    """
    Wraps the generate_literature_review tool as a callable agent.
    """

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv(
            "WX_MODEL_SUMMARIZE", "ibm/granite-3-3-8b-instruct"
        )
        try:
            from ..tools.generate_literature_review import generate_literature_review
        except ImportError:
            from tools.generate_literature_review import generate_literature_review

        self._generate = generate_literature_review

    def run(self, chunks: list[dict]) -> dict[str, Any]:
        """
        Generate a structured literature review.

        Parameters
        ----------
        chunks : list[dict]
            Retrieved chunks from all uploaded papers.

        Returns
        -------
        dict
            Literature review result from generate_literature_review.
        """
        logger.info(
            "LiteratureReviewAgent: generating review from %d chunks.", len(chunks)
        )
        return self._generate(chunks=chunks, model_id=self.model_id)

    @staticmethod
    def get_agent_definition() -> dict[str, Any]:
        return {
            "name": AGENT_NAME,
            "description": AGENT_DESCRIPTION,
            "model_id": os.getenv("WX_MODEL_SUMMARIZE", "ibm/granite-3-3-8b-instruct"),
            "system_prompt": SYSTEM_PROMPT,
            "tools": TOOLS,
        }
