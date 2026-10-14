"""
agents/research_qa_agent.py
────────────────────────────
Agent: research_qa_agent

Role:
    Answers user questions using ONLY retrieved research-paper chunks.
    Provides concise, accurate, well-structured answers with inline citations
    in the format [Paper Title, Page/Section].  Clearly states when requested
    information is not available in the uploaded papers.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

AGENT_NAME = "research_qa_agent"
AGENT_DESCRIPTION = (
    "Answers research questions using only uploaded paper content. "
    "Every answer includes inline citations [Paper Title, Page/Section]."
)

SYSTEM_PROMPT = """You are the question-answering expert for Research Companion.

Your responsibilities:
1. Accept a research question and a set of retrieved paper chunks.
2. Generate a concise, accurate, well-structured answer based ONLY on the provided chunks.
3. Add inline citations after every factual claim: [Paper Title, Page/Section].
4. If the chunks do not contain sufficient information, respond:
   "This information is not available in the uploaded research papers."

Rules:
- NEVER use general knowledge beyond the provided chunks.
- NEVER fabricate paper titles, author names, page numbers, or findings.
- NEVER provide medical, legal, financial, or safety-critical advice.
- Keep answers concise and student-friendly.
"""

TOOLS = [
    {
        "name": "generate_cited_answer",
        "description": "Generate a cited answer from retrieved research paper chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's research question.",
                },
                "chunks": {
                    "type": "array",
                    "description": "Retrieved paper chunks from rag_retrieval_agent.",
                },
            },
            "required": ["query", "chunks"],
        },
    }
]


class ResearchQAAgent:
    """
    Wraps the generate_cited_answer tool as a callable agent.
    """

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("WX_MODEL_QA", "ibm/granite-3-3-8b-instruct")
        try:
            from ..tools.generate_cited_answer import generate_cited_answer
        except ImportError:
            from tools.generate_cited_answer import generate_cited_answer

        self._generate = generate_cited_answer

    def run(self, query: str, chunks: list[dict]) -> dict[str, Any]:
        """
        Answer a research question with inline citations.

        Parameters
        ----------
        query : str
            The user's research question.
        chunks : list[dict]
            Retrieved paper chunks from RAGRetrievalAgent.

        Returns
        -------
        dict
            Answer result from generate_cited_answer.
        """
        logger.info("ResearchQAAgent: answering '%s' with %d chunks.", query[:80], len(chunks))
        return self._generate(query=query, chunks=chunks, model_id=self.model_id)

    @staticmethod
    def get_agent_definition() -> dict[str, Any]:
        return {
            "name": AGENT_NAME,
            "description": AGENT_DESCRIPTION,
            "model_id": os.getenv("WX_MODEL_QA", "ibm/granite-3-3-8b-instruct"),
            "system_prompt": SYSTEM_PROMPT,
            "tools": TOOLS,
        }
