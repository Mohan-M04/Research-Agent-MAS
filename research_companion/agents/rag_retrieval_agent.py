"""
agents/rag_retrieval_agent.py
──────────────────────────────
Agent: rag_retrieval_agent

Role:
    Performs semantic retrieval over the indexed research-paper chunks.
    Returns the top relevant chunks for a user query, including paper title,
    page/section reference, source text, and relevance score.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

AGENT_NAME = "rag_retrieval_agent"
AGENT_DESCRIPTION = (
    "Performs semantic retrieval over the indexed research-paper knowledge base. "
    "Returns top-k relevant chunks with citation metadata for any research query."
)

SYSTEM_PROMPT = """You are the retrieval specialist for Research Companion.

Your responsibilities:
1. Accept a research query from the coordinator or other agents.
2. Generate a query embedding and search the research_papers_kb knowledge base.
3. Return the top-k most relevant chunks.
4. Each chunk must include: paper title, page/section reference, source text, and similarity score.

Rules:
- Return only chunks from the indexed knowledge base.
- Never add, modify, or fabricate chunk content.
- If the knowledge base is empty, report this clearly.
- Aim for diversity: return chunks from multiple papers when available.
"""

TOOLS = [
    {
        "name": "retrieve_relevant_chunks",
        "description": "Semantic search over the research_papers_kb vector store.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research question or search string.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top chunks to return.",
                    "default": 8,
                },
                "paper_filter": {
                    "type": "string",
                    "description": "Optional: restrict search to a specific paper title.",
                },
            },
            "required": ["query"],
        },
    }
]


class RAGRetrievalAgent:
    """
    Wraps the retrieve_relevant_chunks tool as a callable agent.
    """

    def __init__(self) -> None:
        try:
            from ..tools.retrieve_relevant_chunks import retrieve_relevant_chunks
        except ImportError:
            from tools.retrieve_relevant_chunks import retrieve_relevant_chunks

        self._retrieve = retrieve_relevant_chunks

    def run(
        self,
        query: str,
        top_k: int = 8,
        paper_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve relevant chunks for a query.

        Parameters
        ----------
        query : str
            The research question to search for.
        top_k : int
            Number of top chunks to return.
        paper_filter : str, optional
            Restrict search to a specific paper title.

        Returns
        -------
        dict
            Retrieval result from retrieve_relevant_chunks.
        """
        logger.info("RAGRetrievalAgent: retrieving top_%d for '%s'.", top_k, query[:80])
        return self._retrieve(query=query, top_k=top_k, paper_filter=paper_filter)

    @staticmethod
    def get_agent_definition() -> dict[str, Any]:
        return {
            "name": AGENT_NAME,
            "description": AGENT_DESCRIPTION,
            "model_id": os.getenv("WX_MODEL_QA", "ibm/granite-3-3-8b-instruct"),
            "system_prompt": SYSTEM_PROMPT,
            "tools": TOOLS,
        }
