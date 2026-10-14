"""
agents/research_coordinator.py
────────────────────────────────
Agent: research_coordinator

Role:
    Main user-facing coordinator agent.  Parses incoming requests, decides
    which sub-agent or tool to invoke, and returns final responses
    (with citations) to the user.

Responsibilities:
    - Detect intent: PDF upload, research question, paper summary, or
      literature-review request.
    - Route to the correct sub-agent.
    - Merge and return the final cited response.
    - Refuse out-of-scope questions and state the reason politely.

Registered tools (called via tool-use / function-calling):
    - pdf_ingestion_agent.run(pdf_paths)
    - rag_retrieval_agent.run(query)
    - research_qa_agent.run(query, chunks)
    - literature_review_agent.run(chunks)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent definition (watsonx Orchestrate ADK style)
# ---------------------------------------------------------------------------

AGENT_NAME = "research_coordinator"
AGENT_DESCRIPTION = (
    "Main user-facing coordinator for Research Companion. "
    "Understands user intent (PDF upload, question, or literature review) "
    "and routes requests to the appropriate sub-agent."
)

SYSTEM_PROMPT = """You are the Research Companion coordinator. You help students, researchers, and professionals understand research papers.

You have access to the following sub-agents:
1. pdf_ingestion_agent — call this when the user uploads PDF files.
2. rag_retrieval_agent — call this to retrieve relevant text chunks for a user query.
3. research_qa_agent — call this to answer a research question using retrieved chunks.
4. literature_review_agent — call this to generate a structured literature review.

Your routing rules:
- If the user uploads PDF files → call pdf_ingestion_agent first, then confirm ingestion.
- If the user asks a research question → call rag_retrieval_agent, then research_qa_agent.
- If the user asks for a literature review → call rag_retrieval_agent (broad query), then literature_review_agent.
- If the user asks an unrelated question (not about the uploaded papers) → politely state:
  "This information is not available in the uploaded research papers."

Always return the final answer with inline citations in the format [Paper Title, Page/Section].
Never fabricate content, titles, authors, findings, or references.
"""

# Tool definitions for watsonx Orchestrate ADK function-calling registration
TOOLS = [
    {
        "name": "route_to_pdf_ingestion",
        "description": "Ingest one or more uploaded PDF research papers into the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to the uploaded PDFs.",
                }
            },
            "required": ["pdf_paths"],
        },
    },
    {
        "name": "route_to_rag_retrieval",
        "description": "Retrieve relevant research paper chunks for a user query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research question or topic to search for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top chunks to retrieve (default 8).",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "route_to_research_qa",
        "description": "Answer a research question using retrieved paper chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research question to answer.",
                },
                "chunks": {
                    "type": "array",
                    "description": "Retrieved chunks from rag_retrieval_agent.",
                },
            },
            "required": ["query", "chunks"],
        },
    },
    {
        "name": "route_to_literature_review",
        "description": "Generate a structured literature review from retrieved paper chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "chunks": {
                    "type": "array",
                    "description": "Retrieved chunks from all uploaded papers.",
                }
            },
            "required": ["chunks"],
        },
    },
]


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ResearchCoordinatorAgent:
    """
    watsonx Orchestrate ADK-compatible coordinator agent.

    Usage
    -----
    agent = ResearchCoordinatorAgent()
    response = agent.run(user_message="What methods are used in these papers?")
    """

    def __init__(
        self,
        model_id: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        self.model_id = model_id or os.getenv("WX_MODEL_QA", "ibm/granite-3-3-8b-instruct")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._sub_agents: dict[str, Any] = {}

    def register_sub_agents(
        self,
        pdf_ingestion,
        rag_retrieval,
        research_qa,
        literature_review,
    ) -> None:
        """Inject sub-agent instances at runtime."""
        self._sub_agents = {
            "pdf_ingestion": pdf_ingestion,
            "rag_retrieval": rag_retrieval,
            "research_qa": research_qa,
            "literature_review": literature_review,
        }

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_intent(message: str, pdf_paths: list[str]) -> str:
        """
        Classify the user's intent from the message and context.

        Returns one of:
            'ingest'            – new PDFs uploaded
            'literature_review' – user requests a lit review
            'question'          – research question
            'out_of_scope'      – unrelated request
        """
        msg_lower = message.lower().strip()

        if pdf_paths:
            return "ingest"

        lit_review_triggers = [
            "literature review",
            "lit review",
            "generate a review",
            "write a review",
            "survey of",
            "synthesize",
            "overview of the papers",
        ]
        for trigger in lit_review_triggers:
            if trigger in msg_lower:
                return "literature_review"

        # Research-domain triggers only — generic question words alone are too broad
        # and would match clearly off-topic queries such as "what is pizza?".
        question_triggers = [
            "method", "approach", "algorithm", "technique", "architecture",
            "finding", "result", "conclusion", "contribution",
            "limitation", "drawback", "weakness", "constraint",
            "gap", "future work", "future research", "open problem",
            "paper", "papers", "study", "studies", "research", "authors",
            "dataset", "experiment", "evaluation", "benchmark",
            "model", "framework", "system", "proposed",
            "abstract", "introduction", "background", "related work",
            "summarize the paper", "summarize these papers",
            "what did the authors", "what do the papers",
        ]
        for trigger in question_triggers:
            if trigger in msg_lower:
                return "question"

        return "out_of_scope"

    # ------------------------------------------------------------------
    # Tool routing handlers
    # ------------------------------------------------------------------

    def _handle_ingest(self, pdf_paths: list[str]) -> dict[str, Any]:
        """Route PDF files to the pdf_ingestion_agent."""
        agent = self._sub_agents.get("pdf_ingestion")
        if agent is None:
            try:
                from .pdf_ingestion_agent import PDFIngestionAgent
            except ImportError:
                from agents.pdf_ingestion_agent import PDFIngestionAgent
            agent = PDFIngestionAgent()

        logger.info("Routing to pdf_ingestion_agent for %d files.", len(pdf_paths))
        return agent.run(pdf_paths=pdf_paths)

    def _handle_question(self, query: str) -> dict[str, Any]:
        """Route a research question through retrieval → QA."""
        rag_agent = self._sub_agents.get("rag_retrieval")
        qa_agent = self._sub_agents.get("research_qa")

        if rag_agent is None:
            try:
                from .rag_retrieval_agent import RAGRetrievalAgent
            except ImportError:
                from agents.rag_retrieval_agent import RAGRetrievalAgent
            rag_agent = RAGRetrievalAgent()

        if qa_agent is None:
            try:
                from .research_qa_agent import ResearchQAAgent
            except ImportError:
                from agents.research_qa_agent import ResearchQAAgent
            qa_agent = ResearchQAAgent()

        logger.info("Routing question to rag_retrieval_agent: '%s'", query[:80])
        retrieval_result = rag_agent.run(query=query)
        chunks = retrieval_result.get("chunks", [])

        logger.info("Retrieved %d chunks; routing to research_qa_agent.", len(chunks))
        return qa_agent.run(query=query, chunks=chunks)

    def _handle_literature_review(self) -> dict[str, Any]:
        """Retrieve broad context from all papers and generate a literature review."""
        rag_agent = self._sub_agents.get("rag_retrieval")
        lit_agent = self._sub_agents.get("literature_review")

        if rag_agent is None:
            try:
                from .rag_retrieval_agent import RAGRetrievalAgent
            except ImportError:
                from agents.rag_retrieval_agent import RAGRetrievalAgent
            rag_agent = RAGRetrievalAgent()

        if lit_agent is None:
            try:
                from .literature_review_agent import LiteratureReviewAgent
            except ImportError:
                from agents.literature_review_agent import LiteratureReviewAgent
            lit_agent = LiteratureReviewAgent()

        broad_query = (
            "methods findings limitations research gaps future directions background"
        )
        logger.info("Routing literature review request to rag_retrieval_agent.")
        retrieval_result = rag_agent.run(query=broad_query, top_k=20)
        chunks = retrieval_result.get("chunks", [])

        logger.info("Retrieved %d chunks; routing to literature_review_agent.", len(chunks))
        return lit_agent.run(chunks=chunks)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        user_message: str,
        pdf_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Process a user message and return a structured response.

        Parameters
        ----------
        user_message : str
            The user's text input.
        pdf_paths : list[str], optional
            Paths to newly uploaded PDF files.

        Returns
        -------
        dict
            {
                "agent": "research_coordinator",
                "intent": str,
                "response": str,
                "data": dict,
                "error": str | None
            }
        """
        pdf_paths = pdf_paths or []
        result: dict[str, Any] = {
            "agent": AGENT_NAME,
            "intent": "unknown",
            "response": "",
            "data": {},
            "error": None,
        }

        try:
            intent = self._detect_intent(user_message, pdf_paths)
            result["intent"] = intent
            logger.info("Detected intent: %s", intent)

            if intent == "ingest":
                data = self._handle_ingest(pdf_paths)
                if data.get("success"):
                    papers = data.get("processed_papers", [])
                    result["response"] = (
                        f"Successfully ingested {len(papers)} paper(s): "
                        + ", ".join(f"'{p}'" for p in papers)
                        + ". You can now ask questions about these papers."
                    )
                else:
                    result["response"] = (
                        f"Ingestion encountered issues: {data.get('error', 'Unknown error')}"
                    )
                result["data"] = data

            elif intent == "literature_review":
                data = self._handle_literature_review()
                result["response"] = data.get(
                    "literature_review",
                    "This information is not available in the uploaded research papers.",
                )
                result["data"] = data

            elif intent == "question":
                data = self._handle_question(user_message)
                result["response"] = data.get(
                    "answer",
                    "This information is not available in the uploaded research papers.",
                )
                result["data"] = data

            else:  # out_of_scope
                result["response"] = (
                    "This information is not available in the uploaded research papers. "
                    "I can only answer questions based on the PDFs you have uploaded."
                )
                result["intent"] = "out_of_scope"

        except Exception as exc:
            result["error"] = str(exc)
            result["response"] = (
                "An internal error occurred. Please try again or contact support."
            )
            logger.exception("ResearchCoordinatorAgent.run() failed.")

        return result

    # ------------------------------------------------------------------
    # ADK metadata export
    # ------------------------------------------------------------------

    @staticmethod
    def get_agent_definition() -> dict[str, Any]:
        """Return the ADK-compatible agent definition dict."""
        return {
            "name": AGENT_NAME,
            "description": AGENT_DESCRIPTION,
            "model_id": os.getenv("WX_MODEL_QA", "ibm/granite-3-3-8b-instruct"),
            "system_prompt": SYSTEM_PROMPT,
            "tools": TOOLS,
        }
