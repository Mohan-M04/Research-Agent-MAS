"""
workflow/research_companion_workflow.py
────────────────────────────────────────
Workflow: research_companion_workflow

Steps:
    1. Receive uploaded PDFs and the user request.
    2. If new PDFs are uploaded, call pdf_ingestion_agent.
    3. Retrieve relevant chunks using rag_retrieval_agent.
    4. If the user asks a question, call research_qa_agent.
    5. If the user requests a literature review, call literature_review_agent.
    6. Return the final answer or literature review with citations.

This module implements the workflow as a Python orchestration class that can
be executed locally or registered with the watsonx Orchestrate environment
via the deploy_assets script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workflow intent enum
# ---------------------------------------------------------------------------

class WorkflowIntent(str, Enum):
    INGEST = "ingest"
    QUESTION = "question"
    LITERATURE_REVIEW = "literature_review"
    OUT_OF_SCOPE = "out_of_scope"


# ---------------------------------------------------------------------------
# Workflow state (tracks what has been processed in this session)
# ---------------------------------------------------------------------------

@dataclass
class WorkflowState:
    """Mutable state for a single workflow execution session."""
    pdf_paths: list[str] = field(default_factory=list)
    ingested_papers: list[str] = field(default_factory=list)
    last_retrieved_chunks: list[dict] = field(default_factory=list)
    last_answer: str = ""
    last_literature_review: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------

WORKFLOW_NAME = "research_companion_workflow"
WORKFLOW_DESCRIPTION = (
    "End-to-end research companion workflow. "
    "Handles PDF ingestion, semantic retrieval, question answering, "
    "and structured literature review generation with citations."
)

# ADK-style step definitions (used for registration and documentation)
WORKFLOW_STEPS = [
    {
        "step": 1,
        "name": "receive_request",
        "description": "Receive uploaded PDFs and the user request.",
        "type": "input",
    },
    {
        "step": 2,
        "name": "ingest_pdfs",
        "description": "If new PDFs are uploaded, call pdf_ingestion_agent to extract, chunk, and index them.",
        "type": "conditional",
        "condition": "pdf_paths is not empty",
        "agent": "pdf_ingestion_agent",
    },
    {
        "step": 3,
        "name": "retrieve_chunks",
        "description": "Retrieve the most relevant paper chunks using rag_retrieval_agent.",
        "type": "tool",
        "agent": "rag_retrieval_agent",
    },
    {
        "step": 4,
        "name": "answer_question",
        "description": "If user asks a question, call research_qa_agent with retrieved chunks.",
        "type": "conditional",
        "condition": "intent == 'question'",
        "agent": "research_qa_agent",
    },
    {
        "step": 5,
        "name": "generate_literature_review",
        "description": "If user requests a literature review, call literature_review_agent with retrieved chunks.",
        "type": "conditional",
        "condition": "intent == 'literature_review'",
        "agent": "literature_review_agent",
    },
    {
        "step": 6,
        "name": "return_response",
        "description": "Return the final answer or literature review with citations to the user.",
        "type": "output",
    },
]


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class ResearchCompanionWorkflow:
    """
    Orchestrates the end-to-end Research Companion pipeline.

    Usage
    -----
    workflow = ResearchCompanionWorkflow()
    result = workflow.execute(
        user_message="What methods are used in these papers?",
        pdf_paths=["paper1.pdf", "paper2.pdf"],
    )
    """

    def __init__(self) -> None:
        # Lazy-load agents to avoid circular imports
        self._coordinator = None
        self._pdf_ingestion = None
        self._rag_retrieval = None
        self._research_qa = None
        self._literature_review = None

    # ------------------------------------------------------------------
    # Lazy agent accessors
    # ------------------------------------------------------------------

    def _get_coordinator(self):
        if self._coordinator is None:
            try:
                from ..agents.research_coordinator import ResearchCoordinatorAgent
            except ImportError:
                from agents.research_coordinator import ResearchCoordinatorAgent
            self._coordinator = ResearchCoordinatorAgent()
        return self._coordinator

    def _get_pdf_ingestion(self):
        if self._pdf_ingestion is None:
            try:
                from ..agents.pdf_ingestion_agent import PDFIngestionAgent
            except ImportError:
                from agents.pdf_ingestion_agent import PDFIngestionAgent
            self._pdf_ingestion = PDFIngestionAgent()
        return self._pdf_ingestion

    def _get_rag_retrieval(self):
        if self._rag_retrieval is None:
            try:
                from ..agents.rag_retrieval_agent import RAGRetrievalAgent
            except ImportError:
                from agents.rag_retrieval_agent import RAGRetrievalAgent
            self._rag_retrieval = RAGRetrievalAgent()
        return self._rag_retrieval

    def _get_research_qa(self):
        if self._research_qa is None:
            try:
                from ..agents.research_qa_agent import ResearchQAAgent
            except ImportError:
                from agents.research_qa_agent import ResearchQAAgent
            self._research_qa = ResearchQAAgent()
        return self._research_qa

    def _get_literature_review(self):
        if self._literature_review is None:
            try:
                from ..agents.literature_review_agent import LiteratureReviewAgent
            except ImportError:
                from agents.literature_review_agent import LiteratureReviewAgent
            self._literature_review = LiteratureReviewAgent()
        return self._literature_review

    # ------------------------------------------------------------------
    # Intent detection (mirrors coordinator logic, kept here for clarity)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_intent(user_message: str, pdf_paths: list[str]) -> WorkflowIntent:
        msg = user_message.lower().strip()
        if pdf_paths:
            return WorkflowIntent.INGEST

        # Literature-review triggers (check first — most specific)
        for kw in [
            "literature review", "lit review", "generate a review", "write a review",
            "survey of", "overview of the papers", "synthesize the papers",
        ]:
            if kw in msg:
                return WorkflowIntent.LITERATURE_REVIEW

        # Research-domain triggers — must reference paper/research topics explicitly.
        # Generic question words (what, how, why) alone are NOT enough to route to QA
        # because they also appear in clearly off-topic queries ("what is pizza?").
        research_triggers = [
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
        for kw in research_triggers:
            if kw in msg:
                return WorkflowIntent.QUESTION

        return WorkflowIntent.OUT_OF_SCOPE

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step_ingest(self, pdf_paths: list[str], state: WorkflowState) -> None:
        """Step 2: Ingest PDFs if provided."""
        logger.info("[Workflow] Step 2 — Ingesting %d PDF(s).", len(pdf_paths))
        result = self._get_pdf_ingestion().run(pdf_paths=pdf_paths)
        if result.get("success"):
            state.ingested_papers.extend(result.get("processed_papers", []))
            logger.info("[Workflow] Ingested: %s", state.ingested_papers)
        else:
            state.error = result.get("error", "PDF ingestion failed.")
            logger.warning("[Workflow] Ingestion error: %s", state.error)

    def _step_retrieve(self, query: str, state: WorkflowState, top_k: int = 20) -> None:
        """Step 3: Retrieve relevant chunks."""
        logger.info("[Workflow] Step 3 — Retrieving chunks for query: '%s'.", query[:80])
        result = self._get_rag_retrieval().run(query=query, top_k=top_k)
        state.last_retrieved_chunks = result.get("chunks", [])
        logger.info("[Workflow] Retrieved %d chunks.", len(state.last_retrieved_chunks))

    def _step_answer_question(self, query: str, state: WorkflowState) -> None:
        """Step 4: Answer a research question."""
        logger.info("[Workflow] Step 4 — Answering question.")
        result = self._get_research_qa().run(
            query=query, chunks=state.last_retrieved_chunks
        )
        state.last_answer = result.get(
            "answer", "This information is not available in the uploaded research papers."
        )

    def _step_generate_literature_review(self, state: WorkflowState) -> None:
        """Step 5: Generate literature review."""
        logger.info("[Workflow] Step 5 — Generating literature review.")
        result = self._get_literature_review().run(chunks=state.last_retrieved_chunks)
        state.last_literature_review = result.get(
            "literature_review",
            "This information is not available in the uploaded research papers.",
        )

    # ------------------------------------------------------------------
    # Main execution entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        user_message: str,
        pdf_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full research companion workflow.

        Parameters
        ----------
        user_message : str
            The user's text input.
        pdf_paths : list[str], optional
            Paths to newly uploaded PDF files (triggers ingestion).

        Returns
        -------
        dict
            {
                "workflow": "research_companion_workflow",
                "intent": str,
                "response": str,
                "ingested_papers": list[str],
                "retrieved_chunks_count": int,
                "error": str | None
            }
        """
        pdf_paths = pdf_paths or []
        state = WorkflowState(pdf_paths=pdf_paths)

        output: dict[str, Any] = {
            "workflow": WORKFLOW_NAME,
            "intent": "unknown",
            "response": "",
            "ingested_papers": [],
            "retrieved_chunks_count": 0,
            "error": None,
        }

        try:
            # ── Step 1: Receive request ─────────────────────────────────────
            logger.info("[Workflow] Step 1 — Received request: '%s'.", user_message[:80])
            intent = self._detect_intent(user_message, pdf_paths)
            output["intent"] = intent.value

            # ── Step 2: Ingest PDFs (conditional) ──────────────────────────
            if pdf_paths:
                self._step_ingest(pdf_paths, state)
                output["ingested_papers"] = state.ingested_papers

                if state.error and not state.ingested_papers:
                    output["error"] = state.error
                    output["response"] = (
                        f"PDF ingestion failed: {state.error} "
                        "Please check the files and try again."
                    )
                    return output

                # After ingestion, inform the user and wait for a question
                output["response"] = (
                    f"Successfully ingested {len(state.ingested_papers)} paper(s): "
                    + ", ".join(f"'{p}'" for p in state.ingested_papers)
                    + ". You can now ask questions or request a literature review."
                )

                # If the user also asked something along with the upload,
                # continue processing that request below.
                if intent == WorkflowIntent.INGEST:
                    return output
                # Otherwise fall through to Q&A or lit-review

            # ── Step 3: Retrieve relevant chunks ───────────────────────────
            if intent in (WorkflowIntent.QUESTION, WorkflowIntent.LITERATURE_REVIEW):
                retrieval_query = (
                    user_message
                    if intent == WorkflowIntent.QUESTION
                    else "methods findings limitations research gaps future directions background"
                )
                self._step_retrieve(retrieval_query, state)
                output["retrieved_chunks_count"] = len(state.last_retrieved_chunks)

            # ── Step 4: Answer question (conditional) ──────────────────────
            if intent == WorkflowIntent.QUESTION:
                self._step_answer_question(user_message, state)
                output["response"] = state.last_answer

            # ── Step 5: Literature review (conditional) ────────────────────
            elif intent == WorkflowIntent.LITERATURE_REVIEW:
                self._step_generate_literature_review(state)
                output["response"] = state.last_literature_review

            # ── Out-of-scope ───────────────────────────────────────────────
            elif intent == WorkflowIntent.OUT_OF_SCOPE:
                output["response"] = (
                    "This information is not available in the uploaded research papers. "
                    "I can only answer questions based on the PDFs you have uploaded."
                )

            # ── Step 6: Return response ─────────────────────────────────────
            logger.info("[Workflow] Step 6 — Returning response (intent=%s).", intent.value)

        except Exception as exc:
            output["error"] = str(exc)
            output["response"] = (
                "An internal workflow error occurred. Please try again."
            )
            logger.exception("[Workflow] Unhandled exception in execute().")

        return output

    # ------------------------------------------------------------------
    # ADK metadata export
    # ------------------------------------------------------------------

    @staticmethod
    def get_workflow_definition() -> dict[str, Any]:
        """Return the ADK-compatible workflow definition dict."""
        return {
            "name": WORKFLOW_NAME,
            "description": WORKFLOW_DESCRIPTION,
            "steps": WORKFLOW_STEPS,
            "agents": [
                "pdf_ingestion_agent",
                "rag_retrieval_agent",
                "research_qa_agent",
                "literature_review_agent",
            ],
        }
