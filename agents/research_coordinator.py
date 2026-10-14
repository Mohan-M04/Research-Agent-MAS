import re
from typing import Dict, Any, List, Optional
from agents.pdf_ingestion_agent import pdf_ingestion_agent
from agents.rag_retrieval_agent import rag_retrieval_agent
from agents.research_qa_agent import research_qa_agent
from agents.literature_review_agent import literature_review_agent

class ResearchCoordinatorAgent:
    name: str = "research_coordinator"
    description: str = "Main coordinator agent that routes requests, coordinates sub-agents, and returns citation-backed responses."

    def classify_intent(self, user_message: str, has_uploaded_files: bool = False) -> str:
        if has_uploaded_files:
            return "ingest_pdf"
        msg = user_message.lower().strip()
        lit_keywords = ["literature review", "lit review", "synthesize all papers", "review of the papers", "overall review"]
        if any(k in msg for k in lit_keywords):
            return "literature_review"
        if msg in ["hi", "hello", "hey", "help", "what can you do?", "what are your capabilities?"]:
            return "greeting_or_help"
        return "research_qa"

    def handle_request(
        self,
        user_message: str,
        uploaded_files: Optional[List[Any]] = None,
        file_names: Optional[List[str]] = None,
        use_watsonx_live: bool = False
    ) -> Dict[str, Any]:
        has_files = bool(uploaded_files and len(uploaded_files) > 0)
        intent = self.classify_intent(user_message, has_uploaded_files=has_files)

        if intent == "ingest_pdf":
            ingest_result = pdf_ingestion_agent.run(file_inputs=uploaded_files, file_names=file_names)
            return {
                "coordinator": self.name,
                "intent": "ingest_pdf",
                "status": ingest_result["status"],
                "response": (
                    f"### Papers Ingested Successfully\n"
                    f"{ingest_result['message']}\n\n"
                    f"**Processed Papers:**\n" + 
                    "\n".join([f"- **{p['paper_title']}** ({p['chunks_indexed']} chunks, {p['pages']} pages)" for p in ingest_result['papers']]) +
                    f"\n\nYou can now ask questions such as:\n"
                    f"- *'What methods are used in these papers?'*\n"
                    f"- *'What are the main findings?'*\n"
                    f"- *'What are the limitations?'*\n"
                    f"- *'What research gaps exist?'*\n"
                    f"- *'Generate a literature review.'*"
                ),
                "data": ingest_result
            }

        elif intent == "literature_review":
            review_res = literature_review_agent.run(
                domain_topic="Comparative Study of Uploaded Research Papers",
                use_watsonx_live=use_watsonx_live
            )
            return {
                "coordinator": self.name,
                "intent": "literature_review",
                "status": review_res["status"],
                "response": review_res["review_markdown"],
                "citations_count": review_res.get("total_citations", 0),
                "data": review_res
            }

        elif intent == "greeting_or_help":
            return {
                "coordinator": self.name,
                "intent": "greeting_or_help",
                "status": "success",
                "response": (
                    "### Welcome to Research Companion (IBM watsonx Orchestrate)\n"
                    "I am your multi-agent research assistant powered by IBM Granite.\n\n"
                    "**How to use:**\n"
                    "1. Upload one or more research-paper PDF files.\n"
                    "2. Ask any research question (e.g., methods, key findings, limitations, or gaps).\n"
                    "3. Request *'Generate a literature review'* for a synthesized 6-section comparative analysis.\n\n"
                    "All responses are strictly grounded in your uploaded papers with citations `[Paper Title, Page/Section]`."
                ),
                "data": {}
            }

        else:
            retrieval_res = rag_retrieval_agent.run(query=user_message, top_k=5)
            chunks = retrieval_res.get("chunks", [])
            qa_res = research_qa_agent.run(
                query=user_message,
                retrieved_chunks=chunks,
                use_watsonx_live=use_watsonx_live
            )
            return {
                "coordinator": self.name,
                "intent": "research_qa",
                "status": qa_res["status"],
                "response": qa_res["answer"],
                "citations": qa_res.get("citations", []),
                "sources_used": qa_res.get("sources_used", []),
                "retrieved_chunks_count": len(chunks),
                "data": {
                    "retrieval": retrieval_res,
                    "qa": qa_res
                }
            }

research_coordinator = ResearchCoordinatorAgent()
