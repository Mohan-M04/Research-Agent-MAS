from typing import Dict, Any, List
from tools.generate_cited_answer import generate_cited_answer, STRICT_REFUSAL_MESSAGE

class ResearchQAAgent:
    name: str = "research_qa_agent"
    description: str = "Answers research questions using only retrieved paper content with inline citations [Paper Title, Page/Section]."

    def run(self, query: str, retrieved_chunks: List[Dict[str, Any]], use_watsonx_live: bool = False) -> Dict[str, Any]:
        qa_output = generate_cited_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            use_watsonx_live=use_watsonx_live
        )
        return {
            "agent": self.name,
            "status": qa_output["status"],
            "query": query,
            "answer": qa_output["answer"],
            "citations": qa_output["citations"],
            "sources_used": qa_output["sources_used"],
            "is_refusal": qa_output["answer"].strip() == STRICT_REFUSAL_MESSAGE
        }

research_qa_agent = ResearchQAAgent()
