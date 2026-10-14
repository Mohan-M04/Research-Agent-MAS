from typing import Dict, Any, List, Optional
from tools.retrieve_relevant_chunks import retrieve_relevant_chunks

class RAGRetrievalAgent:
    name: str = "rag_retrieval_agent"
    description: str = "Performs semantic and hybrid retrieval over indexed research papers in research_papers_kb."

    def run(self, query: str, top_k: int = 5, paper_filter: Optional[str] = None) -> Dict[str, Any]:
        retrieval_output = retrieve_relevant_chunks(query=query, top_k=top_k, paper_filter=paper_filter)
        return {
            "status": retrieval_output["status"],
            "agent": self.name,
            "query": query,
            "total_chunks_retrieved": retrieval_output["total_retrieved"],
            "chunks": retrieval_output["chunks"],
            "retrieval_metadata": {
                "top_k": top_k,
                "applied_filter": paper_filter,
                "knowledge_base": "research_papers_kb"
            }
        }

rag_retrieval_agent = RAGRetrievalAgent()
