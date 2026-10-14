from typing import Dict, Any, List, Optional
from tools.generate_literature_review import generate_literature_review

class LiteratureReviewAgent:
    name: str = "literature_review_agent"
    description: str = "Generates a 6-section structured literature review across all uploaded papers with citations."

    def run(
        self,
        domain_topic: str = "Artificial Intelligence & Large Language Models",
        chunks: Optional[List[Dict[str, Any]]] = None,
        use_watsonx_live: bool = False
    ) -> Dict[str, Any]:
        review_output = generate_literature_review(
            all_chunks=chunks,
            domain_topic=domain_topic,
            use_watsonx_live=use_watsonx_live
        )
        return {
            "agent": self.name,
            "status": review_output["status"],
            "topic": review_output.get("topic", domain_topic),
            "papers_reviewed": review_output.get("papers_reviewed", []),
            "total_citations": review_output.get("total_citations", 0),
            "review_markdown": review_output.get("review_markdown", ""),
            "sections": review_output.get("sections", {}),
            "citations": review_output.get("citations", [])
        }

literature_review_agent = LiteratureReviewAgent()
