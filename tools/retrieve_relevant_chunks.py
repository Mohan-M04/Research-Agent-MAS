"""
Python Tool 3: retrieve_relevant_chunks
"""

from typing import Dict, Any, List, Optional
from knowledge_base.kb_manager import global_kb

def retrieve_relevant_chunks(
    query: str,
    top_k: int = 5,
    paper_filter: Optional[str] = None
) -> Dict[str, Any]:
    if not query or not query.strip():
        return {
            "status": "empty",
            "query": "",
            "total_retrieved": 0,
            "chunks": []
        }

    raw_results = global_kb.retrieve(query=query.strip(), top_k=top_k * 2)
    filtered_results = []
    for r in raw_results:
        if paper_filter:
            if paper_filter.lower() not in r["paper_title"].lower():
                continue
        filtered_results.append({
            "chunk_id": r["chunk_id"],
            "paper_title": r["paper_title"],
            "section_or_page": r["section_or_page"],
            "source_text": r["source_text"],
            "similarity_score": r["score"],
            "authors": r.get("authors", []),
            "year": r.get("year", 2024)
        })

    final_chunks = filtered_results[:top_k]

    return {
        "status": "success" if final_chunks else "empty",
        "query": query,
        "total_retrieved": len(final_chunks),
        "chunks": final_chunks
    }
