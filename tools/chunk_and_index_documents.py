"""
Python Tool 2: chunk_and_index_documents
Clean text segmentation respecting sentence and word boundaries.
"""

import re
from typing import Dict, Any, List, Optional
from knowledge_base.kb_manager import global_kb

def chunk_and_index_documents(
    extracted_data: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 80
) -> Dict[str, Any]:
    try:
        paper_title = extracted_data.get("paper_title", "Untitled Research Paper")
        pages = extracted_data.get("pages", [])
        authors = extracted_data.get("authors", [])
        year = extracted_data.get("year", 2024)
        paper_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", paper_title.lower())[:32]

        global_kb.add_paper(paper_id=paper_id, title=paper_title, authors=authors, year=year)
        chunks_to_index: List[Dict[str, Any]] = []

        for p in pages:
            page_num = p.get("page_number", 1)
            raw_page_text = p.get("text", "")
            clean_text = re.sub(r"\s+", " ", raw_page_text).strip()
            if not clean_text:
                continue

            section_header = _detect_section_header(clean_text)
            section_ref = f"Page {page_num} - {section_header}" if section_header else f"Page {page_num}"

            # Split into natural sentences first
            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if s.strip()]
            
            current_chunk = []
            current_len = 0
            chunk_idx = 1

            for s in raw_sentences:
                current_chunk.append(s)
                current_len += len(s) + 1

                if current_len >= chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunk_id = f"{paper_id}_p{page_num}_c{chunk_idx}"
                    chunks_to_index.append({
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "authors": authors,
                        "year": year,
                        "section_or_page": section_ref,
                        "text": chunk_text
                    })
                    chunk_idx += 1
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1]] if len(current_chunk) > 1 else []
                    current_len = sum(len(x) + 1 for x in current_chunk)

            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) > 30:
                    chunk_id = f"{paper_id}_p{page_num}_c{chunk_idx}"
                    chunks_to_index.append({
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "authors": authors,
                        "year": year,
                        "section_or_page": section_ref,
                        "text": chunk_text
                    })

        indexed_ids = global_kb.add_chunks(chunks_to_index)

        return {
            "status": "success",
            "paper_title": paper_title,
            "total_chunks_created": len(indexed_ids),
            "chunk_ids": indexed_ids,
            "embedding_status": "embedded_with_granite_embeddings",
            "indexing_status": f"indexed_{len(indexed_ids)}_chunks_into_research_papers_kb",
            "knowledge_base": global_kb.name
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "paper_title": extracted_data.get("paper_title", "Unknown"),
            "total_chunks_created": 0,
            "chunk_ids": [],
            "embedding_status": "failed",
            "indexing_status": "failed",
            "knowledge_base": "research_papers_kb"
        }


def _detect_section_header(text: str) -> Optional[str]:
    common_sections = [
        "Abstract", "Introduction", "Related Work", "Background", 
        "Methodology", "Methods", "Model Architecture", "Experiments", 
        "Results", "Discussion", "Ablation Study", "Limitations", 
        "Research Gaps", "Future Work", "Conclusion"
    ]
    for sec in common_sections:
        pattern = rf"\b(?:[0-9]\.?\s+)?{sec}\b"
        if re.search(pattern, text[:250], re.IGNORECASE):
            return sec
    return None
