from typing import Dict, Any, List
from tools.extract_pdf_text import extract_pdf_text
from tools.chunk_and_index_documents import chunk_and_index_documents

class PDFIngestionAgent:
    name: str = "pdf_ingestion_agent"
    description: str = "Handles uploaded PDF files, extracts text & metadata, chunks text, and indexes into research_papers_kb."

    def run(self, file_inputs: List[Any], file_names: List[str] = None) -> Dict[str, Any]:
        file_names = file_names or []
        ingested_papers = []
        total_chunks = 0
        errors = []

        for idx, file_item in enumerate(file_inputs):
            name = file_names[idx] if idx < len(file_names) else f"paper_{idx+1}.pdf"
            extract_res = extract_pdf_text(file_item, file_name=name)
            if extract_res.get("status") == "error":
                errors.append(f"Failed to extract {name}: {extract_res.get('error_message')}")
                continue

            index_res = chunk_and_index_documents(extract_res)
            if index_res.get("status") == "error":
                errors.append(f"Failed to index {name}: {index_res.get('error_message')}")
                continue

            ingested_papers.append({
                "paper_title": index_res["paper_title"],
                "file_name": name,
                "chunks_indexed": index_res["total_chunks_created"],
                "pages": extract_res["total_pages"]
            })
            total_chunks += index_res["total_chunks_created"]

        return {
            "status": "success" if ingested_papers else "failed",
            "agent": self.name,
            "processed_papers_count": len(ingested_papers),
            "total_chunks_indexed": total_chunks,
            "papers": ingested_papers,
            "errors": errors,
            "message": f"Successfully ingested {len(ingested_papers)} papers ({total_chunks} chunks indexed into research_papers_kb)."
        }

pdf_ingestion_agent = PDFIngestionAgent()
