"""
Python Tool 1: extract_pdf_text
"""

import os
import re
from typing import Dict, Any, List

def extract_pdf_text(file_path_or_bytes: Any, file_name: str = "") -> Dict[str, Any]:
    pages_data: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []
    paper_title = ""
    authors: List[str] = []
    year = 2024
    metadata: Dict[str, Any] = {}

    try:
        if isinstance(file_path_or_bytes, str) and os.path.exists(file_path_or_bytes):
            file_name = file_name or os.path.basename(file_path_or_bytes)
            
            if file_path_or_bytes.endswith(".txt"):
                with open(file_path_or_bytes, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                return _process_raw_text(content, file_name)
            
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path_or_bytes)
                pdf_meta = reader.metadata or {}
                if pdf_meta.title:
                    paper_title = str(pdf_meta.title)
                if pdf_meta.author:
                    authors = [a.strip() for a in str(pdf_meta.author).split(",") if a.strip()]

                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    clean_page = page_text.strip()
                    pages_data.append({"page_number": i + 1, "text": clean_page})
                    full_text_parts.append(clean_page)

            except ImportError:
                with open(file_path_or_bytes, "rb") as f:
                    raw_bytes = f.read()
                return _extract_from_binary_stream(raw_bytes, file_name)

        elif isinstance(file_path_or_bytes, bytes):
            return _extract_from_binary_stream(file_path_or_bytes, file_name)
            
        elif isinstance(file_path_or_bytes, str):
            return _process_raw_text(file_path_or_bytes, file_name)

        full_text = "\n\n".join(full_text_parts)

        if not paper_title and pages_data:
            first_page_text = pages_data[0]["text"]
            paper_title = _infer_title_from_text(first_page_text, file_name)

        return {
            "status": "success",
            "paper_title": paper_title or file_name.replace(".pdf", "").replace("_", " ").title(),
            "authors": authors,
            "year": year,
            "total_pages": len(pages_data),
            "pages": pages_data,
            "full_text": full_text,
            "metadata": {"file_name": file_name, "character_count": len(full_text)}
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "paper_title": file_name or "Unknown Paper",
            "authors": [],
            "year": year,
            "total_pages": 0,
            "pages": [],
            "full_text": "",
            "metadata": {}
        }


def _process_raw_text(text: str, file_name: str) -> Dict[str, Any]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    page_splits = re.split(r"--- Page (\d+) ---", text)
    pages = []
    
    if len(page_splits) > 1:
        for i in range(1, len(page_splits), 2):
            pg_num = int(page_splits[i])
            pg_txt = page_splits[i+1].strip()
            pages.append({"page_number": pg_num, "text": pg_txt})
    else:
        step = 1200
        for i in range(0, max(len(text), 1), step):
            chunk_slice = text[i:i+step].strip()
            pages.append({"page_number": (i // step) + 1, "text": chunk_slice})

    return {
        "status": "success",
        "paper_title": _infer_title_from_text(text, file_name),
        "authors": ["Research Author(s)"],
        "year": 2024,
        "total_pages": len(pages),
        "pages": pages,
        "full_text": text,
        "metadata": {"file_name": file_name}
    }


def _extract_from_binary_stream(raw_bytes: bytes, file_name: str) -> Dict[str, Any]:
    try:
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            pages.append({"page_number": i + 1, "text": page.extract_text() or ""})
        full_text = "\n\n".join(p["text"] for p in pages)
        title = _infer_title_from_text(pages[0]["text"] if pages else "", file_name)
        return {
            "status": "success",
            "paper_title": title,
            "authors": [],
            "year": 2024,
            "total_pages": len(pages),
            "pages": pages,
            "full_text": full_text,
            "metadata": {"file_name": file_name}
        }
    except Exception:
        text = raw_bytes.decode("utf-8", errors="ignore")
        return _process_raw_text(text, file_name)


def _infer_title_from_text(text: str, fallback_file_name: str) -> str:
    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 5]
    for line in lines[:5]:
        if not line.lower().startswith(("arxiv", "proceedings", "conference", "http", "issn", "doi")):
            return line
    return fallback_file_name.replace(".pdf", "").replace(".txt", "").replace("_", " ").title()
