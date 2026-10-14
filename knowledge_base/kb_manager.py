"""
Knowledge Base: research_papers_kb
Stores paper metadata, text chunks, page/section references, and vector embeddings.
"""

import math
import re
from typing import List, Dict, Any, Optional

class ResearchPapersKB:
    def __init__(self, name: str = "research_papers_kb"):
        self.name = name
        self.papers: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[Dict[str, Any]] = []
        self.vocabulary: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())

    def _compute_embedding(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * 128

        for t in tokens:
            if t not in self.vocabulary and len(self.vocabulary) < 1024:
                self.vocabulary[t] = len(self.vocabulary)

        dim = max(len(self.vocabulary), 64)
        vec = [0.0] * dim
        for t in tokens:
            idx = self.vocabulary.get(t, -1)
            if idx != -1 and idx < dim:
                vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        min_len = min(len(vec_a), len(vec_b))
        if min_len == 0:
            return 0.0
        dot = sum(vec_a[i] * vec_b[i] for i in range(min_len))
        return max(0.0, min(1.0, dot))

    def add_paper(self, paper_id: str, title: str, authors: Optional[List[str]] = None, year: Optional[int] = None) -> None:
        self.papers[paper_id] = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors or [],
            "year": year
        }

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        added_ids = []
        for ch in chunks:
            text = ch.get("text", "")
            embedding = ch.get("embedding")
            if not embedding:
                embedding = self._compute_embedding(text)

            chunk_entry = {
                "chunk_id": ch.get("chunk_id", f"chk_{len(self.chunks) + 1}"),
                "paper_id": ch.get("paper_id", "unknown"),
                "paper_title": ch.get("paper_title", "Untitled Research Paper"),
                "authors": ch.get("authors", []),
                "year": ch.get("year", 2024),
                "section_or_page": ch.get("section_or_page", "Page 1"),
                "text": text,
                "embedding": embedding
            }
            self.chunks.append(chunk_entry)
            added_ids.append(chunk_entry["chunk_id"])
        return added_ids

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.05) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        query_vec = self._compute_embedding(query)
        scored_results = []
        query_tokens = set(self._tokenize(query))

        for ch in self.chunks:
            sim = self._cosine_similarity(query_vec, ch["embedding"])
            ch_tokens = set(self._tokenize(ch["text"]))
            overlap = len(query_tokens.intersection(ch_tokens)) / max(len(query_tokens), 1)
            hybrid_score = 0.65 * sim + 0.35 * overlap

            if hybrid_score >= min_score:
                scored_results.append({
                    "chunk_id": ch["chunk_id"],
                    "paper_title": ch["paper_title"],
                    "section_or_page": ch["section_or_page"],
                    "source_text": ch["text"],
                    "score": round(hybrid_score, 4),
                    "authors": ch.get("authors", []),
                    "year": ch.get("year")
                })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        return self.chunks

    def clear(self) -> None:
        self.papers.clear()
        self.chunks.clear()
        self.vocabulary.clear()

global_kb = ResearchPapersKB("research_papers_kb")
