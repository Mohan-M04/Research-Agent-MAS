"""
Python Tool 5: generate_literature_review
Structured academic literature review generator with grounded per-section citations.
"""

import os
import re
from typing import Dict, Any, List, Tuple
from knowledge_base.kb_manager import global_kb

def generate_literature_review(
    all_chunks: List[Dict[str, Any]] = None,
    domain_topic: str = "Comparative Study of Uploaded Research Papers",
    use_watsonx_live: bool = False
) -> Dict[str, Any]:
    if not all_chunks:
        all_chunks = global_kb.get_all_chunks()

    if not all_chunks:
        return {
            "status": "error",
            "error_message": "No indexed research papers found in research_papers_kb. Please upload and ingest PDFs first.",
            "topic": domain_topic,
            "papers_reviewed": [],
            "total_citations": 0,
            "review_markdown": "This information is not available in the uploaded research papers.",
            "sections": {},
            "citations": []
        }

    papers_catalog: Dict[str, List[Dict[str, Any]]] = {}
    for ch in all_chunks:
        title = ch.get("paper_title", "Untitled Paper")
        papers_catalog.setdefault(title, []).append(ch)

    return _synthesize_structured_literature_review(domain_topic, papers_catalog)


def _find_best_chunk_and_sec(chunks: List[Dict[str, Any]], keywords: List[str]) -> Tuple[str, str]:
    """Finds the most specific chunk and section reference matching the topic keywords."""
    best_text = ""
    best_sec = "Page 1 - Overview"
    
    # 1. First priority: Match in section title
    for ch in chunks:
        sec = ch.get("section_or_page", "")
        if any(k in sec.lower() for k in keywords):
            text = ch.get("text", "").strip()
            # Clean complete sentence
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
            best_text = sentences[0] if sentences else text[:200]
            best_sec = sec
            return best_text, best_sec

    # 2. Second priority: Match in chunk body text
    for ch in chunks:
        text = ch.get("text", "")
        t_lower = text.lower()
        if any(k in t_lower for k in keywords):
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
            for s in sentences:
                if any(k in s.lower() for k in keywords):
                    best_text = s
                    best_sec = ch.get("section_or_page", "Page 1")
                    return best_text, best_sec

    # Fallback to first chunk
    if chunks:
        text = chunks[0].get("text", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
        best_text = sentences[0] if sentences else text[:180]
        best_sec = chunks[0].get("section_or_page", "Page 1")

    return best_text, best_sec


def _synthesize_structured_literature_review(
    domain_topic: str,
    papers_catalog: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    citations: List[Dict[str, str]] = []

    def cite(title: str, sec: str) -> str:
        tag = f"[{title}, {sec}]"
        citations.append({"paper_title": title, "section_or_page": sec, "tag": tag})
        return tag

    # 1. Background
    sec1_lines = [
        f"Recent advancements in **{domain_topic}** have reshaped modern machine learning and computational linguistics.",
        f"This literature review synthesizes insights across {len(papers_catalog)} seminal works: {', '.join(papers_catalog.keys())}."
    ]
    for title, chs in papers_catalog.items():
        text, sec = _find_best_chunk_and_sec(chs, ["abstract", "introduction", "background"])
        c = cite(title, sec)
        sec1_lines.append(f"- **{title}**: {text} {c}")
    sec1 = "\n\n".join(sec1_lines)

    # 2. Methods and Approaches
    sec2_lines = [
        "The examined papers present distinct yet complementary methodological paradigms:"
    ]
    for title, chs in papers_catalog.items():
        text, sec = _find_best_chunk_and_sec(chs, ["method", "architecture", "model", "parameter", "attention", "retrieval", "encoder"])
        c = cite(title, sec)
        sec2_lines.append(f"- **{title} (Methodology)**: {text} {c}")
    sec2 = "\n\n".join(sec2_lines)

    # 3. Key Findings
    sec3_lines = [
        "Across rigorous empirical benchmarks, the key findings demonstrate significant gains:"
    ]
    for title, chs in papers_catalog.items():
        text, sec = _find_best_chunk_and_sec(chs, ["result", "finding", "experiment", "accuracy", "bleu", "performance", "outperform"])
        c = cite(title, sec)
        sec3_lines.append(f"- **{title} (Key Findings)**: {text} {c}")
    sec3 = "\n\n".join(sec3_lines)

    # 4. Limitations
    sec4_lines = [
        "Despite significant performance breakthroughs, the authors explicitly articulate notable constraints:"
    ]
    for title, chs in papers_catalog.items():
        text, sec = _find_best_chunk_and_sec(chs, ["limitation", "complexity", "bottleneck", "latency", "memory", "overhead"])
        c = cite(title, sec)
        sec4_lines.append(f"- **{title} (Limitations)**: {text} {c}")
    sec4 = "\n\n".join(sec4_lines)

    # 5. Research Gaps
    sec5_lines = [
        "Synthesizing the collective findings highlights several critical research gaps in current literature:"
    ]
    for title, chs in papers_catalog.items():
        text, sec = _find_best_chunk_and_sec(chs, ["gap", "discussion", "unresolved", "reasoning"])
        c = cite(title, sec)
        sec5_lines.append(f"- **{title} (Research Gap)**: {text} {c}")
    sec5 = "\n\n".join(sec5_lines)

    # 6. Future Research Directions
    sec6_lines = [
        "To address these identified gaps, the authors and empirical trends point toward the following future trajectories:"
    ]
    for title, chs in papers_catalog.items():
        text, sec = _find_best_chunk_and_sec(chs, ["future", "future work", "plan", "direction", "sparse"])
        c = cite(title, sec)
        sec6_lines.append(f"- **{title} (Future Directions)**: {text} {c}")
    sec6 = "\n\n".join(sec6_lines)

    sections = {
        "1. Background": sec1,
        "2. Methods and Approaches": sec2,
        "3. Key Findings": sec3,
        "4. Limitations": sec4,
        "5. Research Gaps": sec5,
        "6. Future Research Directions": sec6
    }

    full_md = f"""# Comprehensive Literature Review: {domain_topic}
*Synthesized by IBM watsonx Research Companion Agent*

---

## 1. Background
{sec1}

---

## 2. Methods and Approaches
{sec2}

---

## 3. Key Findings
{sec3}

---

## 4. Limitations
{sec4}

---

## 5. Research Gaps
{sec5}

---

## 6. Future Research Directions
{sec6}

---
### Source Papers Included:
"""
    for title in papers_catalog:
        full_md += f"- *{title}*\n"

    return {
        "status": "success",
        "topic": domain_topic,
        "papers_reviewed": list(papers_catalog.keys()),
        "total_citations": len(citations),
        "review_markdown": full_md,
        "sections": sections,
        "citations": citations
    }
