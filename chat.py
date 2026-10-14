"""
Terminal Interactive Chat for Research Companion
Pure CLI - Zero Frontend Required.
IBM watsonx Orchestrate Multi-Agent Research Assistant
"""

import os
import sys
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from knowledge_base.kb_manager import global_kb
from agents.research_coordinator import research_coordinator
from tools.extract_pdf_text import extract_pdf_text
from tools.chunk_and_index_documents import chunk_and_index_documents

def preload_samples():
    global_kb.clear()
    sample_dir = os.path.join(BASE_DIR, "sample_data")
    if os.path.exists(sample_dir):
        papers = ["attention_is_all_you_need.txt", "rag_knowledge_intensive.txt", "lora_low_rank_adaptation.txt"]
        for p in papers:
            fpath = os.path.join(sample_dir, p)
            if os.path.exists(fpath):
                ext = extract_pdf_text(fpath, file_name=p)
                chunk_and_index_documents(ext)

def clean_input(raw: str) -> str:
    """Strips accidental prompt prefixes like 'You > ' or 'You: '."""
    cleaned = raw.strip()
    # Strip repeated 'You >' or 'You:'
    cleaned = re.sub(r"^(?:you\s*[:>]\s*)+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned

def main():
    preload_samples()
    print("=" * 75)
    print(" 🎓 Research Companion: Multi-Agent CLI (IBM watsonx Orchestrate)")
    print(f" Knowledge Base : {global_kb.name} ({len(global_kb.papers)} papers, {len(global_kb.chunks)} chunks)")
    print(" Active Agents  : research_coordinator, pdf_ingestion_agent,")
    print("                  rag_retrieval_agent, research_qa_agent, literature_review_agent")
    print(" Commands       :")
    print("   - review         -> Generate 6-section synthesized literature review")
    print("   - upload <file>  -> Ingest a PDF or text research paper")
    print("   - status         -> View currently indexed papers and chunks")
    print("   - exit or quit   -> Exit CLI")
    print("=" * 75)

    while True:
        try:
            raw_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Research Companion.")
            break

        user_input = clean_input(raw_input)
        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting Research Companion. Goodbye!")
            break

        if user_input.lower() == "status":
            print(f"\n[Status] Papers Indexed: {len(global_kb.papers)} | Chunks: {len(global_kb.chunks)}")
            for pid, pdata in global_kb.papers.items():
                print(f" - {pdata['title']} ({pdata.get('year', 2024)})")
            continue

        if user_input.lower().startswith("upload "):
            fpath = user_input[7:].strip().strip('"').strip("'")
            if not os.path.exists(fpath):
                print(f"[Error] File not found: {fpath}")
                continue
            print(f"Ingesting '{fpath}' via pdf_ingestion_agent...")
            ext = extract_pdf_text(fpath)
            res = chunk_and_index_documents(ext)
            print(f"[Success] Indexed '{res['paper_title']}' ({res['total_chunks_created']} chunks created).")
            continue

        if user_input.lower() == "review":
            user_input = "Generate a literature review."

        # Dispatch via Coordinator Agent
        response = research_coordinator.handle_request(user_message=user_input)
        
        intent = response.get("intent", "research_qa")
        print(f"\n[Coordinator Routing: intent={intent}]")
        print("-" * 75)
        print(response.get("response", ""))
        print("-" * 75)

        citations = response.get("citations", [])
        if citations:
            print(f"\n[Verified Citations ({len(citations)})]:")
            for c in citations:
                print(f"  * {c.get('paper_title')} | {c.get('section_or_page')}")

if __name__ == "__main__":
    main()
