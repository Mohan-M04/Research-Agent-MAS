# Research Companion (IBM Build-a-Bot: Problem Statement 7)
### Multi-Agent Academic Research Assistant & Verification Engine
**Built on IBM watsonx Orchestrate • IBM Granite Models • IBM Langflow**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![watsonx Orchestrate](https://img.shields.io/badge/IBM%20watsonx-Orchestrate-0f62fe.svg)](https://www.ibm.com/products/watsonx-orchestrate)
[![IBM Granite](https://img.shields.io/badge/Foundation%20Model-Granite%203--8B-purple.svg)](https://www.ibm.com/granite)
[![Status](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)]()

---

## 📖 Overview

**Research Companion** addresses **Problem Statement 7: Research Agent** for IBM Build-a-Bot (BoB). Researchers, students, and professionals waste dozens of hours wading through dense, fragmented academic papers. Research Companion aggregates, cleans, indexes, and synthesizes multiple research papers using a coordinated multi-agent system.

It guarantees **zero-hallucination, strict citation-backed responses** (`[Paper Title, Page/Section]`) and automatically compiles structured, publication-grade **6-section literature reviews**.

---

## 🤖 The Multi-Agent System

```
                      ┌─────────────────────────────────┐
                      │      research_coordinator       │ (Main Front Agent)
                      │   - Intent Classification       │
                      │   - Routing & Citations         │
                      └────────────────┬────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│pdf_ingestion_agent│         │rag_retrieval_agent│        │literature_review_│
│  - Text Extract   │         │  - Semantic Search│        │      agent       │
│  - Chunk & Index  │         │  - Score & Rerank │        │  - 6-Section Lit │
└────────┬─────────┘          └────────┬─────────┘          │    Synthesis     │
         │                             │                    └────────┬─────────┘
         │    ┌───────────────────┐    │                             │
         └───►│research_papers_kb │◄───┘                             │
              │(Granite Embeddings│                                  │
              └───────────────────┘                                  │
                                       │                             │
                                       ▼                             │
                              ┌──────────────────┐                   │
                              │research_qa_agent │                   │
                              │ - Grounded Q&A   │                   │
                              │ - Strict Refusal │                   │
                              └────────┬─────────┘                   │
                                       │                             │
                                       ▼                             ▼
                      ┌──────────────────────────────────────────────────┐
                      │ Verified Answer / Literature Review with Citations │
                      └──────────────────────────────────────────────────┘
```

### The 5 Agents
1. **`research_coordinator`**: User-facing orchestrator. Detects whether the user is uploading PDFs, asking a research question, or requesting a literature review.
2. **`pdf_ingestion_agent`**: Extracts text & metadata, chunks documents by section/page, and creates vector embeddings.
3. **`rag_retrieval_agent`**: Performs hybrid semantic search over `research_papers_kb`, returning top relevant chunks with scores.
4. **`research_qa_agent`**: Formulates concise answers with inline citations. Refuses out-of-domain questions with: *"This information is not available in the uploaded research papers."*
5. **`literature_review_agent`**: Synthesizes cross-paper comparative analyses across 6 mandatory sections:
   1. *Background*
   2. *Methods and Approaches*
   3. *Key Findings*
   4. *Limitations*
   5. *Research Gaps*
   6. *Future Research Directions*

---

## 🛠️ The 5 Python Tools (watsonx Orchestrate ADK)

| Tool Name | Input | Output |
|---|---|---|
| `extract_pdf_text` | Uploaded PDF file or path | Clean plain text, inferred title, page-wise content, metadata |
| `chunk_and_index_documents` | Extracted text & metadata | Chunk IDs, embedding status, indexed status in `research_papers_kb` |
| `retrieve_relevant_chunks` | User research query | Top relevant chunks with paper title, page/section, source text, score |
| `generate_cited_answer` | Query & retrieved chunks | Concise answer with inline citations `[Paper Title, Page/Section]` |
| `generate_literature_review` | All chunks from KB | Synthesized 6-section literature review with inline citations |

---

## 🚀 Quickstart Guide

### 1. Setup Environment
```bash
git clone <repo_url>
cd ResearchCompanion-MAS
pip install -r requirements.txt
```

### 2. Configure Credentials (Optional for Live watsonx Cloud)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your `WATSONX_APIKEY` and `WATSONX_PROJECT_ID`.

### 3. Run All Automated Unit & Smoke Tests
```bash
python tests/run_all_tests.py
```

### 4. Start the Interactive Visualization Dashboard
```bash
streamlit run visualization/app.py
```
Open `http://localhost:8501` to test the live chat, review generator, and 2D topic cluster visualizer.

### 5. Launch the REST API for watsonx Orchestrate Skill Studio
```bash
python server.py
```
Open `http://localhost:8000/docs` to view the interactive Swagger/OpenAPI UI.

---

## 📑 IBM Build-a-Bot (BoB) Deployment

Read the complete deployment instructions in:
👉 [IBM_BOB_STEP_BY_STEP_GUIDE.md](IBM_BOB_STEP_BY_STEP_GUIDE.md)

- Upload `orchestrate_specs/openapi_skills.yaml` into watsonx Orchestrate Skill Studio.
- Import `orchestrate_specs/langflow_flow.json` into IBM Langflow.
- Deploy the agent configuration in `orchestrate_specs/agent_specs.yaml`.

---

## 🔬 Sample Research Papers Included
Inside `sample_data/`:
1. *Attention Is All You Need* (Vaswani et al., 2017)
2. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
3. *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
