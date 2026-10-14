# Research Companion — watsonx Orchestrate Multi-Agent Solution

**Problem Statement 7: Research Agent**

Research Companion helps students, researchers, and professionals quickly understand multiple research papers. Users upload PDF research papers and ask questions. The solution retrieves information only from the uploaded papers and generates accurate, citation-backed answers and structured literature reviews.

---

## Project Structure

```
research_companion/
├── README.md
├── requirements.txt
├── .env.example
├── config.toml.example
├── agents/
│   ├── __init__.py
│   ├── research_coordinator.py
│   ├── pdf_ingestion_agent.py
│   ├── rag_retrieval_agent.py
│   ├── research_qa_agent.py
│   └── literature_review_agent.py
├── tools/
│   ├── __init__.py
│   ├── extract_pdf_text.py
│   ├── chunk_and_index_documents.py
│   ├── retrieve_relevant_chunks.py
│   ├── generate_cited_answer.py
│   └── generate_literature_review.py
├── knowledge_base/
│   ├── __init__.py
│   └── research_papers_kb.py
├── workflow/
│   ├── __init__.py
│   └── research_companion_workflow.py
├── deploy/
│   ├── __init__.py
│   └── deploy_assets.py
└── tests/
    ├── __init__.py
    ├── test_extract_pdf_text.py
    ├── test_chunk_and_index_documents.py
    ├── test_retrieve_relevant_chunks.py
    ├── test_generate_cited_answer.py
    ├── test_generate_literature_review.py
    └── smoke_test_coordinator.py
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your watsonx Orchestrate / watsonx.ai credentials
```

### 3. Configure deployment
```bash
cp config.toml.example config.toml
# Edit config.toml with your region and model IDs
```

---

## User Flow

1. Upload one or more research-paper PDFs.
2. The system extracts text, chunks, embeds, and indexes the papers.
3. Ask a research question (e.g., "What methods are used in these papers?").
4. Get accurate, citation-backed answers.
5. Optionally request: "Generate a literature review."

---

## Agents

| Agent | Role |
|---|---|
| `research_coordinator` | Main user-facing router; dispatches to sub-agents |
| `pdf_ingestion_agent` | Extracts, cleans, chunks, and indexes PDF content |
| `rag_retrieval_agent` | Semantic retrieval over indexed paper chunks |
| `research_qa_agent` | Answers questions with inline citations |
| `literature_review_agent` | Generates structured literature reviews |

---

## Python Tools

| Tool | Purpose |
|---|---|
| `extract_pdf_text` | Extract text and metadata from PDF |
| `chunk_and_index_documents` | Chunk, embed, and index extracted text |
| `retrieve_relevant_chunks` | Semantic search over knowledge base |
| `generate_cited_answer` | Produce cited answers from retrieved chunks |
| `generate_literature_review` | Produce structured literature review |

---

## Deployment

```bash
python deploy/deploy_assets.py
```

---

## Testing

```bash
# Unit tests
pytest tests/ -v

# Smoke test against deployed coordinator
python tests/smoke_test_coordinator.py
```

---

## Restrictions

- Answers are sourced **only** from uploaded research papers.
- Fabrication of titles, authors, page numbers, or findings is prohibited.
- When information is absent, the system responds: *"This information is not available in the uploaded research papers."*
- No medical, legal, financial, or safety-critical advice.
