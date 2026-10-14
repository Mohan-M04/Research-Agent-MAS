# IBM Build-a-Bot (BoB) Step-by-Step Implementation Guide
## Problem Statement 7: Research Agent — "Research Companion"

This guide walks you through building, configuring, importing, and demoing **Research Companion** inside the **IBM Build-a-Bot (IBM BoB)** and **IBM watsonx Orchestrate / Langflow** platform.

---

## 🎯 Architecture Summary

- **Primary Coordinator:** `research_coordinator`
- **Sub-Agents:** 
  1. `pdf_ingestion_agent`
  2. `rag_retrieval_agent`
  3. `research_qa_agent`
  4. `literature_review_agent`
- **Knowledge Base:** `research_papers_kb`
- **Foundation Models:** `ibm/granite-3-8b-instruct` / `ibm/granite-13b-chat-v2`
- **Embeddings:** `ibm/granite-embedding-278m-multilingual`
- **Grounding Rule:** Strict RAG — refuse out-of-domain questions with `"This information is not available in the uploaded research papers."`
- **Citations:** Inline format `[Paper Title, Page/Section]`

---

## 🚀 Phase 1: Local Setup & Verification

Before importing into IBM BoB, verify all agents and tools locally:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite (100% Offline Compatible)
```bash
python tests/run_all_tests.py
```
This automatically verifies:
- PDF text extraction & title inference
- Chunking & indexing into `research_papers_kb`
- Semantic hybrid retrieval
- Strict citation formatting `[Paper Title, Page/Section]`
- Strict refusal on out-of-domain questions
- 6-section literature review generation
- All 5 problem statement test scenarios

### 3. Launch the Problem Statement 7 Interactive Dashboard
```bash
streamlit run visualization/app.py
```
This provides:
- Topic cluster scatter plots (PCA)
- Methodology radar charts
- Predictive gaps and future trajectories
- Multi-agent live execution trace

---

## ☁️ Phase 2: Building in IBM Build-a-Bot (watsonx Orchestrate)

### Step 1: Start the REST API Backend
In IBM BoB, Orchestrate skills call REST endpoints via OpenAPI 3.0:
```bash
python server.py
```
This runs the FastAPI server on `http://localhost:8000`.
*(For cloud deployment, deploy this repo to **IBM Cloud Code Engine** using `ibmcloud ce app create` to get a public HTTPS URL).*

---

### Step 2: Import Tools into watsonx Orchestrate Skill Studio

1. Log into your **IBM watsonx Orchestrate** instance in IBM BoB.
2. In the left navigation menu, click **Skill Studio** (or **Skills & Apps**).
3. Click **Add Skill** ➔ Select **From an OpenAPI specification**.
4. Upload the OpenAPI file located at:
   `orchestrate_specs/openapi_skills.yaml`
5. Orchestrate will recognize the 5 Python tools:
   - `extract_pdf_text`
   - `chunk_and_index_documents`
   - `retrieve_relevant_chunks`
   - `generate_cited_answer`
   - `generate_literature_review`
   - `research_companion_workflow`
6. Click **Enhance & Publish** to make all skills active in your catalog.

---

### Step 3: Create the Knowledge Base (`research_papers_kb`)

1. In watsonx Orchestrate / watsonx.ai, navigate to **Knowledge / Data Sources**.
2. Click **Create Knowledge Base**:
   - **Name:** `research_papers_kb`
   - **Embedding Model:** `ibm/granite-embedding-278m-multilingual`
   - **Index Fields:**
     - `paper_title` (String)
     - `authors` (List)
     - `publication_year` (Integer)
     - `section_or_page` (String)
     - `text_chunk` (Text)
3. Connect `research_papers_kb` to the `rag_retrieval_agent` and `pdf_ingestion_agent`.

---

### Step 4: Configure the 5 Agents in IBM BoB Agent Builder

Navigate to **Agent Builder** in watsonx Orchestrate:

#### 1. Configure Sub-Agent: `pdf_ingestion_agent`
- **Role:** PDF document parsing, text extraction, and vector indexing.
- **Model:** `ibm/granite-3-8b-instruct`
- **Attached Skills:** `extract_pdf_text`, `chunk_and_index_documents`
- **System Prompt:**
  ```text
  You are the PDF Ingestion Agent. Ingest uploaded research PDFs, extract metadata (title, authors, pages), clean and chunk text, and index them into research_papers_kb.
  ```

#### 2. Configure Sub-Agent: `rag_retrieval_agent`
- **Role:** Semantic and hybrid search over `research_papers_kb`.
- **Attached Skills:** `retrieve_relevant_chunks`
- **System Prompt:**
  ```text
  You are the RAG Retrieval Agent. Search research_papers_kb for top-k relevant chunks matching the user's research query. Include paper title, section/page, text, and relevance score.
  ```

#### 3. Configure Sub-Agent: `research_qa_agent`
- **Role:** Grounded question answering with inline citations.
- **Model:** `ibm/granite-3-8b-instruct` (Temperature: 0.1)
- **Attached Skills:** `generate_cited_answer`
- **System Prompt:**
  ```text
  You are the Research Q&A Agent. Answer user questions using ONLY retrieved paper content.
  CONSTRAINTS:
  1. If the information is not present in the retrieved chunks, you MUST respond:
     "This information is not available in the uploaded research papers."
  2. Every factual claim must include an inline citation: [Paper Title, Page/Section].
  3. Never use general knowledge or hallucinate.
  ```

#### 4. Configure Sub-Agent: `literature_review_agent`
- **Role:** Cross-paper synthesis and literature review generation.
- **Model:** `ibm/granite-3-8b-instruct` (Temperature: 0.15)
- **Attached Skills:** `generate_literature_review`
- **System Prompt:**
  ```text
  You are the Literature Review Agent. Generate a structured academic literature review across all uploaded papers.
  You MUST include these exact 6 sections:
  1. Background
  2. Methods and Approaches
  3. Key Findings
  4. Limitations
  5. Research Gaps
  6. Future Research Directions
  Every major claim must include at least one citation [Paper Title, Page/Section].
  ```

#### 5. Configure Main Agent: `research_coordinator`
- **Role:** Main User-Facing Orchestrator.
- **Set as:** **Primary Assistant / Front Agent**.
- **Model:** `ibm/granite-3-8b-instruct`
- **Sub-Agents Enabled:**
  - `pdf_ingestion_agent`
  - `rag_retrieval_agent`
  - `research_qa_agent`
  - `literature_review_agent`
- **Routing Rules:**
  - If user uploads files ➔ Call `pdf_ingestion_agent`.
  - If user asks for literature review ➔ Call `literature_review_agent`.
  - If user asks a question ➔ Call `rag_retrieval_agent`, then `research_qa_agent`.
  - Return final answer with verified citations.

---

## 🎨 Phase 3: IBM Langflow Alternative (Visual Flow Track)

If your IBM BoB track uses **IBM Langflow**:
1. Open your IBM Langflow canvas.
2. Click **Import Flow** (top right).
3. Select `orchestrate_specs/langflow_flow.json`.
4. The complete graph will render automatically:
   - `ChatInput` ➔ `research_coordinator` ➔ `rag_retriever` ➔ `IBM Granite 3-8B` ➔ `ChatOutput`.
5. Enter your `IBM Cloud API Key` and `Project ID` in the Granite LLM node.
6. Click **Play / Chat** to run interactive tests.

---

## 🧪 Phase 4: Hackathon Demo & Smoke Testing Checklist

Run through these 5 exact scenarios during your IBM BoB judging presentation:

| # | Test Scenario | Expected Agent Behavior | Verification Standard |
|---|---|---|---|
| 1 | Ingest 3 sample AI papers & Ask: *"What methods are used in these papers?"* | `pdf_ingestion_agent` indexes chunks. `rag_retrieval_agent` retrieves methods. `research_qa_agent` answers with citations. | Answers contain `[Attention Is All You Need, Page 2 - Methodology]`, `[Retrieval-Augmented Generation...]`, etc. |
| 2 | Ask: *"What are the main limitations identified by the authors?"* | Retrieves limitation sections across all 3 papers. | Accurately identifies O(n^2) complexity, retriever recall bottleneck, and LoRA fixed rank constraint. |
| 3 | Ask: *"What research gaps exist?"* | Retrieves research gaps and future work. | Identifies long-context modeling, multi-hop reasoning, and dynamic rank allocation. |
| 4 | Ask: *"Generate a literature review."* | `literature_review_agent` synthesizes full multi-paper review. | Formats into all 6 mandatory sections with citations throughout. |
| 5 | Ask: *"Who won the 2026 FIFA World Cup?"* (or any out-of-domain question) | Strict refusal triggered. | Response is strictly: `"This information is not available in the uploaded research papers."` |

---

## 🏆 Hackathon Winning Highlights (Problem Statement 7)

Highlight these 4 pillars to the judges:
1. **Academic Source Fusion:** Cross-paper synthesis without hallucinations.
2. **Multimodal Input:** Support for PDF research papers, text notes, and structured metadata.
3. **Predictive Agentic Layer:** Emerging trends detection and citation gap forecasting.
4. **Interactive Visualization:** Interactive 2D topic clusters, methodology radar charts, and downloadable Markdown/PDF literature reviews.
