# IBM University Engagement Project Submission
## "Exploring the Power of Agentic AI with IBM Granite and IBM Bob"
### Problem Statement No. 7: Research Agent
### Project Title: Research Companion — Multi-Agent Academic Research Synthesis & Verification Engine

---

## 📋 SLIDE-BY-SLIDE PRESENTATION CONTENT

*Recommended formatting:*
- **Font Style:** Arial
- **Font Size (Heading):** 28 pt
- **Font Size (Content):** 20 pt

---

### SLIDE 1: Title & Student Information

**Heading (28 pt):**  
IBM University Engagement Project Submission: Agentic AI with IBM Bob

**Content (20 pt):**
- **Project Title:** Research Companion: Multi-Agent Academic Research Assistant
- **Problem Statement:** No. 7 — Research Agent
- **Domain of Project:** Education & Academic Research / AI Knowledge Systems
- **Student Name:** Mohan M
- **Email ID:** mohanachar20054@gmail.com
- **WhatsApp / Mobile:** [Your Phone Number]
- **GitHub Repository:** https://github.com/Mohan-M04/Research-Agent-MAS
- *[Insert Recent Passport Photo]*

---

### SLIDE 2: Problem Statement No. 7: Research Agent

**Heading (28 pt):**  
Problem Statement: Research Agent — The Challenge & Objective

**Content (20 pt):**
- **The Challenge:**
  - Researchers, students, and professionals waste dozens of hours navigating scattered academic papers across multiple platforms.
  - Synthesizing comparative insights, detecting methodology gaps, and ensuring accurate, cited facts without AI hallucinations is a major hurdle.

- **The Objective:**
  - Build an intelligent, multi-agent research companion that fuses academic papers, provides strict citation-backed answers (`[Paper Title, Page/Section]`), predicts future research gaps, and automatically compiles structured 6-section literature reviews.

---

### SLIDE 3: Proposed Solution: AI-Powered Research Companion

**Heading (28 pt):**  
Proposed Solution: Research Companion Architecture

**Content (20 pt):**
- **Multi-Agent Orchestration:** Powered by IBM Bob, watsonx Orchestrate, and IBM Granite foundation models.
- **RAG Knowledge Base (`research_papers_kb`):** Stores paper chunks with page/section metadata and Granite vector embeddings.
- **Collaborating Multi-Agent Team:**
  1. **`research_coordinator`:** Main user-facing front agent; classifies user intent and orchestrates routing.
  2. **`pdf_ingestion_agent`:** Parses PDFs, extracts text/metadata, and chunks into vector DB.
  3. **`rag_retrieval_agent`:** Performs hybrid semantic search over indexed literature.
  4. **`research_qa_agent`:** Generates concise answers grounded strictly in retrieved chunks with citations.
  5. **`literature_review_agent`:** Synthesizes comprehensive 6-section academic literature reviews.

---

### SLIDE 4: Technology Stack

**Heading (28 pt):**  
Technology Stack & IBM Ecosystem

**Content (20 pt):**
- **IBM Bob Platform:** For designing, configuring, and orchestrating multi-agent workflows.
- **IBM Granite Models (`ibm/granite-3-8b-instruct`):** Powers reasoning, strict grounded Q&A, and literature review synthesis.
- **IBM watsonx.ai:** Enterprise foundation model deployment, embeddings (`ibm/granite-embedding-278m-multilingual`), and AI guardrails.
- **Hybrid RAG Knowledge Base (`research_papers_kb`):** In-memory semantic vector indexing with strict page and section metadata tracking.
- **OpenAPI 3.0 & watsonx ADK:** Modular tool architecture allowing one-click skill publishing to Orchestrate Skill Studio.
- **Deployment & UI:** FastAPI REST backend (port 3000), terminal CLI (`chat.py`), and interactive Plotly visualizer.

---

### SLIDE 5: Architecture Blueprint for IBM Bob

**Heading (28 pt):**  
Architecture Blueprint: Multi-Agent Execution Flow

**Content (20 pt):**
```
[User Request: Questions / PDF Upload / Lit Review]
                      │
                      ▼
            ┌───────────────────────┐
            │  research_coordinator  │ (Intent Classification & Verification)
            └───────────┬───────────┘
                        │
         ┌──────────────┼──────────────┬──────────────────┐
         │              │              │                  │
         ▼              ▼              ▼                  ▼
    ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────┐
    │   PDF    │  │    RAG    │  │ Research  │  │ Literature Review │
    │Ingestion │  │ Retrieval │  │    Q&A    │  │       Agent       │
    │  Agent   │  │   Agent   │  │   Agent   │  │                   │
    └────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────────┬─────────┘
         │              │              │                  │
         ▼              ▼              ▼                  ▼
    ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────┐
    │extract & │  │ retrieve  │  │ generate  │  │ generate 6-section│
    │  chunk   │  │  chunks   │  │ cited ans │  │  lit review with  │
    │  tools   │  │   tool    │  │   tool    │  │    citations      │
    └────┬─────┘  └─────┬─────┘  └───────────┘  └───────────────────┘
         │              │
         └───────┬──────┘
                 ▼
    ┌───────────────────────────┐
    │    research_papers_kb     │ (Granite Embeddings & Page Metadata)
    └───────────────────────────┘
```

---

### SLIDE 6: Role of Agentic AI in the Solution

**Heading (28 pt):**  
Role of Agentic AI in Research Companion

**Content (20 pt):**
- **Autonomous Task Routing:** Unlike static chatbots, Agentic AI identifies whether the user wants document ingestion, question answering, or multi-paper synthesis, dispatching tasks to dedicated agents.
- **Collaborative Synthesis:** Multiple specialized agents coordinate to solve complex queries—ingestion parses PDFs, retrieval selects relevant evidence, and the QA agent validates citations.
- **Strict Anti-Hallucination Guardrails:** If requested facts do not exist in the uploaded papers, the agent autonomously refuses rather than speculating.
- **Continuous Knowledge Indexing:** New documents are dynamically vectorized, chunked, and indexed without retraining the underlying foundation models.

---

### SLIDE 7: Token Usage & Performance Metrics

**Heading (28 pt):**  
IBM Granite Token Usage & Efficiency

**Content (20 pt):**
- **Token Optimization Strategy:**
  - Chunk size bounded to 500 characters with 80-character sliding overlap.
  - Top-5 semantic chunk retrieval limits prompt token footprint to ~1,200 tokens per query.
- **Before Prompt vs. After Prompt Metrics:**
  - Baseline Prompt Token Footprint: ~150 tokens (system instructions + query).
  - Grounded Context Tokens (Top Chunks): ~1,100 tokens.
  - Granite 3-8B Generation Output: ~250 tokens per cited answer, ~1,800 tokens for full 6-section literature review.
- **Speed & Latency:** Near-instant retrieval (<0.08s) using localized vector indexing.

---

### SLIDE 8: Project Output Screenshot 1 — Methodologies & Limitations

**Heading (28 pt):**  
Project Output: Grounded Research Q&A with Citations

**Content (20 pt):**
- **User Prompt 1:** *"What methods are used in these papers?"*
  - **Agent Output:** Synthesizes Multi-Head Self-Attention (Transformer), Dense Passage Retrieval (RAG), and Low-Rank Decomposition (LoRA).
  - **Inline Citations:** `[Attention Is All You Need, Page 2 - Methodology]`, `[LoRA, Page 2 - Methodology]`.
- **User Prompt 2:** *"What are the main limitations identified by the authors?"*
  - **Agent Output:** Identifies $O(n^2)$ attention memory bottleneck, retriever recall sensitivity, and static rank allocation constraints.
- *[Insert Screenshot of Web UI / Terminal Output]*

---

### SLIDE 9: Project Output Screenshot 2 — 6-Section Literature Review

**Heading (28 pt):**  
Project Output: Synthesized 6-Section Literature Review

**Content (20 pt):**
- **User Prompt:** *"Generate a literature review."*
- **Agent Output:** Complete multi-paper academic synthesis with mandatory sections:
  1. *Background*
  2. *Methods and Approaches*
  3. *Key Findings*
  4. *Limitations*
  5. *Research Gaps*
  6. *Future Research Directions*
- Every section includes grounded citations linked to specific pages and sections.
- *[Insert Screenshot of Literature Review Output]*

---

### SLIDE 10: Project Output Screenshot 3 — Anti-Hallucination & Refusal

**Heading (28 pt):**  
Project Output: Strict Grounding & Anti-Hallucination Defense

**Content (20 pt):**
- **User Prompt:** *"Who won the 2026 FIFA World Cup?"* (Out-of-domain query)
- **Agent Response:**  
  > **`"This information is not available in the uploaded research papers."`**
- **Evaluation Significance:** Proves the system strictly adheres to source literature and will never hallucinate or draw from unauthorized general knowledge.
- *[Insert Screenshot of Refusal Response]*

---

### SLIDE 11: Novelty and Uniqueness

**Heading (28 pt):**  
Novelty and Key Differentiators

**Content (20 pt):**
- **Strict Zero-Hallucination Guarantee:** Enforces deterministic refusal whenever evidence is missing.
- **Granular Page & Section Citations:** Every claim is cited as `[Paper Title, Page/Section]` for auditability.
- **Automated Academic Synthesis:** Generates full 6-section reviews in seconds, saving researchers hours of manual labor.
- **Interactive Visual Insights (Objective 4):** 2D PCA semantic topic clusters, methodology radar tradeoff analysis, and predictive research gap forecasting.
- **Enterprise Ready on IBM Bob:** Fully compatible with IBM watsonx Orchestrate ADK, OpenAPI 3.0, and IBM Langflow.

---

### SLIDE 12: GitHub Submission & Deliverables

**Heading (28 pt):**  
GitHub Repository & Deliverables Checklist

**Content (20 pt):**
- **Public GitHub Repository:**  
  👉 **https://github.com/Mohan-M04/Research-Agent-MAS**
- **Uploaded Assets Checklist:**
  1. ✅ All 5 Orchestrate Agent specifications (`agents/`)
  2. ✅ All 5 Python tools (`tools/`)
  3. ✅ Knowledge base implementation (`knowledge_base/`)
  4. ✅ Agentic workflow (`workflows/`)
  5. ✅ OpenAPI 3.0 specification (`orchestrate_specs/openapi_skills.yaml`)
  6. ✅ Automated test suite & smoke tests (`tests/`)
  7. ✅ Unified Web UI (port 3000) & CLI client (`chat.py`)

---

### SLIDE 13: Future Scope

**Heading (28 pt):**  
Future Scope & Commercial Potential

**Content (20 pt):**
- **1. Multimodal Scientific Figure & Table Extraction:** Integrating Granite Vision models to parse complex charts, mathematical equations, and experimental plots from academic PDFs.
- **2. Live ArXiv & CrossRef Streaming:** Connecting live academic APIs to automatically fetch preprints and update citation graphs dynamically.
- **3. Multilingual Academic Translation:** Enabling real-time translation of papers into multiple languages to democratize research access globally.

---

### SLIDE 14 & 15: Certificates Earned

**Heading (28 pt):**  
IBM SkillsBuild & IBM Bob Certifications

**Content (20 pt):**
- *[Insert your Credly Certificate: Generative AI in Action]*
- *[Insert your IBM SkillsBuild Certificate: Lab: Troubleshoot Your Code Using IBM Bob]*

---

### SLIDE 16: Thank You!

**Heading (28 pt):**  
Thank You!

**Content (20 pt):**
- *Research Companion: Empowering researchers with trustworthy Agentic AI.*
- **Presenter:** Mohan M
- **GitHub:** https://github.com/Mohan-M04/Research-Agent-MAS
- **Powered by:** IBM watsonx Orchestrate & IBM Granite
