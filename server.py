"""
FastAPI Server for Research Companion
Runs directly on http://localhost:3000.
Exposes REST/OpenAPI 3.0 skills + a clean web UI.
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from tools.extract_pdf_text import extract_pdf_text
from tools.chunk_and_index_documents import chunk_and_index_documents
from tools.retrieve_relevant_chunks import retrieve_relevant_chunks
from tools.generate_cited_answer import generate_cited_answer
from tools.generate_literature_review import generate_literature_review
from workflows.research_companion_workflow import research_companion_workflow
from knowledge_base.kb_manager import global_kb

# Preload sample research papers into KB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sample_dir = os.path.join(BASE_DIR, "sample_data")
if os.path.exists(sample_dir) and len(global_kb.chunks) == 0:
    for p in ["attention_is_all_you_need.txt", "rag_knowledge_intensive.txt", "lora_low_rank_adaptation.txt"]:
        fpath = os.path.join(sample_dir, p)
        if os.path.exists(fpath):
            ext = extract_pdf_text(fpath, file_name=p)
            chunk_and_index_documents(ext)

app = FastAPI(
    title="Research Companion - watsonx Orchestrate Multi-Agent API",
    description="IBM Build-a-Bot (BoB) Problem Statement 7: Research Agent implementation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExtractPDFRequest(BaseModel):
    file_path_or_bytes: str
    file_name: Optional[str] = ""

class ChunkIndexRequest(BaseModel):
    extracted_data: Dict[str, Any]
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 80

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    paper_filter: Optional[str] = None

class CitedAnswerRequest(BaseModel):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    use_watsonx_live: Optional[bool] = False

class LitReviewRequest(BaseModel):
    domain_topic: Optional[str] = "Comparative Study of Uploaded Research Papers"
    use_watsonx_live: Optional[bool] = False

class WorkflowRequest(BaseModel):
    user_request: str
    uploaded_files: Optional[List[str]] = None
    file_names: Optional[List[str]] = None
    use_watsonx_live: Optional[bool] = False


# HTML Web Client on localhost:3000
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Research Companion | IBM watsonx Orchestrate (Port 3000)</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'IBM Plex Sans', sans-serif; }
    body { background: #f4f4f4; color: #161616; height: 100vh; display: flex; flex-direction: column; }
    header { background: #161616; color: #fff; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #0f62fe; }
    header h1 { font-size: 1.25rem; font-weight: 600; display: flex; align-items: center; gap: 10px; }
    .badge { background: #0f62fe; color: #fff; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; }
    .container { display: flex; flex: 1; overflow: hidden; }
    .sidebar { width: 320px; background: #ffffff; border-right: 1px solid #e0e0e0; padding: 20px; display: flex; flex-direction: column; gap: 20px; }
    .sidebar h3 { font-size: 0.95rem; text-transform: uppercase; color: #525252; letter-spacing: 0.5px; }
    .paper-list { list-style: none; font-size: 0.85rem; color: #393939; }
    .paper-list li { padding: 8px 10px; margin-bottom: 6px; background: #f4f4f4; border-left: 3px solid #0f62fe; border-radius: 2px; }
    .chat-area { flex: 1; display: flex; flex-direction: column; background: #f4f7fb; }
    .messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
    .message { max-width: 80%; padding: 14px 18px; border-radius: 8px; font-size: 0.92rem; line-height: 1.5; }
    .message.user { align-self: flex-end; background: #0f62fe; color: #fff; border-bottom-right-radius: 1px; }
    .message.bot { align-self: flex-start; background: #ffffff; color: #161616; border: 1px solid #e0e0e0; border-bottom-left-radius: 1px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .citation { display: inline-block; background: #edf5ff; border: 1px solid #0f62fe; color: #0043ce; padding: 2px 6px; border-radius: 4px; font-size: 0.8rem; margin: 2px 0; font-weight: 500; }
    .quick-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    .chip { background: #e0e0e0; border: none; padding: 6px 12px; border-radius: 16px; font-size: 0.8rem; cursor: pointer; transition: 0.2s; }
    .chip:hover { background: #0f62fe; color: #fff; }
    .input-box { background: #fff; padding: 16px 24px; border-top: 1px solid #e0e0e0; display: flex; gap: 12px; }
    .input-box input { flex: 1; padding: 12px 16px; border: 1px solid #8d8d8d; border-radius: 4px; font-size: 0.95rem; outline: none; }
    .input-box input:focus { border-color: #0f62fe; box-shadow: 0 0 0 2px rgba(15,98,254,0.2); }
    .input-box button { background: #0f62fe; color: #fff; border: none; padding: 12px 24px; border-radius: 4px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    .input-box button:hover { background: #0043ce; }
  </style>
</head>
<body>
  <header>
    <h1>🎓 Research Companion <span class="badge">localhost:3000</span></h1>
    <div>
      <a href="/docs" target="_blank" style="color:#78a9ff; text-decoration:none; font-size:0.85rem; margin-right:15px;">📖 OpenAPI Swagger Docs</a>
      <span style="font-size:0.85rem; color:#c6c6c6;">IBM Build-a-Bot • PS 7</span>
    </div>
  </header>

  <div class="container">
    <div class="sidebar">
      <div>
        <h3>Active Agents</h3>
        <ul style="list-style:none; margin-top:8px; font-size:0.85rem; line-height:1.8;">
          <li>🤖 <b>research_coordinator</b></li>
          <li>📥 pdf_ingestion_agent</li>
          <li>🔍 rag_retrieval_agent</li>
          <li>🛡️ research_qa_agent</li>
          <li>📑 literature_review_agent</li>
        </ul>
      </div>

      <div>
        <h3>Indexed Papers in KB</h3>
        <ul class="paper-list" id="papers">
          <li>Attention Is All You Need (2017)</li>
          <li>RAG for Knowledge Tasks (2020)</li>
          <li>LoRA: Low-Rank Adaptation (2021)</li>
        </ul>
      </div>

      <div style="margin-top:auto;">
        <button onclick="ask('review')" style="width:100%; background:#24a148; color:#fff; border:none; padding:10px; border-radius:4px; font-weight:600; cursor:pointer;">📑 Generate Literature Review</button>
      </div>
    </div>

    <div class="chat-area">
      <div class="messages" id="chat">
        <div class="message bot">
          👋 <b>Welcome to Research Companion!</b><br>
          I am running natively on <b>localhost:3000</b>. You can ask research questions, extract methods, findings, limitations, or request a full 6-section literature review.
        </div>
      </div>

      <div class="input-box">
        <div style="display:flex; flex-direction:column; width:100%;">
          <div class="quick-chips">
            <button class="chip" onclick="ask('What methods are used in these papers?')">🔍 Methods</button>
            <button class="chip" onclick="ask('What are the main limitations identified by the authors?')">⚠️ Limitations</button>
            <button class="chip" onclick="ask('What research gaps exist?')">❓ Research Gaps</button>
            <button class="chip" onclick="ask('review')">📑 Full Review</button>
            <button class="chip" onclick="ask('Who won the 2026 FIFA World Cup?')">🛡️ Test Refusal</button>
          </div>
          <div style="display:flex; gap:10px;">
            <input type="text" id="userInput" placeholder="Ask a research question or request a review..." onkeydown="if(event.key==='Enter') send()">
            <button onclick="send()">Send</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function send() {
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      await ask(text);
    }

    async function ask(query) {
      const chat = document.getElementById('chat');
      chat.innerHTML += `<div class="message user">${escapeHtml(query)}</div>`;
      chat.scrollTop = chat.scrollHeight;

      const loadingId = 'loading-' + Date.now();
      chat.innerHTML += `<div class="message bot" id="${loadingId}"><i>research_coordinator synthesizing response...</i></div>`;
      chat.scrollTop = chat.scrollHeight;

      try {
        const res = await fetch('/workflow/research_companion_workflow', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ user_request: query })
        });
        const data = await res.json();
        const el = document.getElementById(loadingId);
        
        let formatted = data.final_response || "No response.";
        formatted = formatted.replace(/\\[(.*?)\\]/g, '<span class="citation">[$1]</span>');
        formatted = formatted.replace(/\\n/g, '<br>');
        
        el.innerHTML = formatted;
      } catch (err) {
        document.getElementById(loadingId).innerHTML = '<span style="color:#da1e28;">Error connecting to API.</span>';
      }
      chat.scrollTop = chat.scrollHeight;
    }

    function escapeHtml(text) {
      return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root_ui():
    """Serves the web client directly on http://localhost:3000"""
    return HTMLResponse(content=HTML_UI)

@app.get("/api/info")
def api_info():
    return {
        "status": "online",
        "solution": "Research Companion",
        "port": 3000,
        "knowledge_base_chunks": len(global_kb.chunks),
        "docs_url": "/docs"
    }

@app.post("/tools/extract_pdf_text")
def api_extract_pdf_text(req: ExtractPDFRequest):
    return extract_pdf_text(req.file_path_or_bytes, req.file_name or "")

@app.post("/tools/chunk_and_index_documents")
def api_chunk_and_index_documents(req: ChunkIndexRequest):
    return chunk_and_index_documents(req.extracted_data, req.chunk_size, req.chunk_overlap)

@app.post("/tools/retrieve_relevant_chunks")
def api_retrieve_relevant_chunks(req: RetrieveRequest):
    return retrieve_relevant_chunks(req.query, req.top_k, req.paper_filter)

@app.post("/tools/generate_cited_answer")
def api_generate_cited_answer(req: CitedAnswerRequest):
    return generate_cited_answer(req.query, req.retrieved_chunks, req.use_watsonx_live)

@app.post("/tools/generate_literature_review")
def api_generate_literature_review(req: LitReviewRequest):
    return generate_literature_review(domain_topic=req.domain_topic, use_watsonx_live=req.use_watsonx_live)

@app.post("/workflow/research_companion_workflow")
def api_workflow(req: WorkflowRequest):
    return research_companion_workflow.execute(
        user_request=req.user_request,
        uploaded_files=req.uploaded_files,
        file_names=req.file_names,
        use_watsonx_live=req.use_watsonx_live
    )

@app.get("/kb/status")
def kb_status():
    return {
        "knowledge_base": global_kb.name,
        "total_chunks": len(global_kb.chunks),
        "total_papers": len(global_kb.papers),
        "papers": list(global_kb.papers.values())
    }

@app.post("/kb/reset")
def kb_reset():
    global_kb.clear()
    return {"status": "cleared", "knowledge_base": global_kb.name}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    print(f"🚀 Starting Research Companion Server on http://localhost:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
