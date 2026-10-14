# Research Companion: watsonx Multi-Agent Research Assistant
# Configuration for IBM watsonx.ai and watsonx Orchestrate

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class WatsonxConfig:
    """watsonx.ai & Orchestrate connection parameters."""
    api_key: str = os.getenv("WATSONX_APIKEY", "")
    project_id: str = os.getenv("WATSONX_PROJECT_ID", "")
    url: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    
    # Primary model for reasoning, QA, and literature review
    model_id: str = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")
    embedding_model_id: str = os.getenv("WATSONX_EMBEDDING_MODEL_ID", "ibm/granite-embedding-278m-multilingual")
    
    temperature: float = 0.1
    max_new_tokens: int = 1500
    top_p: float = 0.95
    repetition_penalty: float = 1.05

@dataclass
class KnowledgeBaseConfig:
    """Knowledge base indexing parameters."""
    kb_name: str = "research_papers_kb"
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k: int = 5
    similarity_threshold: float = 0.35

watsonx_config = WatsonxConfig()
kb_config = KnowledgeBaseConfig()
