"""
Visualization Components for Problem Statement 7:
- Topic Clustering
- Citation & Knowledge Graph
- Emerging Trends Analysis
- Methodology Radar Comparison
"""

from typing import List, Dict, Any

def get_sample_topic_clusters() -> Dict[str, Any]:
    """Generates 2D cluster points for uploaded papers and methodologies."""
    return {
        "x": [1.2, 1.5, 1.8, -2.1, -1.9, -2.4, 0.4, 0.6, 0.2],
        "y": [2.3, 2.7, 2.1, 1.1, 0.8, 1.4, -2.2, -1.9, -2.5],
        "label": [
            "Transformer: Self-Attention", "Transformer: Multi-Head", "Transformer: Scaled Dot-Product",
            "RAG: Dense Passage Retrieval", "RAG: Token Generator", "RAG: Non-parametric Memory",
            "LoRA: Low-Rank Decomposition", "LoRA: Parameter Efficiency", "LoRA: Adapter Weights"
        ],
        "paper": [
            "Attention Is All You Need", "Attention Is All You Need", "Attention Is All You Need",
            "RAG (Lewis et al.)", "RAG (Lewis et al.)", "RAG (Lewis et al.)",
            "LoRA (Hu et al.)", "LoRA (Hu et al.)", "LoRA (Hu et al.)"
        ],
        "cluster": ["Attention Mechanisms", "Attention Mechanisms", "Attention Mechanisms",
                    "Retrieval Augmentation", "Retrieval Augmentation", "Retrieval Augmentation",
                    "Parameter Efficient Tuning", "Parameter Efficient Tuning", "Parameter Efficient Tuning"]
    }

def get_citation_graph_data() -> Dict[str, Any]:
    """Nodes and edges for citation map and knowledge graph."""
    nodes = [
        {"id": "Transformer", "label": "Attention Is All You Need\n(Vaswani 2017)", "group": "Foundation Architecture", "size": 30},
        {"id": "BART", "label": "BART / Seq2Seq Generator", "group": "Generator", "size": 20},
        {"id": "DPR", "label": "Dense Passage Retrieval (DPR)", "group": "Retriever", "size": 20},
        {"id": "RAG", "label": "RAG for Knowledge-Intensive NLP\n(Lewis 2020)", "group": "Hybrid Architecture", "size": 28},
        {"id": "GPT3", "label": "Large Language Models (GPT-3)", "group": "Base LLM", "size": 22},
        {"id": "LoRA", "label": "LoRA Low-Rank Adaptation\n(Hu 2021)", "group": "Efficiency & Adaptation", "size": 28},
        {"id": "QLoRA", "label": "Dynamic Quantization (QLoRA 2023+)", "group": "Future Trend", "size": 18},
        {"id": "AgenticRAG", "label": "Agentic & Iterative RAG (2024+)", "group": "Future Trend", "size": 18}
    ]

    edges = [
        {"from": "Transformer", "to": "BART", "label": "Architecture Basis"},
        {"from": "Transformer", "to": "GPT3", "label": "Architecture Basis"},
        {"from": "BART", "to": "RAG", "label": "Parametric Generator"},
        {"from": "DPR", "to": "RAG", "label": "Non-Parametric Memory"},
        {"from": "GPT3", "to": "LoRA", "label": "Target for Fine-Tuning"},
        {"from": "Transformer", "to": "LoRA", "label": "Attention Weight Injection"},
        {"from": "LoRA", "to": "QLoRA", "label": "Evolutionary Trajectory"},
        {"from": "RAG", "to": "AgenticRAG", "label": "Agentic Trajectory"}
    ]

    return {"nodes": nodes, "edges": edges}

def get_predictive_trends_data() -> List[Dict[str, Any]]:
    """Predictive insights: emerging trends, gaps, and future trajectories."""
    return [
        {
            "dimension": "Parameter Efficiency",
            "current_state": "LoRA adapters freeze 99.9% weights",
            "citation_gap": "Static rank selection causes suboptimal capacity allocation",
            "predicted_direction": "Dynamic layer-wise rank routing and unified quantization (QLoRA+)",
            "impact_score": 92
        },
        {
            "dimension": "Factuality & Memory",
            "current_state": "RAG couples neural retriever with frozen seq2seq model",
            "citation_gap": "Single-step retrieval fails at multi-hop reasoning and table synthesis",
            "predicted_direction": "Agentic iterative graph-retrieval with self-reflective critique loops",
            "impact_score": 96
        },
        {
            "dimension": "Context Window Scaling",
            "current_state": "Vanilla self-attention exhibits O(n^2) quadratic complexity",
            "citation_gap": "Loss of fine-grained token attention in sub-quadratic approximations",
            "predicted_direction": "Hybrid state-space models (Mamba) coupled with sparse linear attention",
            "impact_score": 88
        }
    ]
