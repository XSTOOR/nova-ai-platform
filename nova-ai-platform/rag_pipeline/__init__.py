"""
NOVA AI Platform — RAG Pipeline Module
========================================
Complete RAG pipeline: embed → store → search → rerank → evaluate.

Components:
  - NOVAEmbedder: HuggingFace embeddings with mock fallback
  - NovAVectorStore: ChromaDB vector store
  - HybridSearchEngine: Dense + sparse search with RRF
  - NOVAReranker: Cross-encoder or score-based reranking
  - RAGASEvaluator: Pipeline evaluation with 4 metrics
"""

from .embedder import NOVAEmbedder, MockEmbedder, HuggingFaceEmbedder
from .vector_store import NovAVectorStore
from .hybrid_search import HybridSearchEngine, SearchStrategy, SparseSearcher
from .reranker import NOVAReranker, CrossEncoderReranker, ScoreBasedReranker
from .ragas_eval import RAGASEvaluator, RAGASMetrics, EVAL_QUESTIONS

__all__ = [
    "NOVAEmbedder", "MockEmbedder", "HuggingFaceEmbedder",
    "NovAVectorStore",
    "HybridSearchEngine", "SearchStrategy", "SparseSearcher",
    "NOVAReranker", "CrossEncoderReranker", "ScoreBasedReranker",
    "RAGASEvaluator", "RAGASMetrics", "EVAL_QUESTIONS",
]