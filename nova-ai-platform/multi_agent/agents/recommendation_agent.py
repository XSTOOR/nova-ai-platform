"""
NOVA AI Platform — Recommendation Agent
==========================================
Handles product inquiries using the RAG pipeline for knowledge retrieval.

Uses:
  - Task 3: RAG Pipeline (embedding, vector search, reranking)
  - Task 4: Brand voice inference for response formatting
  - Task 1: COSTAR prompt templates
"""

import logging
import time
from typing import Optional

from multi_agent.state import NOVAState
from rag_pipeline.vector_store import NovAVectorStore
from rag_pipeline.reranker import NOVAReranker

logger = logging.getLogger(__name__)


def _chroma_to_docs(query_result: dict) -> list[dict]:
    """Convert ChromaDB query result format to list of doc dicts."""
    if not query_result:
        return []

    documents = query_result.get("documents", [[]])
    metadatas = query_result.get("metadatas", [[]])
    distances = query_result.get("distances", [[]])
    ids = query_result.get("ids", [[]])

    # Handle nested lists from ChromaDB
    if documents and isinstance(documents[0], list):
        documents = documents[0]
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if distances and isinstance(distances[0], list):
        distances = distances[0]
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    docs = []
    for i in range(len(documents)):
        doc = {
            "document": documents[i] if i < len(documents) else "",
            "id": ids[i] if i < len(ids) else f"doc_{i}",
            "score": 1.0 - (distances[i] if i < len(distances) else 0.5),
            "type": "unknown",
        }
        if i < len(metadatas) and metadatas[i]:
            doc.update(metadatas[i])
        docs.append(doc)

    return docs


class RecommendationAgent:
    """
    Recommendation agent — uses RAG pipeline for product knowledge.

    Capabilities:
      - Semantic search across products + FAQs
      - Hybrid search (dense + sparse) with RRF fusion
      - Cross-encoder reranking for precision
      - Brand voice formatted responses
    """

    def __init__(
        self,
        vector_store: Optional[NovAVectorStore] = None,
        reranker: Optional[NOVAReranker] = None,
    ):
        self._store = vector_store or NovAVectorStore()
        self._reranker = reranker or NOVAReranker()

        # Build the vector store if not already built
        try:
            self._store.build()
        except Exception:
            logger.warning("Vector store build failed — will use empty store")

    def process(self, state: NOVAState) -> NOVAState:
        """
        Process a product inquiry using the RAG pipeline.
        """
        start = time.perf_counter()
        message = state.get("current_message", "")
        audit_entry = {
            "agent": "recommendation",
            "step": "process",
            "timestamp": time.time(),
        }

        rag_context = []
        rag_scores = []

        try:
            # Step 1: Retrieve from vector store
            query_results = self._store.query(message, n_results=5)

            # Convert ChromaDB format to list of dicts
            docs = _chroma_to_docs(query_results)

            if docs:
                # Step 2: Rerank results
                reranked = self._reranker.rerank(message, docs, top_k=3)
                rag_context = reranked

                # Extract scores
                rag_scores = [
                    r.get("rerank_score", r.get("score", 0.0))
                    for r in reranked
                ]

        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            rag_context = []
            rag_scores = []

        state["rag_context"] = rag_context
        state["rag_scores"] = rag_scores

        audit_entry.update({
            "action": "rag_retrieval",
            "docs_retrieved": len(rag_context),
            "top_score": rag_scores[0] if rag_scores else 0.0,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        })
        state["audit_trail"] = state.get("audit_trail", []) + [audit_entry]

        return state