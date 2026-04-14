"""
NOVA AI Platform — Re-ranker
===============================
Re-ranks search results using cross-encoder scoring for improved relevance.

Two modes:
  1. CROSS_ENCODER: Uses sentence-transformers cross-encoder model
  2. SCORE_BASED: Fallback using combined vector + keyword scores

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import logging
from typing import Optional

from config.settings import Config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Cross-Encoder Reranker
# ──────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """Re-ranks results using a cross-encoder model."""

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or Config.reranker.model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self._model_name}")
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank search results using cross-encoder scoring.

        Args:
            query: The original query.
            results: List of search result dicts with 'document' key.
            top_k: Number of results to return.

        Returns:
            Re-ranked list of results.
        """
        if not results:
            return []

        self._load_model()

        # Score each (query, document) pair
        pairs = [(query, r.get("document", r.get("metadata", {}).get("name", ""))) for r in results]
        scores = self._model.predict(pairs)

        # Combine with original scores
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
            result["original_score"] = result.get("score", 0)
            result["score"] = float(scores[i])  # Use rerank score as primary

        # Sort by rerank score
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return results[:top_k]


# ──────────────────────────────────────────────────────────────────────
# Score-Based Reranker (Fallback)
# ──────────────────────────────────────────────────────────────────────

class ScoreBasedReranker:
    """
    Fallback reranker using heuristic scoring when cross-encoder unavailable.

    Combines: retrieval_score + keyword_overlap + metadata_boosts
    """

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank using combined heuristic scoring.

        Args:
            query: The original query.
            results: Search results.
            top_k: Number of results.

        Returns:
            Re-ranked results.
        """
        if not results:
            return []

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for result in results:
            score = result.get("score", result.get("rrf_score", 0.5))

            # Boost for keyword overlap in document
            doc_text = result.get("document", "").lower()
            overlap = sum(1 for term in query_terms if term in doc_text)
            keyword_boost = overlap / max(len(query_terms), 1) * 0.3
            score += keyword_boost

            # Boost for metadata name match
            name = result.get("metadata", {}).get("name", "").lower()
            for term in query_terms:
                if term in name:
                    score += 0.1

            # Boost bestsellers
            if result.get("metadata", {}).get("bestseller"):
                score += 0.05

            # Boost high-rated products
            rating = result.get("metadata", {}).get("rating", 0)
            if rating > 4.5:
                score += 0.05

            result["rerank_score"] = round(score, 4)
            result["original_score"] = result.get("score", 0)
            result["score"] = round(score, 4)

        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return results[:top_k]


# ──────────────────────────────────────────────────────────────────────
# Unified Reranker
# ──────────────────────────────────────────────────────────────────────

class NOVAReranker:
    """
    NOVA's reranker with automatic fallback.

    Tries cross-encoder first. Falls back to score-based if unavailable.

    Usage:
        reranker = NOVAReranker()
        reranked = reranker.rerank("moisturizer for dry skin", results, top_k=5)
    """

    def __init__(self, force_simple: bool = False):
        """
        Initialize the reranker.

        Args:
            force_simple: If True, always use score-based reranking.
        """
        self._inner = None
        self._is_simple = False

        if force_simple:
            self._inner = ScoreBasedReranker()
            self._is_simple = True
            return

        try:
            self._inner = CrossEncoderReranker()
            # Verify model loads
            self._inner._load_model()
            logger.info("Using CrossEncoder reranker")
        except Exception as e:
            logger.warning(f"Cross-encoder unavailable: {e}. Using score-based reranker.")
            self._inner = ScoreBasedReranker()
            self._is_simple = True

    @property
    def is_simple(self) -> bool:
        return self._is_simple

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-rank search results.

        Args:
            query: Original search query.
            results: List of search results.
            top_k: Number of results to return.

        Returns:
            Re-ranked results.
        """
        return self._inner.rerank(query, results, top_k)