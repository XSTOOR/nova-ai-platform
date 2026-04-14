"""
NOVA AI Platform — Hybrid Search
===================================
Combines dense vector search with sparse keyword search using
Reciprocal Rank Fusion (RRF) for unified ranking.

Search strategies:
  1. DENSE: Pure vector similarity search via ChromaDB
  2. SPARSE: TF-IDF-like keyword matching with term frequency scoring
  3. HYBRID: RRF fusion of dense + sparse results (best of both)
"""

import json
import logging
import math
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Optional

from config.settings import Config
from rag_pipeline.vector_store import NovAVectorStore

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


class SearchStrategy(str, Enum):
    """Search strategy options."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


# ──────────────────────────────────────────────────────────────────────
# Sparse Search (Keyword-based)
# ──────────────────────────────────────────────────────────────────────

class SparseSearcher:
    """
    Simple keyword-based sparse search using TF-IDF-like scoring.

    Builds an inverted index from document content for fast keyword lookups.
    """

    def __init__(self):
        self._documents: list[dict] = []
        self._inverted_index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._doc_lengths: list[int] = []
        self._idf: dict[str, float] = {}

    def build_index(self, data_dir: Optional[str] = None):
        """
        Build the sparse index from data files.

        Args:
            data_dir: Data directory. Defaults to project data/.
        """
        data_path = Path(data_dir) if data_dir else DATA_DIR
        self._documents = []
        self._inverted_index = defaultdict(list)
        self._doc_lengths = []

        # Load products
        products_file = data_path / "products.json"
        if products_file.exists():
            with open(products_file) as f:
                products = json.load(f)
            for p in products:
                doc_text = f"{p['name']} {p.get('description','')} {p.get('category','')} "
                doc_text += f"{' '.join(p.get('tags', []))} {p.get('ingredients','')} "
                doc_text += f"{p.get('materials','')} {p.get('skin_type','')} "
                if p.get("skin_concerns"):
                    doc_text += " ".join(p["skin_concerns"])
                self._documents.append({
                    "id": f"product_{p['id']}",
                    "text": doc_text.lower(),
                    "metadata": {
                        "source": "product",
                        "name": p["name"],
                        "category": p["category"],
                        "price": p["price"],
                        "product_id": p["id"],
                    },
                    "document": (f"Product: {p['name']}\n{p.get('description','')}"),
                })

        # Load FAQs
        faqs_file = data_path / "faqs.json"
        if faqs_file.exists():
            with open(faqs_file) as f:
                faqs = json.load(f)
            for faq in faqs:
                doc_text = f"{faq['question']} {faq['answer']} {' '.join(faq.get('keywords',[]))} {faq['category']}"
                self._documents.append({
                    "id": f"faq_{faq['id']}",
                    "text": doc_text.lower(),
                    "metadata": {
                        "source": "faq",
                        "name": faq["question"][:100],
                        "faq_category": faq["category"],
                    },
                    "document": f"FAQ: {faq['question']}\nAnswer: {faq['answer']}",
                })

        # Tokenize and build inverted index
        for doc_idx, doc in enumerate(self._documents):
            tokens = re.findall(r'\b\w+\b', doc["text"])
            self._doc_lengths.append(len(tokens))

            # Term frequency
            tf_counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_counts[token] += 1

            for token, count in tf_counts.items():
                tf = count / len(tokens) if tokens else 0
                self._inverted_index[token].append((doc_idx, tf))

        # Compute IDF
        n_docs = len(self._documents)
        for token, postings in self._inverted_index.items():
            df = len(postings)
            self._idf[token] = math.log((n_docs + 1) / (df + 1)) + 1

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search using TF-IDF scoring.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of result dicts with id, score, metadata, document.
        """
        if not self._documents:
            return []

        query_tokens = re.findall(r'\b\w+\b', query.lower())
        scores: dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token in self._inverted_index:
                idf = self._idf.get(token, 1.0)
                for doc_idx, tf in self._inverted_index[token]:
                    scores[doc_idx] += tf * idf

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in ranked[:top_k]:
            doc = self._documents[doc_idx]
            results.append({
                "id": doc["id"],
                "score": round(score, 4),
                "metadata": doc["metadata"],
                "document": doc["document"],
            })

        return results


# ──────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        result_lists: Lists of result dicts, each with 'id' and 'score'.
        k: RRF constant (default 60).

    Returns:
        Single fused and sorted list.
    """
    fused_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            doc_id = result["id"]
            fused_scores[doc_id] += 1.0 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = result

    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in sorted_ids:
        result = dict(doc_map[doc_id])
        result["rrf_score"] = round(score, 4)
        results.append(result)

    return results


# ──────────────────────────────────────────────────────────────────────
# Hybrid Search Engine
# ──────────────────────────────────────────────────────────────────────

class HybridSearchEngine:
    """
    NOVA's hybrid search engine combining dense and sparse retrieval.

    Usage:
        engine = HybridSearchEngine(vector_store=vs)
        engine.build_sparse_index()
        results = engine.search("moisturizer for dry skin", strategy="hybrid")
    """

    def __init__(self, vector_store: NovAVectorStore):
        self._vector_store = vector_store
        self._sparse_searcher = SparseSearcher()
        self._sparse_built = False

    def build_sparse_index(self, data_dir: Optional[str] = None) -> None:
        """Build the sparse (keyword) search index."""
        self._sparse_searcher.build_index(data_dir)
        self._sparse_built = True
        logger.info(f"Sparse index built: {len(self._sparse_searcher._documents)} documents")

    def search(
        self,
        query: str,
        n_results: int = 5,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search the knowledge base.

        Args:
            query: Search query.
            n_results: Number of results to return.
            strategy: dense, sparse, or hybrid.
            where: Optional metadata filter for dense search.

        Returns:
            List of result dicts sorted by relevance.
        """
        if not query or not query.strip():
            return []

        results = []

        if strategy == SearchStrategy.DENSE or strategy == SearchStrategy.HYBRID:
            dense_results = self._dense_search(query, n_results * 2, where)

        if strategy == SearchStrategy.SPARSE or strategy == SearchStrategy.HYBRID:
            sparse_results = self._sparse_search(query, n_results * 2)

        if strategy == SearchStrategy.DENSE:
            results = dense_results

        elif strategy == SearchStrategy.SPARSE:
            results = sparse_results

        elif strategy == SearchStrategy.HYBRID:
            results = reciprocal_rank_fusion([dense_results, sparse_results])

        return results[:n_results]

    def _dense_search(self, query: str, n_results: int, where: Optional[dict]) -> list[dict]:
        """Perform dense vector search."""
        raw = self._vector_store.query(query, n_results=n_results, where=where)

        results = []
        if raw and raw.get("documents") and raw["documents"][0]:
            for i in range(len(raw["documents"][0])):
                results.append({
                    "id": raw["ids"][0][i] if raw["ids"] else f"dense_{i}",
                    "score": 1.0 - raw["distances"][0][i] if raw.get("distances") else 0.5,
                    "metadata": raw["metadatas"][0][i] if raw.get("metadatas") else {},
                    "document": raw["documents"][0][i],
                })
        return results

    def _sparse_search(self, query: str, n_results: int) -> list[dict]:
        """Perform sparse keyword search."""
        if not self._sparse_built:
            self.build_sparse_index()
        return self._sparse_searcher.search(query, top_k=n_results)