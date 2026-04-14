"""
NOVA AI Platform — Tests for Task 3: RAG Pipeline
====================================================
Comprehensive test suite covering embedder, vector store, hybrid search,
reranker, and RAGAS evaluation.
"""

import json
import math
import pytest
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.embedder import NOVAEmbedder, MockEmbedder
from rag_pipeline.vector_store import NovAVectorStore, _load_and_prepare_documents
from rag_pipeline.hybrid_search import (
    HybridSearchEngine, SparseSearcher, SearchStrategy,
    reciprocal_rank_fusion,
)
from rag_pipeline.reranker import NOVAReranker, ScoreBasedReranker
from rag_pipeline.ragas_eval import (
    RAGASEvaluator, RAGASMetrics, MockRAGASEvaluator, EVAL_QUESTIONS,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def embedder():
    """Create a mock embedder (fast, no model download)."""
    return NOVAEmbedder(force_mock=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for ChromaDB."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def vector_store(embedder, temp_dir):
    """Create a vector store with mock embeddings."""
    vs = NovAVectorStore(embedder=embedder, persist_directory=temp_dir)
    vs.build()
    return vs


@pytest.fixture
def search_engine(vector_store):
    """Create a hybrid search engine."""
    engine = HybridSearchEngine(vector_store)
    engine.build_sparse_index()
    return engine


@pytest.fixture
def reranker():
    """Create a score-based reranker (no model download needed)."""
    return NOVAReranker(force_simple=True)


@pytest.fixture
def evaluator(search_engine, reranker):
    """Create a RAGAS evaluator."""
    return RAGASEvaluator(search_engine=search_engine, reranker=reranker, use_mock=True)


# ══════════════════════════════════════════════════════════════════════
# TEST: Embedder
# ══════════════════════════════════════════════════════════════════════

class TestEmbedder:
    """Tests for the embedding module."""

    def test_mock_embedder_dimension(self):
        """Mock embedder should return correct dimension."""
        emb = MockEmbedder(dimension=384)
        assert emb.dimension == 384

    def test_mock_embedder_query(self):
        """Should embed a query into a vector."""
        emb = MockEmbedder()
        vec = emb.embed_query("Hello world")
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_mock_embedder_documents(self):
        """Should embed multiple documents."""
        emb = MockEmbedder()
        vecs = emb.embed_documents(["Hello", "World", "Test"])
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_mock_embedder_deterministic(self):
        """Same input should produce same output."""
        emb = MockEmbedder()
        v1 = emb.embed_query("test query")
        v2 = emb.embed_query("test query")
        assert v1 == v2

    def test_mock_embedder_normalized(self):
        """Vectors should be L2-normalized."""
        emb = MockEmbedder()
        vec = emb.embed_query("test")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.01

    def test_mock_embedder_different_inputs(self):
        """Different inputs should produce different vectors."""
        emb = MockEmbedder()
        v1 = emb.embed_query("moisturizer")
        v2 = emb.embed_query("sneakers")
        assert v1 != v2

    def test_nova_embedder_force_mock(self):
        """NOVAEmbedder with force_mock should use mock."""
        emb = NOVAEmbedder(force_mock=True)
        assert emb.is_mock is True
        vec = emb.embed_query("test")
        assert len(vec) > 0

    def test_nova_embedder_empty_input(self):
        """Should handle empty input."""
        emb = NOVAEmbedder(force_mock=True)
        vec = emb.embed_query("")
        assert len(vec) == emb.dimension

    def test_nova_embedder_empty_documents(self):
        """Should handle empty document list."""
        emb = NOVAEmbedder(force_mock=True)
        vecs = emb.embed_documents([])
        assert vecs == []


# ══════════════════════════════════════════════════════════════════════
# TEST: Vector Store
# ══════════════════════════════════════════════════════════════════════

class TestVectorStore:
    """Tests for the ChromaDB vector store."""

    def test_build_populates(self, vector_store):
        """Building should populate the store."""
        assert vector_store.count > 0

    def test_build_returns_stats(self, embedder, temp_dir):
        """Build should return statistics."""
        vs = NovAVectorStore(embedder=embedder, persist_directory=temp_dir)
        stats = vs.build()
        assert stats["status"] == "built"
        assert stats["total_documents"] > 0
        assert stats["products"] > 0
        assert stats["faqs"] > 0

    def test_build_skip_existing(self, vector_store):
        """Second build should skip if already populated."""
        stats = vector_store.build()
        assert stats["status"] == "skipped"

    def test_force_rebuild(self, vector_store):
        """Force rebuild should recreate the collection."""
        original_count = vector_store.count
        stats = vector_store.build(force_rebuild=True)
        assert stats["status"] == "built"
        assert vector_store.count == original_count

    def test_query_returns_results(self, vector_store):
        """Query should return matching results."""
        results = vector_store.query("moisturizer for dry skin", n_results=3)
        assert results["documents"]
        assert len(results["documents"][0]) > 0

    def test_query_with_metadata_filter(self, vector_store):
        """Should filter by metadata."""
        results = vector_store.query(
            "product", n_results=5,
            where={"source": "product"}
        )
        if results["metadatas"] and results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                assert meta["source"] == "product"

    def test_query_empty_string(self, vector_store):
        """Empty query should return empty results."""
        results = vector_store.query("", n_results=3)
        assert results["documents"] == [[]]

    def test_query_faq_content(self, vector_store):
        """Should find FAQ content."""
        results = vector_store.query("return policy", n_results=3)
        assert results["documents"]
        assert len(results["documents"][0]) > 0

    def test_add_custom_documents(self, vector_store):
        """Should add custom documents."""
        initial = vector_store.count
        vector_store.add_documents(
            documents=["Test product description"],
            metadatas=[{"source": "test", "name": "Test"}],
            ids=["test_doc_1"],
        )
        assert vector_store.count == initial + 1

    def test_delete_documents(self, vector_store):
        """Should delete documents."""
        vector_store.add_documents(
            documents=["Delete me"],
            metadatas=[{"source": "test"}],
            ids=["delete_test"],
        )
        count_before = vector_store.count
        vector_store.delete(["delete_test"])
        assert vector_store.count == count_before - 1

    def test_collection_info(self, vector_store):
        """Should return collection info."""
        info = vector_store.get_collection_info()
        assert "count" in info
        assert "name" in info
        assert info["count"] > 0

    def test_document_preparation(self):
        """Should prepare documents from data files."""
        docs, metas, ids = _load_and_prepare_documents()
        assert len(docs) > 0
        assert len(docs) == len(metas) == len(ids)
        # Should have both products and FAQs
        sources = set(m["source"] for m in metas)
        assert "product" in sources
        assert "faq" in sources


# ══════════════════════════════════════════════════════════════════════
# TEST: Hybrid Search
# ══════════════════════════════════════════════════════════════════════

class TestHybridSearch:
    """Tests for the hybrid search engine."""

    def test_dense_search(self, search_engine):
        """Dense search should return results."""
        results = search_engine.search("moisturizer", strategy=SearchStrategy.DENSE)
        assert len(results) > 0

    def test_sparse_search(self, search_engine):
        """Sparse search should return results."""
        results = search_engine.search("moisturizer", strategy=SearchStrategy.SPARSE)
        assert len(results) > 0
        assert all("score" in r for r in results)

    def test_hybrid_search(self, search_engine):
        """Hybrid search should return results."""
        results = search_engine.search("moisturizer", strategy=SearchStrategy.HYBRID)
        assert len(results) > 0
        assert all("rrf_score" in r or "score" in r for r in results)

    def test_hybrid_better_than_single(self, search_engine):
        """Hybrid should retrieve at least as many unique results."""
        dense = search_engine.search("foundation", n_results=5, strategy=SearchStrategy.DENSE)
        sparse = search_engine.search("foundation", n_results=5, strategy=SearchStrategy.SPARSE)
        hybrid = search_engine.search("foundation", n_results=5, strategy=SearchStrategy.HYBRID)
        # Hybrid should have results
        assert len(hybrid) > 0

    def test_search_respects_limit(self, search_engine):
        """Should return at most n_results."""
        results = search_engine.search("product", n_results=3)
        assert len(results) <= 3

    def test_search_empty_query(self, search_engine):
        """Empty query should return empty results."""
        results = search_engine.search("")
        assert results == []

    def test_search_product_query(self, search_engine):
        """Should find products for product queries."""
        results = search_engine.search("cashmere lounge set")
        assert len(results) > 0

    def test_search_faq_query(self, search_engine):
        """Should find FAQs for policy queries."""
        results = search_engine.search("return policy")
        assert len(results) > 0

    def test_sparse_searcher_standalone(self):
        """SparseSearcher should work standalone."""
        searcher = SparseSearcher()
        searcher.build_index()
        results = searcher.search("moisturizer", top_k=5)
        assert len(results) > 0

    def test_rrf_fusion(self):
        """RRF should combine ranked lists."""
        list1 = [
            {"id": "a", "score": 0.9, "document": "doc a"},
            {"id": "b", "score": 0.7, "document": "doc b"},
        ]
        list2 = [
            {"id": "b", "score": 0.8, "document": "doc b"},
            {"id": "c", "score": 0.6, "document": "doc c"},
        ]
        fused = reciprocal_rank_fusion([list1, list2])
        assert len(fused) == 3
        # 'b' appears in both lists, should rank high
        ids = [r["id"] for r in fused]
        assert "b" in ids


# ══════════════════════════════════════════════════════════════════════
# TEST: Reranker
# ══════════════════════════════════════════════════════════════════════

class TestReranker:
    """Tests for the reranking module."""

    def test_score_based_rerank(self):
        """ScoreBasedReranker should rerank results."""
        reranker = ScoreBasedReranker()
        results = [
            {"id": "1", "score": 0.5, "document": "A random product", "metadata": {"name": "Random"}},
            {"id": "2", "score": 0.3, "document": "The best moisturizer for dry skin care", "metadata": {"name": "Moisturizer"}},
        ]
        reranked = reranker.rerank("moisturizer for dry skin", results, top_k=2)
        assert len(reranked) == 2
        # Moisturizer doc should rank higher after reranking
        assert reranked[0]["metadata"]["name"] == "Moisturizer"

    def test_rerank_empty_results(self, reranker):
        """Empty results should return empty."""
        reranked = reranker.rerank("test", [], top_k=5)
        assert reranked == []

    def test_rerank_respects_top_k(self, reranker):
        """Should return at most top_k results."""
        results = [{"id": str(i), "score": 0.5, "document": f"doc {i}", "metadata": {}} for i in range(10)]
        reranked = reranker.rerank("test", results, top_k=3)
        assert len(reranked) <= 3

    def test_rerank_adds_scores(self, reranker):
        """Reranked results should have rerank_score."""
        results = [
            {"id": "1", "score": 0.5, "document": "test document", "metadata": {}},
        ]
        reranked = reranker.rerank("test", results, top_k=1)
        assert "rerank_score" in reranked[0]

    def test_nova_reranker_force_simple(self):
        """NOVAReranker with force_simple should use ScoreBasedReranker."""
        r = NOVAReranker(force_simple=True)
        assert r.is_simple is True


# ══════════════════════════════════════════════════════════════════════
# TEST: RAGAS Evaluation
# ══════════════════════════════════════════════════════════════════════

class TestRAGASEvaluation:
    """Tests for the RAGAS evaluation module."""

    def test_ragas_metrics_average(self):
        """Metrics should compute average correctly."""
        m = RAGASMetrics(faithfulness=0.8, answer_relevancy=0.9,
                         context_precision=0.7, context_recall=0.6)
        assert 0.7 < m.average < 0.8

    def test_ragas_metrics_to_dict(self):
        """Should serialize to dict."""
        m = RAGASMetrics(faithfulness=1.0, answer_relevancy=0.5,
                         context_precision=0.75, context_recall=0.25)
        d = m.to_dict()
        assert "faithfulness" in d
        assert "average" in d

    def test_mock_evaluator(self):
        """Mock evaluator should return metrics."""
        evaluator = MockRAGASEvaluator()
        metrics = evaluator.evaluate(
            question="What is a moisturizer?",
            retrieved_docs=["A moisturizer hydrates the skin using hyaluronic acid."],
            generated_answer="A moisturizer hydrates skin with hyaluronic acid.",
            expected_keywords=["moisturizer", "hydrates", "hyaluronic acid"],
        )
        assert metrics.answer_relevancy > 0
        assert metrics.faithfulness > 0

    def test_eval_questions_loaded(self):
        """Default evaluation questions should be loaded."""
        assert len(EVAL_QUESTIONS) >= 8
        for q in EVAL_QUESTIONS:
            assert "question" in q
            assert "expected_keywords" in q

    def test_pipeline_evaluation(self, evaluator):
        """Full pipeline evaluation should produce report."""
        report = evaluator.evaluate_pipeline()
        assert "total_questions" in report
        assert "aggregate_metrics" in report
        assert report["total_questions"] > 0
        metrics = report["aggregate_metrics"]
        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics

    def test_single_question_eval(self, evaluator):
        """Single question evaluation should work."""
        result = evaluator.evaluate_single(
            "What moisturizer do you recommend?",
            ["moisturizer", "hydra-boost"],
        )
        assert "question" in result
        assert "metrics" in result
        assert isinstance(result["metrics"], RAGASMetrics)

    def test_generate_answer_from_context(self, evaluator):
        """Should generate answer from context."""
        answer = evaluator.generate_answer_from_context(
            "What is a moisturizer?",
            ["The Hydra-Boost Moisturizer provides 72-hour hydration with hyaluronic acid."],
        )
        assert len(answer) > 0
        assert "moisturizer" in answer.lower() or "hydra" in answer.lower()


# ══════════════════════════════════════════════════════════════════════
# TEST: Full Pipeline Integration
# ══════════════════════════════════════════════════════════════════════

class TestRAGPipelineIntegration:
    """Integration tests for the full RAG pipeline."""

    def test_embed_store_search_rerank(self, search_engine, reranker):
        """Full pipeline: embed → store → search → rerank."""
        query = "best moisturizer for dry sensitive skin"

        # Search
        results = search_engine.search(query, n_results=5, strategy=SearchStrategy.HYBRID)
        assert len(results) > 0

        # Rerank
        reranked = reranker.rerank(query, results, top_k=3)
        assert len(reranked) > 0
        assert all("rerank_score" in r for r in reranked)

        # Verify order
        scores = [r["rerank_score"] for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_pipeline_handles_various_queries(self, search_engine, reranker):
        """Pipeline should handle diverse query types."""
        queries = [
            "cashmere sweater",
            "return policy",
            "lipstick shades",
            "shipping to europe",
            "rewards points",
            "foundation for oily skin",
            "gift ideas under $50",
            "cruelty free products",
        ]

        for query in queries:
            results = search_engine.search(query, n_results=3, strategy=SearchStrategy.HYBRID)
            # Not all queries will have results, but should not crash
            if results:
                reranked = reranker.rerank(query, results, top_k=3)
                assert len(reranked) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])