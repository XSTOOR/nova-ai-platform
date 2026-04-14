# Task 3: RAG Pipeline — NOVA Product Knowledge Base

## Objective
Build a complete Retrieval-Augmented Generation (RAG) pipeline using ChromaDB for
vector storage, HuggingFace embeddings, hybrid search, cross-encoder reranking,
and RAGAS evaluation.

## Files to Create/Modify
- `rag_pipeline/embedder.py` — HuggingFace embeddings wrapper
- `rag_pipeline/vector_store.py` — ChromaDB integration (build, query, persist)
- `rag_pipeline/hybrid_search.py` — Dense + sparse search with scoring
- `rag_pipeline/reranker.py` — Cross-encoder reranking
- `rag_pipeline/ragas_eval.py` — RAGAS evaluation (4 metrics)
- `rag_pipeline/__init__.py` — Updated module exports
- `tests/test_rag_pipeline.py` — Full test suite
- `notebooks/task3_rag_pipeline.ipynb` — Colab notebook

## Acceptance Criteria

### Embedder
- [ ] Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- [ ] Embeds documents and queries with same model
- [ ] Batch embedding support for efficiency
- [ ] Fallback mock embedder for environments without model download

### Vector Store (ChromaDB)
- [ ] Builds collection from products.json + faqs.json
- [ ] Persists to disk (chroma_db/ directory)
- [ ] Supports add, query, delete operations
- [ ] Returns results with scores and metadata

### Hybrid Search
- [ ] Dense search (vector similarity via ChromaDB)
- [ ] Sparse search (keyword/TF-IDF based)
- [ ] Score fusion (Reciprocal Rank Fusion)
- [ ] Returns unified ranked results

### Reranker
- [ ] Cross-encoder scoring for initial results
- [ ] Re-sorts by relevance score
- [ ] Top-k selection
- [ ] Fallback to score-based reranking when cross-encoder unavailable

### RAGAS Evaluation
- [ ] Evaluates with 4 metrics: faithfulness, answer_relevancy, context_precision, context_recall
- [ ] Works with mock LLM for environments without API keys
- [ ] Generates evaluation report with per-metric scores

### Tests
- [ ] Unit tests for each module
- [ ] Integration test: full pipeline from query to reranked results
- [ ] Edge cases: empty queries, no results, very long queries

### Notebook
- [ ] Self-contained, Colab free tier compatible
- [ ] Demonstrates full pipeline: embed → store → search → rerank → evaluate
- [ ] Includes visualizations and metrics