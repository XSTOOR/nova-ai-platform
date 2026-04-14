# NOVA AI Platform - Spec Overview

## Tech Stack
- **Language**: Python 3.10+
- **LLM**: OpenRouter (google/gemini-2.0-flash-001) or Groq (llama-3.1-8b-instant)
- **Framework**: LangChain + LangGraph for multi-agent orchestration
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **Vector Store**: ChromaDB with persistence
- **Reranker**: Cross-encoder ms-marco-MiniLM-L-6-v2
- **Fine-Tuning**: QLoRA on Llama-3.2-1B-Instruct
- **Experiment Tracking**: Weights & Biases
- **RAG Eval**: RAGAS
- **MCP**: FastMCP library
- **Testing**: pytest + pytest-asyncio

## Key Decisions
1. All LLM calls go through a single abstraction layer (config/settings.py) for provider switching
2. ChromaDB used over Pinecone/Weaviate for zero-cost local operation
3. Smaller embedding model (384-dim) chosen for Colab free-tier RAM constraints
4. Llama-3.2-1B chosen as base for fine-tuning to fit in Colab T4 GPU (16GB VRAM with 4-bit quantization)
5. All mock data embedded in data/ JSON files — no external database dependency