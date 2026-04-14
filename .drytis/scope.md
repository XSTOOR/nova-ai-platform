# NOVA AI Platform - Scope

## In Scope
### Task 1: Prompt Engineering
- COSTAR framework prompt templates for all 5 intent categories
- Intent classifier with confidence scores
- Escalation logic (threshold-based + keyword-based)
- Prompt injection defense (regex + LLM scoring)
- Unit tests for all components

### Task 2: MCP Server
- 5 backend tools: order_lookup, product_search, return_initiate, loyalty_check, escalation_tool
- Audit logging for all tool invocations
- MCP protocol server using FastMCP
- Tool input validation with Pydantic schemas
- Integration tests

### Task 3: RAG Pipeline
- Document embedding with HuggingFace models
- ChromaDB vector store with persistence
- Hybrid search (dense + sparse)
- Cross-encoder reranking
- RAGAS evaluation with 4 metrics
- Performance benchmarks

### Task 4: Fine-Tuning
- Brand voice dataset preparation (12 samples → augmented to 100+)
- QLoRA configuration (4-bit, LoRA rank 16)
- Training script with W&B tracking
- Inference pipeline for brand voice responses
- BLEU/ROUGE evaluation

### Task 5: Multi-Agent Platform
- LangGraph state machine with 4 agents
- Triage → Route → Execute → Respond flow
- Human-in-the-loop escalation
- Full audit trail logging
- End-to-end integration tests

## Out of Scope
- Web UI / Streamlit dashboard (notebook-only interaction)
- Production deployment (Docker, K8s, cloud)
- Real payment processing
- Real database connections (all mock data)
- User authentication
- Multi-language support
- Voice/chat real-time integration