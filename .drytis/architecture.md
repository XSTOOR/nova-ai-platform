# NOVA AI Platform - Architecture

## Directory Structure
```
/workspace/nova-ai-platform/
├── prompts/          # Task 1: All prompt engineering logic
├── mcp_server/       # Task 2: MCP protocol server + tools
├── rag_pipeline/     # Task 3: RAG retrieval pipeline
├── fine_tuning/      # Task 4: QLoRA fine-tuning
├── multi_agent/      # Task 5: LangGraph orchestration
├── notebooks/        # Colab notebooks (one per task)
├── data/             # Mock data (JSON files)
├── tests/            # pytest test suite
└── config/           # Central configuration
```

## Data Flow
```
User Input
    │
    ▼
Injection Defense (prompts/injection_defense.py)
    │ safe? ── No ──▶ Reject + Log
    │
    ▼ Yes
Triage Agent (multi_agent/agents/triage_agent.py)
    │ Uses: IntentClassifier (prompts/intent_classifier.py)
    │
    ▼
Router Decision (multi_agent/graph.py)
    │
    ├─── order_status ──▶ Support Agent + order_lookup tool
    ├─── product_inquiry ──▶ Recommendation Agent + RAG Pipeline
    ├─── return_refund ──▶ Support Agent + return_initiate tool
    ├─── loyalty_rewards ──▶ Support Agent + loyalty_check tool
    ├─── general_support ──▶ Support Agent + FAQ lookup
    └─── escalation ──▶ Escalation Agent + human_loop
    │
    ▼
Response Generation (with brand voice)
    │ Uses: COSTAR templates (prompts/costar_templates.py)
    │
    ▼
Audit Trail (multi_agent/audit_trail.py)
    │
    ▼
Response to User
```

## Module Dependencies
```
config/settings.py ←── All modules
prompts/ ←── multi_agent/agents/
mcp_server/tools/ ←── multi_agent/agents/
rag_pipeline/ ←── multi_agent/agents/recommendation_agent.py
fine_tuning/ ←── multi_agent/ (for brand voice)
mcp_server/audit_logger.py ←── mcp_server/tools/ + multi_agent/audit_trail.py
```