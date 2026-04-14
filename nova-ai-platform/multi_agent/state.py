"""
NOVA AI Platform — Shared State Schema
=========================================
TypedDict defining the state that flows through the LangGraph agent graph.

Every agent reads from and writes to this shared state.
"""

from typing import TypedDict, Optional


class NOVAState(TypedDict, total=False):
    """
    Shared state for the NOVA multi-agent support graph.

    Flows through: Triage → Support/Recommendation → Escalation Check → Respond
    """

    # ── Input ──────────────────────────────────────────────────────
    messages: list[dict]              # Conversation history [{"role": ..., "content": ...}]
    customer_id: str                  # Customer identifier (email or ID)
    customer_name: str                # Customer name for personalization
    current_message: str              # The latest customer message

    # ── Triage Agent Output ────────────────────────────────────────
    intent: str                       # Classified intent (order_status, product_inquiry, etc.)
    confidence: float                 # Intent classification confidence (0.0-1.0)
    injection_blocked: bool           # Whether an injection attack was detected
    injection_threat_level: str       # SAFE, SUSPICIOUS, or BLOCKED
    injection_score: float            # Injection threat score

    # ── Routing Decision ───────────────────────────────────────────
    route: str                        # "support", "recommendation", "block", "escalation"

    # ── Support Agent Output ───────────────────────────────────────
    tools_used: list[str]             # Audit trail of tools invoked
    tool_results: dict                # Raw tool outputs keyed by tool name

    # ── Recommendation Agent Output ────────────────────────────────
    rag_context: list[dict]           # Retrieved documents from RAG pipeline
    rag_scores: list[float]           # Relevance scores for retrieved docs

    # ── Escalation Agent Output ────────────────────────────────────
    escalation_needed: bool           # Whether human escalation is required
    escalation_reason: str            # Why the conversation was escalated
    escalation_priority: str          # P1 (critical) through P4 (low)
    escalation_ticket_id: str         # Ticket ID if escalation created

    # ── Response ───────────────────────────────────────────────────
    response: str                     # Final NOVA-branded response to customer
    brand_voice_score: float          # Brand voice consistency score (0.0-1.0)

    # ── Audit Trail ────────────────────────────────────────────────
    audit_trail: list[dict]           # Full decision log for every agent step
    session_id: str                   # Unique session identifier
    error: str                        # Error message if something failed


def create_initial_state(
    message: str,
    customer_id: str = "guest",
    customer_name: str = "Friend",
    session_id: str = "",
) -> NOVAState:
    """Create a fresh initial state for a new conversation."""
    import uuid

    return NOVAState(
        messages=[{"role": "user", "content": message}],
        customer_id=customer_id,
        customer_name=customer_name,
        current_message=message,
        intent="",
        confidence=0.0,
        injection_blocked=False,
        injection_threat_level="SAFE",
        injection_score=0.0,
        route="",
        tools_used=[],
        tool_results={},
        rag_context=[],
        rag_scores=[],
        escalation_needed=False,
        escalation_reason="",
        escalation_priority="",
        escalation_ticket_id="",
        response="",
        brand_voice_score=0.0,
        audit_trail=[],
        session_id=session_id or str(uuid.uuid4())[:8],
        error="",
    )