"""
NOVA AI Platform — Escalation Agent
======================================
Handles human-in-the-loop escalation for sensitive or complex cases.

Uses:
  - Task 1: EscalationLogic (threshold + keyword triggers)
  - Task 2: EscalationTool (ticket creation with SLA)
  - Full audit trail for compliance
"""

import logging
import time
from typing import Optional

from multi_agent.state import NOVAState
from prompts.escalation_logic import EscalationManager
from mcp_server.tools.escalation_tool import escalation_tool

logger = logging.getLogger(__name__)


class EscalationAgent:
    """
    Escalation agent — determines if human intervention is needed.

    Triggers:
      1. Low confidence (<0.5) from intent classifier
      2. Sensitive topics (medical, legal, billing disputes)
      3. Explicit human request ("speak to manager")
      4. Compound/multi-issue queries
      5. Negative sentiment detection
    """

    def __init__(self, escalation_manager: Optional[EscalationManager] = None):
        if escalation_manager is None:
            self._manager = EscalationManager(
                confidence_threshold_high=0.5,  # Only escalate if confidence < 0.5
            )
        else:
            self._manager = escalation_manager

    def check_escalation(self, state: NOVAState) -> NOVAState:
        """
        Evaluate whether escalation is needed.
        Called after triage AND after support/recommendation agents.
        """
        start = time.perf_counter()
        audit_entry = {
            "agent": "escalation",
            "step": "check",
            "timestamp": time.time(),
        }

        message = state.get("current_message", "")
        confidence = state.get("confidence", 1.0)
        intent = state.get("intent", "")
        tool_results = state.get("tool_results", {})

        # Evaluate escalation triggers
        decision = self._manager.evaluate(
            message=message,
            intent=intent,
            confidence=confidence,
            previous_turns=len(state.get("messages", [])),
            tool_results=tool_results,
        )

        state["escalation_needed"] = decision.should_escalate
        state["escalation_reason"] = decision.reason
        state["escalation_priority"] = decision.priority.value

        audit_entry.update({
            "action": "check_escalation",
            "should_escalate": decision.should_escalate,
            "reason": decision.reason,
            "priority": decision.priority.value,
            "triggers_matched": decision.triggered_by,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        })
        state["audit_trail"] = state.get("audit_trail", []) + [audit_entry]

        return state

    def create_ticket(self, state: NOVAState) -> NOVAState:
        """
        Create an escalation ticket for human follow-up.
        Called when escalation_needed is True.
        """
        start = time.perf_counter()
        audit_entry = {
            "agent": "escalation",
            "step": "create_ticket",
            "timestamp": time.time(),
        }

        reason = state.get("escalation_reason", "General escalation")
        priority = state.get("escalation_priority", "P3")
        customer_id = state.get("customer_id", "guest")
        customer_name = state.get("customer_name", "Friend")
        message = state.get("current_message", "")
        intent = state.get("intent", "")
        confidence = state.get("confidence", 0.0)

        # Priority already in correct format (low/medium/high/urgent)
        tool_priority = priority

        ticket_result = escalation_tool(
            reason=reason,
            priority=tool_priority,
            customer_id=customer_id,
            customer_name=customer_name,
            customer_email=customer_id if "@" in customer_id else None,
            intent=intent,
            confidence=confidence,
            conversation_summary=f"Session {state.get('session_id', 'N/A')}: {message[:200]}",
        )

        ticket_data = ticket_result.get("data", {})
        state["escalation_ticket_id"] = ticket_data.get("ticket_id", "N/A")

        audit_entry.update({
            "action": "ticket_created",
            "ticket_id": ticket_data.get("ticket_id", ""),
            "priority": tool_priority,
            "success": ticket_result.get("success", False),
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        })
        state["audit_trail"] = state.get("audit_trail", []) + [audit_entry]

        return state

    def generate_escalation_response(self, state: NOVAState) -> NOVAState:
        """Generate a warm NOVA-branded escalation response."""
        name = state.get("customer_name", "there")
        ticket_id = state.get("escalation_ticket_id", "")
        priority = state.get("escalation_priority", "P3")

        sla_map = {
            "urgent": "within 15 minutes",
            "high": "within 1 hour",
            "medium": "within 4 hours",
            "low": "within 24 hours",
        }
        sla = sla_map.get(priority, "within 24 hours")

        state["response"] = (
            f"Hey {name}! 💕 I want to make sure you get the best possible help, "
            f"so I'm connecting you with one of our senior team members. "
            f"Your ticket ID is **{ticket_id}** and they'll reach out {sla}. "
            f"Thank you for your patience — we've got you! ✨"
        )

        return state