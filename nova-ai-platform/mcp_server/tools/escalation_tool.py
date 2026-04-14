"""
NOVA AI Platform — Escalation Tool
=====================================
Creates escalation tickets to route customers to human agents.

Escalation priorities:
  - LOW:      Non-urgent, general handoff
  - MEDIUM:   Customer requested, moderate issue
  - HIGH:     Sensitive topic, medical, billing
  - URGENT:   Legal threats, severe complaints, data issues
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from mcp_server.audit_logger import AuditLogger

# Mock escalation queue
_escalation_queue: list[dict] = []


def escalation_tool(
    reason: str,
    priority: str = "medium",
    customer_id: Optional[str] = None,
    customer_name: Optional[str] = None,
    customer_email: Optional[str] = None,
    order_id: Optional[str] = None,
    intent: Optional[str] = None,
    conversation_summary: Optional[str] = None,
    confidence: Optional[float] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> dict:
    """
    Create an escalation ticket for human agent handoff.

    Args:
        reason: Why the escalation is needed.
        priority: "low", "medium", "high", or "urgent".
        customer_id: Customer ID.
        customer_name: Customer name.
        customer_email: Customer email.
        order_id: Related order ID.
        intent: The classified intent that triggered escalation.
        conversation_summary: Summary of the conversation so far.
        confidence: The intent confidence score.
        audit_logger: Optional audit logger.

    Returns:
        Dict with success status and escalation ticket details.
    """
    params = {
        "reason": reason,
        "priority": priority,
        "customer_id": customer_id,
        "order_id": order_id,
        "intent": intent,
        "confidence": confidence,
    }
    tracker = audit_logger.track("escalation_tool", params) if audit_logger else None

    if tracker:
        tracker.__enter__()

    try:
        # Validate priority
        valid_priorities = ["low", "medium", "high", "urgent"]
        if priority not in valid_priorities:
            result = {
                "success": False,
                "error": f"Invalid priority '{priority}'. Must be: {valid_priorities}",
                "data": None,
            }
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        # Create escalation ticket
        ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc)

        ticket = {
            "ticket_id": ticket_id,
            "status": "open",
            "priority": priority,
            "reason": reason,
            "customer": {
                "id": customer_id,
                "name": customer_name,
                "email": customer_email,
            },
            "order_id": order_id,
            "intent": intent,
            "confidence": confidence,
            "conversation_summary": conversation_summary,
            "created_at": now.isoformat(),
            "assigned_to": None,  # Would be assigned by routing system
            "sla_minutes": {
                "urgent": 15,
                "high": 60,
                "medium": 240,
                "low": 480,
            }.get(priority, 240),
        }

        # Add to queue
        _escalation_queue.append(ticket)

        result = {
            "success": True,
            "data": ticket,
            "message": (
                f"Escalation ticket {ticket_id} created with {priority} priority. "
                f"SLA: response within {ticket['sla_minutes']} minutes."
            ),
        }

        if tracker:
            tracker.set_output(result)
        return result

    except Exception as e:
        result = {"success": False, "error": str(e), "data": None}
        if tracker:
            tracker.set_output(result)
            tracker.set_error(str(e))
        return result

    finally:
        if tracker:
            tracker.__exit__(None, None, None)


def get_escalation_queue() -> list[dict]:
    """Get the current escalation queue (for testing/monitoring)."""
    return _escalation_queue.copy()


def clear_escalation_queue() -> None:
    """Clear the escalation queue (for testing)."""
    _escalation_queue.clear()