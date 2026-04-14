"""
NOVA AI Platform — Return Initiate Tool
==========================================
Processes return/exchange requests with eligibility validation.

Return Policy:
  - 30-day window from delivery date
  - Items must be unused, in original packaging
  - Beauty products must be unopened for full refund
  - Sale items: exchange or store credit only
  - Allergic reactions: full refund even if opened (no questions)
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp_server.audit_logger import AuditLogger

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_orders() -> list[dict]:
    with open(DATA_DIR / "orders.json") as f:
        return json.load(f)


def _save_orders(orders: list[dict]) -> None:
    with open(DATA_DIR / "orders.json", "w") as f:
        json.dump(orders, f, indent=2)


def return_initiate(
    order_id: str,
    reason: str,
    item_index: Optional[int] = None,
    return_type: str = "refund",
    customer_email: Optional[str] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> dict:
    """
    Initiate a return or exchange for an order.

    Args:
        order_id: The order ID to return items from.
        reason: Reason for the return.
        item_index: Specific item index to return (None = all items).
        return_type: "refund", "exchange", or "store_credit".
        customer_email: Customer email for verification.
        audit_logger: Optional audit logger.

    Returns:
        Dict with success status and return ticket details.
    """
    params = {
        "order_id": order_id,
        "reason": reason,
        "item_index": item_index,
        "return_type": return_type,
        "customer_email": customer_email,
    }
    tracker = audit_logger.track("return_initiate", params) if audit_logger else None

    if tracker:
        tracker.__enter__()

    try:
        # Validate inputs
        if not order_id:
            result = {"success": False, "error": "order_id is required", "data": None}
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        if return_type not in ("refund", "exchange", "store_credit"):
            result = {
                "success": False,
                "error": f"Invalid return_type '{return_type}'. Must be: refund, exchange, store_credit",
                "data": None,
            }
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        # Load and find order
        orders = _load_orders()
        order = None
        for o in orders:
            if o["order_id"] == order_id:
                order = o
                break

        if not order:
            result = {"success": False, "error": f"Order not found: {order_id}", "data": None}
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        # Verify customer email if provided
        if customer_email and order.get("customer_email") != customer_email:
            result = {
                "success": False,
                "error": "Email does not match order records",
                "data": None,
            }
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        # Check if allergic reaction (bypasses all rules)
        is_allergic = any(
            kw in reason.lower()
            for kw in ["allergic", "allergy", "reaction", "irritation", "rash"]
        )

        if is_allergic:
            ticket_id = f"RET-{uuid.uuid4().hex[:8].upper()}"
            result = {
                "success": True,
                "data": {
                    "ticket_id": ticket_id,
                    "order_id": order_id,
                    "return_type": "refund",
                    "reason": reason,
                    "status": "approved",
                    "refund_amount": order["total"],
                    "note": "Allergic reaction detected — immediate full refund approved. No return required.",
                    "estimated_refund_days": 3,
                },
                "message": f"Return ticket {ticket_id} created. Full refund of ${order['total']:.2f} approved.",
            }
            if tracker:
                tracker.set_output(result)
            return result

        # Check return eligibility
        if not order.get("return_eligible", False):
            deadline = order.get("return_deadline", "N/A")
            result = {
                "success": False,
                "error": f"Order is not eligible for return. Return deadline was {deadline}",
                "data": {"order_id": order_id, "return_eligible": False, "deadline": deadline},
            }
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        # Check order status
        if order.get("status") in ("cancelled", "processing"):
            result = {
                "success": False,
                "error": f"Cannot return order with status '{order.get('status')}'",
                "data": None,
            }
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        # Calculate refund amount
        items = order.get("items", [])
        if item_index is not None and 0 <= item_index < len(items):
            refund_amount = items[item_index]["price"] * items[item_index].get("qty", 1)
            return_items = [items[item_index]]
        else:
            refund_amount = order["total"]
            return_items = items

        # Create return ticket
        ticket_id = f"RET-{uuid.uuid4().hex[:8].upper()}"

        result = {
            "success": True,
            "data": {
                "ticket_id": ticket_id,
                "order_id": order_id,
                "return_type": return_type,
                "reason": reason,
                "status": "pending_approval",
                "items": return_items,
                "refund_amount": refund_amount,
                "return_shipping": "free",
                "estimated_refund_days": 5,
                "instructions": (
                    "Pack items in original packaging. "
                    "A prepaid shipping label will be emailed within 24 hours. "
                    "Refund will be processed within 5 business days of receiving the return."
                ),
            },
            "message": f"Return ticket {ticket_id} created for ${refund_amount:.2f}. Pending approval.",
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