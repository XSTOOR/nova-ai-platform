"""
NOVA AI Platform — Order Lookup Tool
======================================
Retrieves order details from the mock data store.

Supports lookup by:
  - order_id (exact match)
  - customer_email (returns all orders for that email)
"""

import json
from pathlib import Path
from typing import Optional

from mcp_server.audit_logger import AuditLogger

# Load mock data
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_orders() -> list[dict]:
    """Load orders from JSON data file."""
    orders_file = DATA_DIR / "orders.json"
    with open(orders_file) as f:
        return json.load(f)


def _load_products() -> list[dict]:
    """Load products from JSON data file."""
    products_file = DATA_DIR / "products.json"
    with open(products_file) as f:
        return json.load(f)


def _enrich_order(order: dict, products: list[dict]) -> dict:
    """Enrich order items with product details."""
    product_map = {p["id"]: p for p in products}
    enriched_items = []
    for item in order.get("items", []):
        enriched = dict(item)
        product = product_map.get(item.get("product_id"))
        if product:
            enriched["category"] = product.get("category")
            enriched["subcategory"] = product.get("subcategory")
            enriched["rating"] = product.get("rating")
        enriched_items.append(enriched)
    order["items"] = enriched_items
    return order


def order_lookup(
    order_id: Optional[str] = None,
    customer_email: Optional[str] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> dict:
    """
    Look up order details.

    Args:
        order_id: Specific order ID to look up.
        customer_email: Email to find all orders for a customer.
        audit_logger: Optional audit logger for tracking.

    Returns:
        Dict with success status and order data.
    """
    params = {"order_id": order_id, "customer_email": customer_email}
    tracker = audit_logger.track("order_lookup", params) if audit_logger else None

    if tracker:
        tracker.__enter__()

    try:
        if not order_id and not customer_email:
            result = {
                "success": False,
                "error": "Must provide either order_id or customer_email",
                "data": None,
            }
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        orders = _load_orders()
        products = _load_products()

        if order_id:
            # Find specific order
            matching = [o for o in orders if o["order_id"] == order_id]
            if not matching:
                result = {
                    "success": False,
                    "error": f"Order not found: {order_id}",
                    "data": None,
                }
                if tracker:
                    tracker.set_output(result)
                    tracker.set_error(result["error"])
                return result
            order = _enrich_order(matching[0], products)
            result = {
                "success": True,
                "data": order,
                "message": f"Found order {order_id}",
            }

        else:
            # Find all orders for customer email
            matching = [o for o in orders if o["customer_email"] == customer_email]
            if not matching:
                result = {
                    "success": False,
                    "error": f"No orders found for email: {customer_email}",
                    "data": None,
                }
                if tracker:
                    tracker.set_output(result)
                    tracker.set_error(result["error"])
                return result
            enriched = [_enrich_order(o, products) for o in matching]
            result = {
                "success": True,
                "data": enriched,
                "count": len(enriched),
                "message": f"Found {len(enriched)} order(s) for {customer_email}",
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