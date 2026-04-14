"""
NOVA AI Platform — Loyalty Check Tool
=======================================
Queries NOVA Rewards program data: points balance, tier status, history.

NOVA Rewards Program:
  - 1 point per $1 spent
  - Bronze (0-499 pts/yr): Standard earning, birthday reward
  - Silver (500-1499 pts/yr): 1.5x earning, free express shipping
  - Gold (1500+ pts/yr): 2x earning, free overnight, personal styling
  - Redemptions: 100pts=$5, 250pts=$15, 500pts=$35
"""

import json
from pathlib import Path
from typing import Optional

from mcp_server.audit_logger import AuditLogger

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_orders() -> list[dict]:
    with open(DATA_DIR / "orders.json") as f:
        return json.load(f)


# Mock loyalty database (in a real system, this would be a database)
_LOYALTY_DB = {
    "CUST-5001": {
        "customer_id": "CUST-5001",
        "customer_name": "Sarah Mitchell",
        "email": "sarah.mitchell@email.com",
        "points": 1245,
        "lifetime_points": 3820,
        "tier": "Silver",
        "year_to_date_points": 1245,
        "points_to_next_tier": 255,  # to Gold (1500)
        "earning_rate": 1.5,
        "member_since": "2023-03-15",
        "perks": ["1.5x point earning", "Free express shipping", "Early access to new drops"],
        "recent_transactions": [
            {"date": "2024-12-15", "type": "earn", "points": 98, "order": "ORD-2024-10001"},
            {"date": "2024-11-20", "type": "earn", "points": 104, "order": "ORD-2024-10004"},
            {"date": "2024-10-05", "type": "redeem", "points": -250, "reward": "$15 off"},
            {"date": "2024-09-12", "type": "earn", "points": 189, "order": "ORD-2024-10007"},
        ],
    },
    "CUST-5002": {
        "customer_id": "CUST-5002",
        "customer_name": "James Rodriguez",
        "email": "james.r@email.com",
        "points": 129,
        "lifetime_points": 129,
        "tier": "Bronze",
        "year_to_date_points": 129,
        "points_to_next_tier": 371,  # to Silver (500)
        "earning_rate": 1.0,
        "member_since": "2024-12-28",
        "perks": ["Standard earning rate", "Birthday reward", "Sale access"],
        "recent_transactions": [
            {"date": "2024-12-28", "type": "earn", "points": 129, "order": "ORD-2024-10002"},
        ],
    },
    "CUST-5003": {
        "customer_id": "CUST-5003",
        "customer_name": "Emily Chen",
        "email": "emily.chen@email.com",
        "points": 0,
        "lifetime_points": 354,
        "tier": "Bronze",
        "year_to_date_points": 354,
        "points_to_next_tier": 146,  # to Silver (500)
        "earning_rate": 1.0,
        "member_since": "2024-12-30",
        "perks": ["Standard earning rate", "Birthday reward", "Sale access"],
        "recent_transactions": [
            {"date": "2024-12-30", "type": "earn", "points": 354, "order": "ORD-2024-10003"},
        ],
    },
    "CUST-5004": {
        "customer_id": "CUST-5004",
        "customer_name": "Aisha Johnson",
        "email": "aisha.j@email.com",
        "points": 2315,
        "lifetime_points": 7890,
        "tier": "Gold",
        "year_to_date_points": 2315,
        "points_to_next_tier": 0,
        "earning_rate": 2.0,
        "member_since": "2022-06-10",
        "perks": [
            "2x point earning", "Free overnight shipping",
            "Personal styling session", "Exclusive events", "Birthday surprise",
        ],
        "recent_transactions": [
            {"date": "2024-12-10", "type": "earn", "points": 538, "order": "ORD-2024-10005"},
            {"date": "2024-11-25", "type": "redeem", "points": -500, "reward": "$35 off"},
            {"date": "2024-10-18", "type": "earn", "points": 445, "order": "ORD-2024-10008"},
            {"date": "2024-09-05", "type": "earn", "points": 320, "order": "ORD-2024-10009"},
        ],
    },
    "CUST-5005": {
        "customer_id": "CUST-5005",
        "customer_name": "Marcus Lee",
        "email": "marcus.lee@email.com",
        "points": 0,
        "lifetime_points": 0,
        "tier": "Bronze",
        "year_to_date_points": 0,
        "points_to_next_tier": 500,
        "earning_rate": 1.0,
        "member_since": "2024-12-18",
        "perks": ["Standard earning rate", "Birthday reward", "Sale access"],
        "recent_transactions": [],
    },
}

# Redemption tiers
REDEMPTION_TIERS = {
    100: 5,    # 100 points = $5 off
    250: 15,   # 250 points = $15 off
    500: 35,   # 500 points = $35 off
}


def loyalty_check(
    customer_id: Optional[str] = None,
    customer_email: Optional[str] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> dict:
    """
    Check loyalty/rewards information for a customer.

    Args:
        customer_id: Customer ID to look up.
        customer_email: Email to find customer.
        audit_logger: Optional audit logger.

    Returns:
        Dict with success status and loyalty data.
    """
    params = {"customer_id": customer_id, "customer_email": customer_email}
    tracker = audit_logger.track("loyalty_check", params) if audit_logger else None

    if tracker:
        tracker.__enter__()

    try:
        if not customer_id and not customer_email:
            result = {
                "success": False,
                "error": "Must provide either customer_id or customer_email",
                "data": None,
            }
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        # Find customer
        customer_data = None

        if customer_id and customer_id in _LOYALTY_DB:
            customer_data = _LOYALTY_DB[customer_id]
        elif customer_email:
            for cid, data in _LOYALTY_DB.items():
                if data["email"] == customer_email:
                    customer_data = data
                    break

        if not customer_data:
            result = {
                "success": False,
                "error": f"Customer not found: {customer_id or customer_email}",
                "data": None,
            }
            if tracker:
                tracker.set_output(result)
                tracker.set_error(result["error"])
            return result

        # Calculate redemption options
        available_points = customer_data["points"]
        redemption_options = []
        for pts, dollars in sorted(REDEMPTION_TIERS.items()):
            if available_points >= pts:
                redemption_options.append({
                    "points": pts,
                    "value": f"${dollars} off",
                    "remaining_after": available_points - pts,
                })

        result = {
            "success": True,
            "data": {
                "customer_id": customer_data["customer_id"],
                "customer_name": customer_data["customer_name"],
                "tier": customer_data["tier"],
                "points": customer_data["points"],
                "lifetime_points": customer_data["lifetime_points"],
                "year_to_date_points": customer_data["year_to_date_points"],
                "earning_rate": f"{customer_data['earning_rate']}x",
                "points_to_next_tier": customer_data["points_to_next_tier"],
                "member_since": customer_data["member_since"],
                "perks": customer_data["perks"],
                "redemption_options": redemption_options,
                "recent_transactions": customer_data["recent_transactions"],
            },
            "message": (
                f"{customer_data['customer_name']} is a {customer_data['tier']} member "
                f"with {customer_data['points']} points"
            ),
        }

        if tracker:
            tracker.set_output(result)
            tracker.set_customer_id(customer_data["customer_id"])
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