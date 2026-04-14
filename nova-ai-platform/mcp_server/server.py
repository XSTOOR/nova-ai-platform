"""
NOVA AI Platform — MCP Server Orchestration
==============================================
Central server that coordinates all 5 backend tools.
Includes compound demo scenarios for multi-tool orchestration.

Tools Available:
  1. order_lookup    — Find orders by ID or email
  2. product_search  — Search product catalog
  3. return_initiate — Process returns/exchanges
  4. loyalty_check   — Query rewards points & tier
  5. escalation_tool — Route to human agent

Usage:
    server = MCPServer()
    result = server.call_tool("order_lookup", {"order_id": "ORD-2024-10001"})

    # Or run compound scenarios
    server.run_compound_scenario("order_investigation", customer_email="sarah.mitchell@email.com")
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from mcp_server.audit_logger import AuditLogger, AuditEntry
from mcp_server.tools.order_lookup import order_lookup
from mcp_server.tools.product_search import product_search
from mcp_server.tools.return_initiate import return_initiate
from mcp_server.tools.loyalty_check import loyalty_check
from mcp_server.tools.escalation_tool import escalation_tool, clear_escalation_queue


# ──────────────────────────────────────────────────────────────────────
# Tool Registry
# ──────────────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "order_lookup": {
        "function": order_lookup,
        "description": "Look up order details by order_id or customer_email",
        "parameters": {
            "order_id": {"type": "string", "required": False, "description": "Order ID to look up"},
            "customer_email": {"type": "string", "required": False, "description": "Customer email"},
        },
    },
    "product_search": {
        "function": product_search,
        "description": "Search the NOVA product catalog",
        "parameters": {
            "query": {"type": "string", "required": False, "description": "Free-text search query"},
            "category": {"type": "string", "required": False, "description": "Category filter (Beauty/Fashion)"},
            "tags": {"type": "array", "required": False, "description": "Tags to filter by"},
            "max_price": {"type": "number", "required": False, "description": "Maximum price"},
            "limit": {"type": "integer", "required": False, "description": "Max results (default 5)"},
        },
    },
    "return_initiate": {
        "function": return_initiate,
        "description": "Initiate a return or exchange for an order",
        "parameters": {
            "order_id": {"type": "string", "required": True, "description": "Order ID"},
            "reason": {"type": "string", "required": True, "description": "Reason for return"},
            "return_type": {"type": "string", "required": False, "description": "refund/exchange/store_credit"},
        },
    },
    "loyalty_check": {
        "function": loyalty_check,
        "description": "Check loyalty points, tier, and redemption options",
        "parameters": {
            "customer_id": {"type": "string", "required": False, "description": "Customer ID"},
            "customer_email": {"type": "string", "required": False, "description": "Customer email"},
        },
    },
    "escalation_tool": {
        "function": escalation_tool,
        "description": "Create escalation ticket for human agent handoff",
        "parameters": {
            "reason": {"type": "string", "required": True, "description": "Escalation reason"},
            "priority": {"type": "string", "required": False, "description": "low/medium/high/urgent"},
            "customer_id": {"type": "string", "required": False, "description": "Customer ID"},
        },
    },
}


# ──────────────────────────────────────────────────────────────────────
# MCP Server
# ──────────────────────────────────────────────────────────────────────

class MCPServer:
    """
    NOVA MCP Server — orchestrates all backend tools with audit logging.

    Usage:
        server = MCPServer()
        result = server.call_tool("order_lookup", {"order_id": "ORD-2024-10001"})
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self.audit = AuditLogger(persist_dir=persist_dir)
        self._tool_registry = TOOL_REGISTRY

    def list_tools(self) -> dict:
        """List all available tools with their descriptions."""
        return {
            name: {
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for name, info in self._tool_registry.items()
        }

    def call_tool(self, tool_name: str, params: dict) -> dict:
        """
        Call a registered tool with the given parameters.

        Args:
            tool_name: Name of the tool to call.
            params: Parameters to pass to the tool.

        Returns:
            Tool result dict.
        """
        if tool_name not in self._tool_registry:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}. Available: {list(self._tool_registry.keys())}",
            }

        tool_func = self._tool_registry[tool_name]["function"]

        # Call with audit logger
        return tool_func(**params, audit_logger=self.audit)

    def get_audit_summary(self) -> dict:
        """Get audit summary statistics."""
        return self.audit.get_summary()

    def get_audit_trail(self) -> list[dict]:
        """Get full audit trail."""
        return self.audit.get_audit_trail()

    # ──────────────────────────────────────────────────────────────
    # Compound Scenarios
    # ──────────────────────────────────────────────────────────────

    def run_compound_scenario(
        self, scenario: str, **kwargs
    ) -> dict:
        """
        Run a compound multi-tool scenario.

        Args:
            scenario: Name of the scenario to run.
            **kwargs: Scenario-specific parameters.

        Returns:
            Dict with scenario results and audit trail.
        """
        scenarios = {
            "order_investigation": self._scenario_order_investigation,
            "return_with_loyalty": self._scenario_return_with_loyalty,
            "product_recommendation": self._scenario_product_recommendation,
            "full_support_journey": self._scenario_full_support_journey,
        }

        if scenario not in scenarios:
            return {
                "success": False,
                "error": f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}",
            }

        handler = scenarios[scenario]
        start = time.perf_counter()
        result = handler(**kwargs)
        total_ms = (time.perf_counter() - start) * 1000

        result["scenario"] = scenario
        result["total_latency_ms"] = round(total_ms, 2)
        result["audit_summary"] = self.audit.get_summary()

        return result

    def _scenario_order_investigation(
        self,
        customer_email: str = "sarah.mitchell@email.com",
        **kwargs,
    ) -> dict:
        """
        Scenario: Customer asks "Where is my order?"
        Steps:
          1. Look up all orders for the customer
          2. For each recent order, search for related products (cross-sell)
          3. Check loyalty status for personalized recommendations
        """
        steps = {}

        # Step 1: Find customer orders
        orders_result = self.call_tool("order_lookup", {"customer_email": customer_email})
        steps["order_lookup"] = orders_result

        # Step 2: Search for related products
        if orders_result["success"] and orders_result.get("data"):
            orders_data = orders_result["data"]
            if isinstance(orders_data, list) and len(orders_data) > 0:
                last_order = orders_data[0]
                # Get the first item's category for recommendations
                items = last_order.get("items", [])
                if items:
                    first_item = items[0]
                    search_query = first_item.get("name", "").split()[0]
                    product_result = self.call_tool("product_search", {
                        "query": search_query,
                        "limit": 3,
                    })
                    steps["related_products"] = product_result

        # Step 3: Check loyalty
        if orders_result["success"] and orders_result.get("data"):
            orders_data = orders_result["data"]
            if isinstance(orders_data, list):
                cid = orders_data[0].get("customer_id")
            else:
                cid = orders_data.get("customer_id")

            if cid:
                loyalty_result = self.call_tool("loyalty_check", {"customer_id": cid})
                steps["loyalty_status"] = loyalty_result

        return {
            "success": True,
            "scenario_name": "Order Investigation",
            "description": f"Full order investigation for {customer_email}",
            "steps": steps,
            "step_count": len(steps),
        }

    def _scenario_return_with_loyalty(
        self,
        order_id: str = "ORD-2024-10001",
        reason: str = "Doesn't fit right",
        customer_email: str = "sarah.mitchell@email.com",
        **kwargs,
    ) -> dict:
        """
        Scenario: Customer wants to return an item
        Steps:
          1. Look up the order
          2. Initiate return
          3. Check loyalty status for goodwill gesture
          4. Search for alternative products (exchange suggestion)
        """
        steps = {}

        # Step 1: Lookup order
        order_result = self.call_tool("order_lookup", {
            "order_id": order_id,
            "customer_email": customer_email,
        })
        steps["order_lookup"] = order_result

        # Step 2: Initiate return
        if order_result["success"]:
            return_result = self.call_tool("return_initiate", {
                "order_id": order_id,
                "reason": reason,
                "customer_email": customer_email,
            })
            steps["return_initiate"] = return_result

        # Step 3: Check loyalty
        if order_result["success"]:
            order_data = order_result["data"]
            if isinstance(order_data, list):
                cid = order_data[0].get("customer_id")
            else:
                cid = order_data.get("customer_id")
            if cid:
                loyalty_result = self.call_tool("loyalty_check", {"customer_id": cid})
                steps["loyalty_status"] = loyalty_result

        # Step 4: Suggest alternatives
        alt_result = self.call_tool("product_search", {
            "query": "foundation",  # Based on original order
            "category": "Beauty",
            "limit": 3,
        })
        steps["alternative_products"] = alt_result

        return {
            "success": True,
            "scenario_name": "Return with Loyalty Check",
            "description": f"Return processing for order {order_id} with loyalty integration",
            "steps": steps,
            "step_count": len(steps),
        }

    def _scenario_product_recommendation(
        self,
        query: str = "skincare for dry skin",
        customer_email: str = "sarah.mitchell@email.com",
        **kwargs,
    ) -> dict:
        """
        Scenario: Customer asks for product recommendations
        Steps:
          1. Search products matching query
          2. Look up customer's order history (for personalization)
          3. Check loyalty status (for discounts/points)
        """
        steps = {}

        # Step 1: Product search
        search_result = self.call_tool("product_search", {
            "query": query,
            "limit": 5,
        })
        steps["product_search"] = search_result

        # Step 2: Customer history
        orders_result = self.call_tool("order_lookup", {"customer_email": customer_email})
        steps["order_history"] = orders_result

        # Step 3: Loyalty check
        if orders_result["success"] and orders_result.get("data"):
            orders_data = orders_result["data"]
            if isinstance(orders_data, list):
                cid = orders_data[0].get("customer_id")
            else:
                cid = orders_data.get("customer_id")
            if cid:
                loyalty_result = self.call_tool("loyalty_check", {"customer_id": cid})
                steps["loyalty_status"] = loyalty_result

        return {
            "success": True,
            "scenario_name": "Product Recommendation",
            "description": f"Personalized recommendations for '{query}'",
            "steps": steps,
            "step_count": len(steps),
        }

    def _scenario_full_support_journey(
        self,
        customer_email: str = "sarah.mitchell@email.com",
        order_id: str = "ORD-2024-10005",
        **kwargs,
    ) -> dict:
        """
        Scenario: Full customer support journey
        Steps:
          1. Check loyalty status
          2. Look up order
          3. Search for alternative products
          4. Initiate return
          5. Escalate to human (for personalized follow-up)
        """
        steps = {}

        # Step 1: Loyalty check
        loyalty_result = self.call_tool("loyalty_check", {"customer_email": customer_email})
        steps["loyalty_check"] = loyalty_result

        # Step 2: Order lookup
        order_result = self.call_tool("order_lookup", {"order_id": order_id})
        steps["order_lookup"] = order_result

        # Step 3: Product search
        product_result = self.call_tool("product_search", {
            "query": "cashmere loungewear",
            "limit": 3,
        })
        steps["product_search"] = product_result

        # Step 4: Return
        return_result = self.call_tool("return_initiate", {
            "order_id": order_id,
            "reason": "Size doesn't fit, looking for different size",
        })
        steps["return_initiate"] = return_result

        # Step 5: Escalate
        customer_id = None
        customer_name = None
        if loyalty_result["success"]:
            customer_id = loyalty_result["data"]["customer_id"]
            customer_name = loyalty_result["data"]["customer_name"]

        esc_result = self.call_tool("escalation_tool", {
            "reason": "Customer needs size exchange for cashmere lounge set. Gold tier member — priority handling.",
            "priority": "high",
            "customer_id": customer_id,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "order_id": order_id,
        })
        steps["escalation"] = esc_result

        return {
            "success": True,
            "scenario_name": "Full Support Journey",
            "description": f"Complete support journey for {customer_email}",
            "steps": steps,
            "step_count": len(steps),
        }