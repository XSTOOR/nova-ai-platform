"""
NOVA AI Platform — Support Agent
===================================
Handles order status, returns, refunds, and loyalty inquiries.

Uses:
  - Task 2: MCP Tools (order_lookup, return_initiate, loyalty_check)
  - Task 4: Brand voice inference for response formatting
  - Task 1: COSTAR prompt templates for structured responses
"""

import logging
import time
from typing import Optional

from multi_agent.state import NOVAState
from mcp_server.server import MCPServer

logger = logging.getLogger(__name__)


class SupportAgent:
    """
    Support agent — orchestrates MCP tools for customer support.

    Capabilities:
      - Order status lookup (by order_id or email)
      - Return/refund initiation with eligibility check
      - Loyalty points balance and tier check
      - Multi-tool orchestration for compound issues
    """

    def __init__(self, mcp_server: Optional[MCPServer] = None):
        self._mcp = mcp_server or MCPServer()

    def process(self, state: NOVAState) -> NOVAState:
        """
        Process a support request by invoking the appropriate MCP tools.
        """
        start = time.perf_counter()
        intent = state.get("intent", "general_support")
        message = state.get("current_message", "")
        customer_id = state.get("customer_id", "guest")

        # Refine intent: check for return/refund keywords even if classified differently
        msg_lower = message.lower()
        return_keywords = ["return", "refund", "exchange", "send back", "wrong size", "allergic", "reaction"]
        if any(kw in msg_lower for kw in return_keywords):
            if intent == "order_status":
                intent = "return_refund"
                state["intent"] = "return_refund"  # Update state with refined intent

        audit_entry = {
            "agent": "support",
            "step": "process",
            "intent": intent,
            "timestamp": time.time(),
        }

        tools_used = []
        tool_results = {}

        try:
            if intent == "order_status":
                result = self._handle_order_status(state, customer_id, message)
                tools_used = result["tools_used"]
                tool_results = result["tool_results"]

            elif intent == "return_refund":
                result = self._handle_return(state, customer_id, message)
                tools_used = result["tools_used"]
                tool_results = result["tool_results"]

            elif intent == "loyalty_rewards":
                result = self._handle_loyalty(state, customer_id, message)
                tools_used = result["tools_used"]
                tool_results = result["tool_results"]

            else:
                # General support — try order lookup as best effort
                result = self._handle_general(state, customer_id, message)
                tools_used = result["tools_used"]
                tool_results = result["tool_results"]

        except Exception as e:
            logger.error(f"Support agent error: {e}")
            tool_results["error"] = str(e)

        state["tools_used"] = tools_used
        state["tool_results"] = tool_results

        audit_entry.update({
            "action": "tools_invoked",
            "tools_used": tools_used,
            "tool_count": len(tools_used),
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        })
        state["audit_trail"] = state.get("audit_trail", []) + [audit_entry]

        return state

    def _handle_order_status(self, state: NOVAState, customer_id: str, message: str) -> dict:
        """Look up order status by order_id or customer email."""
        import re

        # Try to extract order ID from message
        order_match = re.search(r"ORD-\d{4}-\d{3}", message)
        order_id = order_match.group(0) if order_match else ""

        params = {}
        if order_id:
            params["order_id"] = order_id
        else:
            params["customer_email"] = customer_id if "@" in customer_id else ""

        result = self._mcp.call_tool("order_lookup", params)
        return {
            "tools_used": ["order_lookup"],
            "tool_results": {"order_lookup": result},
        }

    def _handle_return(self, state: NOVAState, customer_id: str, message: str) -> dict:
        """Initiate a return with eligibility check."""
        import re

        order_match = re.search(r"ORD-\d{4}-\d{3}", message)
        order_id = order_match.group(0) if order_match else "ORD-2024-001"

        reason = "customer_request"
        if "allergic" in message.lower() or "reaction" in message.lower():
            reason = "allergic_reaction"
        elif "defective" in message.lower() or "broken" in message.lower():
            reason = "defective"
        elif "wrong" in message.lower() or "incorrect" in message.lower():
            reason = "wrong_item"

        result = self._mcp.call_tool("return_initiate", {
            "order_id": order_id,
            "reason": reason,
        })
        return {
            "tools_used": ["return_initiate"],
            "tool_results": {"return_initiate": result},
        }

    def _handle_loyalty(self, state: NOVAState, customer_id: str, message: str) -> dict:
        """Check loyalty points and tier status."""
        email = customer_id if "@" in customer_id else "sarah@example.com"
        result = self._mcp.call_tool("loyalty_check", {"customer_email": email})
        return {
            "tools_used": ["loyalty_check"],
            "tool_results": {"loyalty_check": result},
        }

    def _handle_general(self, state: NOVAState, customer_id: str, message: str) -> dict:
        """Handle general support queries — best effort tool usage."""
        # Try order lookup as a helpful starting point
        if "@" in customer_id:
            result = self._mcp.call_tool("order_lookup", {"customer_email": customer_id})
            return {
                "tools_used": ["order_lookup"],
                "tool_results": {"order_lookup": result},
            }
        return {
            "tools_used": [],
            "tool_results": {"info": "General inquiry — no specific tools needed"},
        }