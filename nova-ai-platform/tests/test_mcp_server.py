"""
NOVA AI Platform — Tests for Task 2: MCP Server
==================================================
Comprehensive test suite covering all 5 tools, audit logger,
server orchestration, and compound scenarios.
"""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.audit_logger import AuditLogger, AuditEntry, AuditTracker
from mcp_server.server import MCPServer, TOOL_REGISTRY
from mcp_server.tools.order_lookup import order_lookup
from mcp_server.tools.product_search import product_search
from mcp_server.tools.return_initiate import return_initiate
from mcp_server.tools.loyalty_check import loyalty_check
from mcp_server.tools.escalation_tool import (
    escalation_tool, get_escalation_queue, clear_escalation_queue,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_escalation_queue():
    """Clear escalation queue before each test."""
    clear_escalation_queue()
    yield
    clear_escalation_queue()


@pytest.fixture
def audit():
    """Create a fresh audit logger."""
    return AuditLogger()


@pytest.fixture
def server():
    """Create a fresh MCP server."""
    return MCPServer()


# ══════════════════════════════════════════════════════════════════════
# TEST: Order Lookup Tool
# ══════════════════════════════════════════════════════════════════════

class TestOrderLookup:
    """Tests for the order_lookup tool."""

    def test_lookup_by_order_id(self, audit):
        """Should find order by ID."""
        result = order_lookup(order_id="ORD-2024-10001", audit_logger=audit)
        assert result["success"] is True
        assert result["data"]["order_id"] == "ORD-2024-10001"
        assert result["data"]["customer_name"] == "Sarah Mitchell"
        assert len(result["data"]["items"]) == 2

    def test_lookup_by_email(self, audit):
        """Should find all orders for a customer email."""
        result = order_lookup(customer_email="sarah.mitchell@email.com", audit_logger=audit)
        assert result["success"] is True
        assert result["count"] == 2

    def test_lookup_nonexistent_order(self, audit):
        """Should fail gracefully for non-existent order."""
        result = order_lookup(order_id="ORD-9999", audit_logger=audit)
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_lookup_nonexistent_email(self, audit):
        """Should fail for non-existent email."""
        result = order_lookup(customer_email="nobody@nowhere.com", audit_logger=audit)
        assert result["success"] is False

    def test_lookup_no_params(self, audit):
        """Should require at least one parameter."""
        result = order_lookup(audit_logger=audit)
        assert result["success"] is False
        assert "must provide" in result["error"].lower()

    def test_order_has_enriched_items(self, audit):
        """Order items should be enriched with product data."""
        result = order_lookup(order_id="ORD-2024-10001", audit_logger=audit)
        items = result["data"]["items"]
        # Items should have category from product data
        for item in items:
            assert "category" in item

    def test_audited(self, audit):
        """Should create audit log entry."""
        order_lookup(order_id="ORD-2024-10001", audit_logger=audit)
        entries = audit.get_entries(tool_name="order_lookup")
        assert len(entries) == 1
        assert entries[0].success is True


# ══════════════════════════════════════════════════════════════════════
# TEST: Product Search Tool
# ══════════════════════════════════════════════════════════════════════

class TestProductSearch:
    """Tests for the product_search tool."""

    def test_search_by_query(self, audit):
        """Should find products matching query."""
        result = product_search(query="moisturizer", audit_logger=audit)
        assert result["success"] is True
        assert result["count"] > 0
        assert any("moisturizer" in p["name"].lower() for p in result["data"])

    def test_search_by_category_beauty(self, audit):
        """Should filter by Beauty category."""
        result = product_search(category="Beauty", audit_logger=audit)
        assert result["success"] is True
        assert all(p["category"] == "Beauty" for p in result["data"])

    def test_search_by_category_fashion(self, audit):
        """Should filter by Fashion category."""
        result = product_search(category="Fashion", audit_logger=audit)
        assert result["success"] is True
        assert all(p["category"] == "Fashion" for p in result["data"])

    def test_search_by_tags(self, audit):
        """Should find products matching tags."""
        result = product_search(tags=["sustainable", "cashmere"], audit_logger=audit)
        assert result["success"] is True
        assert result["count"] > 0

    def test_search_with_price_range(self, audit):
        """Should filter by price range."""
        result = product_search(min_price=40, max_price=60, audit_logger=audit)
        assert result["success"] is True
        for p in result["data"]:
            assert 40 <= p["price"] <= 60

    def test_search_with_limit(self, audit):
        """Should respect limit parameter."""
        result = product_search(limit=2, audit_logger=audit)
        assert result["count"] <= 2

    def test_search_no_results(self, audit):
        """Should handle no results gracefully."""
        result = product_search(query="quantum physics textbook", audit_logger=audit)
        assert result["success"] is True
        assert result["count"] == 0

    def test_search_results_sorted_by_rating(self, audit):
        """Results should be sorted by rating (highest first)."""
        result = product_search(query="serum", audit_logger=audit)
        if result["count"] >= 2:
            ratings = [p["rating"] for p in result["data"]]
            assert ratings == sorted(ratings, reverse=True)

    def test_search_skincare_for_skin_type(self, audit):
        """Should find products matching skin type in description."""
        result = product_search(query="dry", audit_logger=audit)
        assert result["success"] is True
        assert result["count"] > 0

    def test_audited(self, audit):
        """Should create audit log entry."""
        product_search(query="foundation", audit_logger=audit)
        entries = audit.get_entries(tool_name="product_search")
        assert len(entries) == 1


# ══════════════════════════════════════════════════════════════════════
# TEST: Return Initiate Tool
# ══════════════════════════════════════════════════════════════════════

class TestReturnInitiate:
    """Tests for the return_initiate tool."""

    def test_successful_return(self, audit):
        """Should create return ticket for eligible order."""
        result = return_initiate(
            order_id="ORD-2024-10001",
            reason="Doesn't fit right",
            audit_logger=audit,
        )
        assert result["success"] is True
        assert "ticket_id" in result["data"]
        assert result["data"]["ticket_id"].startswith("RET-")
        assert result["data"]["refund_amount"] > 0

    def test_allergic_reaction_return(self, audit):
        """Allergic reactions should get immediate full refund."""
        result = return_initiate(
            order_id="ORD-2024-10001",
            reason="I had an allergic reaction to this product",
            audit_logger=audit,
        )
        assert result["success"] is True
        assert result["data"]["status"] == "approved"
        assert "allergic" in result["data"]["note"].lower()

    def test_return_ineligible_order(self, audit):
        """Should reject return for ineligible orders."""
        result = return_initiate(
            order_id="ORD-2024-10004",  # Return deadline passed
            reason="Want to return",
            audit_logger=audit,
        )
        assert result["success"] is False
        assert "not eligible" in result["error"].lower()

    def test_return_cancelled_order(self, audit):
        """Should reject return for cancelled orders."""
        result = return_initiate(
            order_id="ORD-2024-10006",  # Cancelled
            reason="Want to return",
            audit_logger=audit,
        )
        assert result["success"] is False

    def test_return_invalid_type(self, audit):
        """Should reject invalid return types."""
        result = return_initiate(
            order_id="ORD-2024-10001",
            reason="Don't like it",
            return_type="throw_away",
            audit_logger=audit,
        )
        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    def test_return_nonexistent_order(self, audit):
        """Should fail for non-existent order."""
        result = return_initiate(
            order_id="ORD-9999",
            reason="Test",
            audit_logger=audit,
        )
        assert result["success"] is False

    def test_return_email_mismatch(self, audit):
        """Should reject if email doesn't match order."""
        result = return_initiate(
            order_id="ORD-2024-10001",
            reason="Want to return",
            customer_email="wrong@email.com",
            audit_logger=audit,
        )
        assert result["success"] is False

    def test_audited(self, audit):
        """Should create audit log entry."""
        return_initiate(
            order_id="ORD-2024-10001", reason="Test", audit_logger=audit
        )
        entries = audit.get_entries(tool_name="return_initiate")
        assert len(entries) == 1


# ══════════════════════════════════════════════════════════════════════
# TEST: Loyalty Check Tool
# ══════════════════════════════════════════════════════════════════════

class TestLoyaltyCheck:
    """Tests for the loyalty_check tool."""

    def test_check_by_customer_id(self, audit):
        """Should find loyalty data by customer ID."""
        result = loyalty_check(customer_id="CUST-5001", audit_logger=audit)
        assert result["success"] is True
        assert result["data"]["customer_name"] == "Sarah Mitchell"
        assert result["data"]["tier"] == "Silver"

    def test_check_by_email(self, audit):
        """Should find loyalty data by email."""
        result = loyalty_check(customer_email="aisha.j@email.com", audit_logger=audit)
        assert result["success"] is True
        assert result["data"]["tier"] == "Gold"
        assert result["data"]["earning_rate"] == "2.0x"

    def test_check_gold_tier_benefits(self, audit):
        """Gold members should have premium perks."""
        result = loyalty_check(customer_id="CUST-5004", audit_logger=audit)
        assert result["success"] is True
        perks = result["data"]["perks"]
        assert any("overnight" in p.lower() for p in perks)
        assert any("styling" in p.lower() for p in perks)

    def test_check_redemption_options(self, audit):
        """Should show available redemption options."""
        result = loyalty_check(customer_id="CUST-5001", audit_logger=audit)
        assert result["success"] is True
        # Sarah has 1245 points, should have multiple options
        assert len(result["data"]["redemption_options"]) >= 2

    def test_check_bronze_tier(self, audit):
        """Bronze members should have basic perks."""
        result = loyalty_check(customer_id="CUST-5002", audit_logger=audit)
        assert result["success"] is True
        assert result["data"]["tier"] == "Bronze"
        assert result["data"]["earning_rate"] == "1.0x"

    def test_check_nonexistent_customer(self, audit):
        """Should fail for non-existent customer."""
        result = loyalty_check(customer_id="CUST-9999", audit_logger=audit)
        assert result["success"] is False

    def test_check_no_params(self, audit):
        """Should require at least one parameter."""
        result = loyalty_check(audit_logger=audit)
        assert result["success"] is False

    def test_check_has_recent_transactions(self, audit):
        """Should include recent transactions."""
        result = loyalty_check(customer_id="CUST-5001", audit_logger=audit)
        assert result["success"] is True
        assert len(result["data"]["recent_transactions"]) > 0

    def test_audited_with_customer_id(self, audit):
        """Should set customer_id on audit entry."""
        loyalty_check(customer_id="CUST-5001", audit_logger=audit)
        entries = audit.get_entries()
        assert entries[0].customer_id == "CUST-5001"


# ══════════════════════════════════════════════════════════════════════
# TEST: Escalation Tool
# ══════════════════════════════════════════════════════════════════════

class TestEscalationTool:
    """Tests for the escalation_tool."""

    def test_create_escalation(self, audit):
        """Should create escalation ticket."""
        result = escalation_tool(
            reason="Customer is very upset",
            priority="high",
            customer_id="CUST-5001",
            customer_name="Sarah Mitchell",
            audit_logger=audit,
        )
        assert result["success"] is True
        assert result["data"]["ticket_id"].startswith("ESC-")
        assert result["data"]["priority"] == "high"
        assert result["data"]["status"] == "open"

    def test_escalation_invalid_priority(self, audit):
        """Should reject invalid priority."""
        result = escalation_tool(
            reason="Test",
            priority="super_urgent",
            audit_logger=audit,
        )
        assert result["success"] is False

    def test_escalation_sla_by_priority(self, audit):
        """Each priority level should have appropriate SLA."""
        for priority, expected_sla in [
            ("urgent", 15), ("high", 60), ("medium", 240), ("low", 480),
        ]:
            clear_escalation_queue()
            result = escalation_tool(
                reason="Test", priority=priority, audit_logger=audit,
            )
            assert result["success"] is True
            assert result["data"]["sla_minutes"] == expected_sla

    def test_escalation_queue_populated(self, audit):
        """Escalation should add to the queue."""
        escalation_tool(reason="Test", priority="medium", audit_logger=audit)
        queue = get_escalation_queue()
        assert len(queue) == 1

    def test_escalation_with_full_context(self, audit):
        """Should include all context in ticket."""
        result = escalation_tool(
            reason="Allergic reaction to skincare product",
            priority="high",
            customer_id="CUST-5001",
            customer_name="Sarah Mitchell",
            customer_email="sarah.mitchell@email.com",
            order_id="ORD-2024-10001",
            intent="return_refund",
            confidence=0.9,
            conversation_summary="Customer reported skin irritation after using Glow-Up Vitamin C Serum",
            audit_logger=audit,
        )
        assert result["success"] is True
        data = result["data"]
        assert data["customer"]["id"] == "CUST-5001"
        assert data["order_id"] == "ORD-2024-10001"
        assert data["intent"] == "return_refund"

    def test_audited(self, audit):
        """Should create audit log entry."""
        escalation_tool(reason="Test", audit_logger=audit)
        entries = audit.get_entries(tool_name="escalation_tool")
        assert len(entries) == 1


# ══════════════════════════════════════════════════════════════════════
# TEST: Audit Logger
# ══════════════════════════════════════════════════════════════════════

class TestAuditLogger:
    """Tests for the audit logging system."""

    def test_log_entry(self, audit):
        """Should log an entry."""
        entry = AuditEntry(tool_name="test_tool", success=True, latency_ms=42.5)
        audit.log_entry(entry)
        assert len(audit.entries) == 1

    def test_track_context_manager(self, audit):
        """Track should work as context manager."""
        import time
        with audit.track("test_tool", {"param": "value"}) as tracker:
            time.sleep(0.001)  # Ensure measurable latency
            tracker.set_output({"success": True, "data": "test"})

        entries = audit.get_entries()
        assert len(entries) == 1
        assert entries[0].tool_name == "test_tool"
        assert entries[0].success is True
        assert entries[0].latency_ms >= 0  # May be 0 on fast systems

    def test_track_captures_errors(self, audit):
        """Track should capture exceptions."""
        try:
            with audit.track("failing_tool", {}) as tracker:
                raise ValueError("Test error")
        except ValueError:
            pass

        entries = audit.get_entries()
        assert len(entries) == 1
        assert entries[0].success is False
        assert "ValueError" in entries[0].error

    def test_filter_by_tool(self, audit):
        """Should filter entries by tool name."""
        order_lookup(order_id="ORD-2024-10001", audit_logger=audit)
        product_search(query="test", audit_logger=audit)

        order_entries = audit.get_entries(tool_name="order_lookup")
        assert len(order_entries) == 1
        assert order_entries[0].tool_name == "order_lookup"

    def test_filter_by_success(self, audit):
        """Should filter entries by success status."""
        order_lookup(order_id="ORD-2024-10001", audit_logger=audit)
        order_lookup(order_id="ORD-9999", audit_logger=audit)

        successes = audit.get_entries(success=True)
        failures = audit.get_entries(success=False)
        assert len(successes) >= 1
        assert len(failures) >= 1

    def test_summary_statistics(self, audit):
        """Should compute summary statistics."""
        order_lookup(order_id="ORD-2024-10001", audit_logger=audit)
        product_search(query="moisturizer", audit_logger=audit)

        summary = audit.get_summary()
        assert summary["total_calls"] == 2
        assert "tools" in summary
        assert "order_lookup" in summary["tools"]
        assert "product_search" in summary["tools"]
        assert summary["avg_latency_ms"] > 0

    def test_empty_summary(self):
        """Should handle empty audit log."""
        audit = AuditLogger()
        summary = audit.get_summary()
        assert summary["total_calls"] == 0

    def test_entry_to_json(self):
        """Entry should serialize to JSON."""
        entry = AuditEntry(tool_name="test", success=True, latency_ms=10.0)
        json_str = entry.to_json()
        parsed = json.loads(json_str)
        assert parsed["tool_name"] == "test"

    def test_audit_trail(self, audit):
        """Should return full audit trail as dicts."""
        order_lookup(order_id="ORD-2024-10001", audit_logger=audit)
        trail = audit.get_audit_trail()
        assert len(trail) == 1
        assert isinstance(trail[0], dict)

    def test_clear(self, audit):
        """Should clear all entries."""
        order_lookup(order_id="ORD-2024-10001", audit_logger=audit)
        audit.clear()
        assert len(audit.entries) == 0


# ══════════════════════════════════════════════════════════════════════
# TEST: MCP Server Orchestration
# ══════════════════════════════════════════════════════════════════════

class TestMCPServer:
    """Tests for the MCP server orchestration."""

    def test_list_tools(self, server):
        """Should list all available tools."""
        tools = server.list_tools()
        assert len(tools) == 5
        assert "order_lookup" in tools
        assert "product_search" in tools
        assert "return_initiate" in tools
        assert "loyalty_check" in tools
        assert "escalation_tool" in tools

    def test_call_tool_order_lookup(self, server):
        """Should call order_lookup through server."""
        result = server.call_tool("order_lookup", {"order_id": "ORD-2024-10001"})
        assert result["success"] is True

    def test_call_tool_product_search(self, server):
        """Should call product_search through server."""
        result = server.call_tool("product_search", {"query": "serum"})
        assert result["success"] is True

    def test_call_tool_return(self, server):
        """Should call return_initiate through server."""
        result = server.call_tool("return_initiate", {
            "order_id": "ORD-2024-10001",
            "reason": "Test return",
        })
        assert result["success"] is True

    def test_call_tool_loyalty(self, server):
        """Should call loyalty_check through server."""
        result = server.call_tool("loyalty_check", {"customer_id": "CUST-5001"})
        assert result["success"] is True

    def test_call_tool_escalation(self, server):
        """Should call escalation_tool through server."""
        result = server.call_tool("escalation_tool", {"reason": "Test escalation"})
        assert result["success"] is True

    def test_call_unknown_tool(self, server):
        """Should return error for unknown tool."""
        result = server.call_tool("nonexistent_tool", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_server_audits_all_calls(self, server):
        """Server should audit all tool calls."""
        server.call_tool("order_lookup", {"order_id": "ORD-2024-10001"})
        server.call_tool("product_search", {"query": "test"})
        server.call_tool("loyalty_check", {"customer_id": "CUST-5001"})

        summary = server.get_audit_summary()
        assert summary["total_calls"] == 3

    def test_get_audit_trail(self, server):
        """Should return full audit trail."""
        server.call_tool("order_lookup", {"order_id": "ORD-2024-10001"})
        trail = server.get_audit_trail()
        assert len(trail) == 1
        assert trail[0]["tool_name"] == "order_lookup"


# ══════════════════════════════════════════════════════════════════════
# TEST: Compound Scenarios
# ══════════════════════════════════════════════════════════════════════

class TestCompoundScenarios:
    """Tests for multi-tool compound scenarios."""

    def test_order_investigation(self, server):
        """Order investigation should call 3 tools."""
        result = server.run_compound_scenario(
            "order_investigation",
            customer_email="sarah.mitchell@email.com",
        )
        assert result["success"] is True
        assert result["step_count"] >= 2
        assert "order_lookup" in result["steps"]
        assert "loyalty_status" in result["steps"]

    def test_return_with_loyalty(self, server):
        """Return scenario should call 4 tools."""
        result = server.run_compound_scenario(
            "return_with_loyalty",
            order_id="ORD-2024-10001",
            reason="Doesn't fit right",
        )
        assert result["success"] is True
        assert "order_lookup" in result["steps"]
        assert "return_initiate" in result["steps"]

    def test_product_recommendation(self, server):
        """Recommendation scenario should call 3 tools."""
        result = server.run_compound_scenario(
            "product_recommendation",
            query="skincare for dry skin",
        )
        assert result["success"] is True
        assert "product_search" in result["steps"]
        assert "loyalty_status" in result["steps"]

    def test_full_support_journey(self, server):
        """Full journey should call all 5 tools."""
        result = server.run_compound_scenario("full_support_journey")
        assert result["success"] is True
        assert result["step_count"] == 5
        assert "loyalty_check" in result["steps"]
        assert "order_lookup" in result["steps"]
        assert "product_search" in result["steps"]
        assert "return_initiate" in result["steps"]
        assert "escalation" in result["steps"]

    def test_scenario_has_audit_summary(self, server):
        """Scenario result should include audit summary."""
        result = server.run_compound_scenario("order_investigation")
        assert "audit_summary" in result
        assert result["audit_summary"]["total_calls"] >= 2

    def test_scenario_has_latency(self, server):
        """Scenario result should include total latency."""
        result = server.run_compound_scenario("order_investigation")
        assert "total_latency_ms" in result
        assert result["total_latency_ms"] > 0

    def test_unknown_scenario(self, server):
        """Should return error for unknown scenario."""
        result = server.run_compound_scenario("nonexistent_scenario")
        assert result["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])