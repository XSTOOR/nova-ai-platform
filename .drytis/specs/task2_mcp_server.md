# Task 2: MCP Server — NOVA Backend Tool Integration

## Objective
Build a complete MCP (Model Context Protocol) server with 5 backend tools, audit logging for every call, and a compound demo scenario showing multi-tool orchestration.

## Files to Create/Modify
- `mcp_server/audit_logger.py` — Audit logging for all tool invocations
- `mcp_server/tools/order_lookup.py` — Order retrieval & tracking
- `mcp_server/tools/product_search.py` — Product catalog search
- `mcp_server/tools/return_initiate.py` — Return/exchange processing
- `mcp_server/tools/loyalty_check.py` — Loyalty points & tier queries
- `mcp_server/tools/escalation_tool.py` — Human agent escalation
- `mcp_server/server.py` — MCP server orchestration with compound scenarios
- `mcp_server/__init__.py` — Updated module exports
- `mcp_server/tools/__init__.py` — Updated tool exports
- `tests/test_mcp_server.py` — Full test suite
- `notebooks/task2_mcp_server.ipynb` — Colab notebook

## Acceptance Criteria

### 5 Backend Tools
- [ ] **order_lookup**: Lookup by order_id or customer_email, returns full order details
- [ ] **product_search**: Search by query/category/tags, returns matching products
- [ ] **return_initiate**: Validate eligibility, create return ticket, update order status
- [ ] **loyalty_check**: Lookup by customer_id/email, returns points/tier/history
- [ ] **escalation_tool**: Create escalation ticket with priority, context, and reason

### Audit Logger
- [ ] Logs every tool call: timestamp, tool_name, input, output, latency_ms
- [ ] Structured JSON logs with configurable output
- [ ] Query/filter audit logs by tool, date range, success/failure
- [ ] Summary statistics (call counts, avg latency, error rates)

### Compound Demo
- [ ] Scenario: Customer asks "Where's my order?" → order_lookup → product_search for related items → loyalty_check for upsell
- [ ] Shows sequential multi-tool orchestration
- [ ] All calls logged with full audit trail

### Tests
- [ ] Unit tests for each tool (happy path + error cases)
- [ ] Audit logger tests (logging, querying, stats)
- [ ] Compound scenario integration test
- [ ] Edge cases: invalid inputs, missing data, concurrent calls

### Colab Notebook
- [ ] Self-contained, runs on free tier
- [ ] Demonstrates all 5 tools with example calls
- [ ] Shows compound scenario with audit trail
- [ ] Includes latency metrics and visualizations