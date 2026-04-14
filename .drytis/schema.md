# NOVA AI Platform - Data Schema

## Mock Data Files

### products.json
```json
{
  "id": "PRD-XXX",
  "name": "string",
  "category": "Beauty | Fashion",
  "subcategory": "string",
  "price": "float",
  "description": "string",
  "ingredients": "string (beauty)",
  "materials": "string (fashion)",
  "shades/sizes/colors": "array",
  "in_stock": "boolean",
  "rating": "float",
  "reviews_count": "int",
  "bestseller": "boolean",
  "tags": "array<string>"
}
```

### orders.json
```json
{
  "order_id": "ORD-2024-XXXXX",
  "customer_id": "CUST-XXXX",
  "customer_name": "string",
  "customer_email": "string",
  "items": [{"product_id": "string", "name": "string", "qty": "int", "price": "float"}],
  "total": "float",
  "status": "delivered | shipped | processing | cancelled",
  "order_date": "date",
  "shipping_address": "string",
  "tracking_number": "string | null",
  "payment_method": "string",
  "return_eligible": "boolean",
  "return_deadline": "date | null"
}
```

### faqs.json
```json
{
  "id": "FAQ-XXX",
  "category": "string",
  "question": "string",
  "answer": "string",
  "keywords": "array<string>"
}
```

### brand_voice_samples.json
```json
{
  "id": "BVS-XXX",
  "category": "string",
  "context": "string",
  "input": "string",
  "ideal_response": "string"
}
```

## API Endpoints (MCP Tools)

| Tool | Input | Output |
|------|-------|--------|
| order_lookup | order_id or customer_email | Order details |
| product_search | query, category?, filters? | Product list |
| return_initiate | order_id, item_id, reason | Return confirmation |
| loyalty_check | customer_id or email | Points, tier, history |
| escalation_tool | reason, priority, context | Escalation ticket |

## LangGraph State Schema
```python
class ConversationState(TypedDict):
    messages: list[BaseMessage]
    intent: str | None
    confidence: float
    customer_id: str | None
    order_id: str | None
    tool_results: list[dict]
    escalation_needed: bool
    escalation_reason: str | None
    audit_trail: list[dict]
    response: str | None
```