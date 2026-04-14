"""
NOVA AI Platform — Product Search Tool
========================================
Searches the NOVA product catalog by query, category, tags, or attributes.
"""

import json
from pathlib import Path
from typing import Optional

from mcp_server.audit_logger import AuditLogger

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_products() -> list[dict]:
    """Load products from JSON data file."""
    with open(DATA_DIR / "products.json") as f:
        return json.load(f)


def product_search(
    query: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[list[str]] = None,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    in_stock_only: bool = True,
    limit: int = 5,
    audit_logger: Optional[AuditLogger] = None,
) -> dict:
    """
    Search the NOVA product catalog.

    Args:
        query: Free-text search across name, description, tags.
        category: Filter by category ("Beauty" or "Fashion").
        tags: Filter by product tags (any match).
        max_price: Maximum price filter.
        min_price: Minimum price filter.
        in_stock_only: Only return in-stock products.
        limit: Maximum number of results.
        audit_logger: Optional audit logger.

    Returns:
        Dict with success status and matching products.
    """
    params = {
        "query": query, "category": category, "tags": tags,
        "max_price": max_price, "min_price": min_price,
        "in_stock_only": in_stock_only, "limit": limit,
    }
    tracker = audit_logger.track("product_search", params) if audit_logger else None

    if tracker:
        tracker.__enter__()

    try:
        products = _load_products()
        results = list(products)  # Start with all

        # Filter: in stock
        if in_stock_only:
            results = [p for p in results if p.get("in_stock", True)]

        # Filter: category
        if category:
            cat_lower = category.lower()
            results = [
                p for p in results
                if p.get("category", "").lower() == cat_lower
                or p.get("subcategory", "").lower() == cat_lower
            ]

        # Filter: price range
        if min_price is not None:
            results = [p for p in results if p.get("price", 0) >= min_price]
        if max_price is not None:
            results = [p for p in results if p.get("price", 0) <= max_price]

        # Filter: tags (any match)
        if tags:
            tag_lowers = [t.lower() for t in tags]
            results = [
                p for p in results
                if any(t.lower() in tag_lowers for t in p.get("tags", []))
            ]

        # Filter: free-text query
        if query:
            q_lower = query.lower()
            scored = []
            for p in results:
                score = 0.0
                # Name match (highest weight)
                if q_lower in p.get("name", "").lower():
                    score += 3.0
                # Description match
                if q_lower in p.get("description", "").lower():
                    score += 2.0
                # Tag match
                if any(q_lower in t.lower() for t in p.get("tags", [])):
                    score += 1.5
                # Category/subcategory match
                if q_lower in p.get("category", "").lower():
                    score += 1.0
                if q_lower in p.get("subcategory", "").lower():
                    score += 1.0
                # Ingredient/material match
                for field_name in ["ingredients", "materials"]:
                    if q_lower in p.get(field_name, "").lower():
                        score += 1.0
                # Skin type/concern match
                for field_name in ["skin_type", "skin_concerns"]:
                    val = p.get(field_name, "")
                    if isinstance(val, str) and q_lower in val.lower():
                        score += 1.0
                    elif isinstance(val, list) and any(q_lower in str(v).lower() for v in val):
                        score += 1.0
                if score > 0:
                    scored.append((score, p))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = [p for _, p in scored]

        # Sort by rating (highest first)
        results.sort(key=lambda p: p.get("rating", 0), reverse=True)

        # Apply limit
        results = results[:limit]

        result = {
            "success": True,
            "data": results,
            "count": len(results),
            "message": f"Found {len(results)} product(s)" + (f" matching '{query}'" if query else ""),
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