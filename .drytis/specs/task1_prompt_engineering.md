# Task 1: Prompt Engineering — NOVA Support Brain

## Objective
Build the complete prompt engineering system for NOVA's AI support agent using the COSTAR framework, with intent classification, escalation logic, and prompt injection defense.

## Files to Create/Modify
- `prompts/costar_templates.py` — COSTAR framework prompt builder
- `prompts/intent_classifier.py` — 5-category intent classification with confidence
- `prompts/escalation_logic.py` — Threshold + keyword-based escalation
- `prompts/injection_defense.py` — Regex + LLM injection detection
- `tests/test_prompts.py` — Full test suite
- `notebooks/task1_prompt_engineering.ipynb` — Colab notebook

## Acceptance Criteria

### COSTAR Framework
- [ ] Prompt builder accepts (C)ontext, (O)bjective, (S)teps, (A)udience, (R)esponse format
- [ ] Pre-built templates for all 5 intent categories
- [ ] Templates enforce NOVA brand voice (warm, emoji-friendly, proactive)
- [ ] Prompts are token-efficient (<1000 tokens for system prompts)

### Intent Classifier
- [ ] Classifies into 5 categories: order_status, product_inquiry, return_refund, loyalty_rewards, general_support
- [ ] Returns confidence score (0.0–1.0) for each classification
- [ ] Handles ambiguous inputs with appropriate low confidence
- [ ] Works with both structured and freeform customer messages
- [ ] Handles edge cases: empty input, non-English, gibberish

### Escalation Logic
- [ ] Triggers escalation when confidence < 0.4 (low confidence)
- [ ] Triggers escalation on sensitive topic keywords (medical, legal, fraud)
- [ ] Triggers escalation on explicit human request keywords
- [ ] Logs escalation reason with full context
- [ ] Supports configurable thresholds via config/settings.py

### Injection Defense
- [ ] Blocks 10+ known injection patterns via regex
- [ ] LLM-based secondary scoring for subtle injections
- [ ] Returns threat level (safe/suspicious/blocked)
- [ ] Sanitizes input while preserving legitimate content
- [ ] Logs all blocked attempts for security review

### Tests
- [ ] Unit tests for each module (≥15 test cases total)
- [ ] Edge case coverage (empty, very long, special chars, unicode)
- [ ] All tests pass with `pytest tests/test_prompts.py -v`

### Notebook
- [ ] Self-contained Colab-compatible notebook
- [ ] Demonstrates all components with example interactions
- [ ] Includes evaluation metrics and visualizations