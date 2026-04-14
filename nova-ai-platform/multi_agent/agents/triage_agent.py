"""
NOVA AI Platform — Triage Agent
==================================
First point of contact for every customer message.

Responsibilities:
  1. Defend against prompt injection attacks (Task 1: InjectionDefender)
  2. Classify customer intent (Task 1: IntentClassifier)
  3. Route to the appropriate downstream agent
"""

import asyncio
import logging
import time
from typing import Optional

from multi_agent.state import NOVAState
from prompts.injection_defender import InjectionDefender
from prompts.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)


class TriageAgent:
    """
    Triage agent — classifies intent and blocks injection attacks.

    Uses:
      - Task 1: InjectionDefender (16 regex patterns + scoring)
      - Task 1: IntentClassifier (keyword + LLM fallback)
    """

    def __init__(self):
        self._defender = InjectionDefender()
        self._classifier = IntentClassifier()

    def process(self, state: NOVAState) -> NOVAState:
        """
        Process the current message through triage.

        Steps:
          1. Check for injection attacks
          2. If blocked → set route to "block"
          3. If safe → classify intent
          4. Route based on intent
        """
        start = time.perf_counter()
        message = state.get("current_message", "")
        audit_entry = {
            "agent": "triage",
            "step": "process",
            "timestamp": time.time(),
        }

        # ── Step 1: Injection Defense ──────────────────────────────
        defense_result = self._defender.check(message)
        threat_level = defense_result.threat_level.value
        threat_score = defense_result.score

        state["injection_threat_level"] = threat_level
        state["injection_score"] = threat_score

        if threat_level == "blocked":
            state["injection_blocked"] = True
            state["route"] = "block"
            state["response"] = (
                "I appreciate your message! However, I'm designed to help "
                "with NOVA orders, products, and support. How can I assist you today? 💕"
            )
            state["intent"] = "blocked"
            state["confidence"] = 1.0

            audit_entry.update({
                "action": "injection_blocked",
                "threat_level": threat_level,
                "threat_score": threat_score,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            })
            state["audit_trail"] = state.get("audit_trail", []) + [audit_entry]
            return state

        # ── Step 2: Intent Classification ──────────────────────────
        try:
            loop = asyncio.get_running_loop()
            # We're inside an async context — use sync fallback
            intent_result = self._classifier.classify_sync(message)
        except RuntimeError:
            intent_result = self._classifier.classify_sync(message)

        state["intent"] = intent_result.intent
        state["confidence"] = intent_result.confidence

        # ── Step 3: Route Decision ─────────────────────────────────
        route = self._determine_route(intent_result.intent, intent_result.confidence)
        state["route"] = route

        audit_entry.update({
            "action": "classified",
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "route": route,
            "threat_level": threat_level,
            "threat_score": threat_score,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        })
        state["audit_trail"] = state.get("audit_trail", []) + [audit_entry]

        return state

    def _determine_route(self, intent: str, confidence: float) -> str:
        """Determine which downstream agent to route to."""
        # Low confidence → escalate
        if confidence < 0.3:
            return "escalation"

        intent_to_route = {
            "order_status": "support",
            "return_refund": "support",
            "loyalty_rewards": "support",
            "general_support": "support",
            "product_inquiry": "recommendation",
        }

        return intent_to_route.get(intent, "support")