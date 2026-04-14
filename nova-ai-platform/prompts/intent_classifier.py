"""
NOVA AI Platform — Intent Classifier
======================================
Classifies customer messages into one of 5 intent categories using
an LLM with structured JSON output. Returns category + confidence score.

Intent Categories:
  - order_status:     Order tracking, delivery, shipment inquiries
  - product_inquiry:  Product details, recommendations, availability
  - return_refund:    Returns, exchanges, refunds, damaged items
  - loyalty_rewards:  Points, tiers, reward redemptions, perks
  - general_support:  Shipping policies, account issues, FAQs, misc
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import Config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Intent Classification Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class IntentResult:
    """Structured result from intent classification."""
    intent: str
    confidence: float
    reasoning: str
    keywords_detected: list[str]
    is_ambiguous: bool

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "keywords_detected": self.keywords_detected,
            "is_ambiguous": self.is_ambiguous,
        }

    def __repr__(self) -> str:
        amb = " ⚠️ AMBIGUOUS" if self.is_ambiguous else ""
        return f"IntentResult({self.intent}, confidence={self.confidence:.2f}{amb})"


# ──────────────────────────────────────────────────────────────────────
# Keyword-Based Pre-Screening (Fast Path)
# ──────────────────────────────────────────────────────────────────────

INTENT_KEYWORDS: dict[str, list[str]] = {
    "order_status": [
        "order", "track", "tracking", "shipment", "shipped", "delivery",
        "delivered", "where is", "when will", "estimated delivery",
        "order status", "package", "arrive", "shipping",
        "on its way", "in transit", "out for delivery",
        "haven't received", "missing order",
    ],
    "product_inquiry": [
        "recommend", "suggestion", "what should", "looking for",
        "product", "ingredient", "shade", "size", "color",
        "available", "in stock", "material", "what do you have",
        "best for", "suitable for", "skin type", "gift",
        "new arrivals", "collection", "bestseller",
    ],
    "return_refund": [
        "return", "refund", "exchange", "send back", "money back",
        "wrong size", "doesn't fit", "damaged", "broken", "defective",
        "allergic", "reaction", "wrong item", "not what i ordered",
        "cancel order", "want my money", "disappointed",
    ],
    "loyalty_rewards": [
        "points", "rewards", "loyalty", "tier", "membership",
        "earn", "redeem", "perks", "vip", "bronze", "silver", "gold",
        "birthday", "discount code", "coupon", "promotion",
        "exclusive", "early access",
    ],
    "general_support": [
        "shipping", "account", "password", "login", "email",
        "policy", "faq", "help", "question", "how do i",
        "store", "location", "hours", "contact", "complaint",
        "feedback", "suggestion", "website", "app",
    ],
}

# Escalation keywords (detected alongside intent)
ESCALATION_KEYWORDS = [
    "manager", "supervisor", "speak to someone", "real person",
    "human", "complaint", "lawyer", "attorney", "legal",
    "lawsuit", "better business bureau", "unacceptable",
    "furious", "angry", "ridiculous", "worst", "terrible service",
]


def _keyword_pre_screen(message: str) -> tuple[Optional[str], list[str]]:
    """
    Fast keyword-based pre-screening.

    Returns:
        Tuple of (matched_intent_or_None, list_of_matched_keywords).
    """
    message_lower = message.lower()
    scores: dict[str, int] = {}
    matched_keywords: dict[str, list[str]] = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in message_lower:
                scores[intent] = scores.get(intent, 0) + 1
                matched_keywords.setdefault(intent, []).append(kw)

    if not scores:
        return None, []

    best_intent = max(scores, key=scores.get)
    return best_intent, matched_keywords.get(best_intent, [])


# ──────────────────────────────────────────────────────────────────────
# Intent Classifier (LLM-Based)
# ──────────────────────────────────────────────────────────────────────

INTENT_CLASSIFICATION_SYSTEM = """You are an intent classifier for NOVA, a premium D2C fashion & beauty brand's customer support system.

CLASSIFY the customer message into EXACTLY ONE of these categories:
- order_status:     Order tracking, delivery, shipment inquiries
- product_inquiry:  Product details, recommendations, availability
- return_refund:    Returns, exchanges, refunds, damaged items
- loyalty_rewards:  Points, tiers, reward redemptions, perks
- general_support:  Shipping policies, account issues, FAQs, misc

RESPOND WITH ONLY VALID JSON (no markdown, no explanation):
{
  "intent": "one of: order_status, product_inquiry, return_refund, loyalty_rewards, general_support",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation",
  "keywords": ["detected", "keywords"]
}

CONFIDENCE GUIDELINES:
- 0.9+: Very clear intent with explicit keywords
- 0.7-0.9: Clear intent but could have minor ambiguity
- 0.4-0.7: Ambiguous — multiple intents possible
- 0.0-0.4: Very unclear, confused, or nonsensical input

IMPORTANT: Return ONLY the JSON object. No other text."""


class IntentClassifier:
    """
    Classifies customer messages into NOVA's 5 intent categories.

    Uses a two-stage approach:
    1. Fast keyword pre-screening for obvious cases
    2. LLM-based classification for nuanced/ambiguous messages

    Usage:
        classifier = IntentClassifier(llm=my_llm)
        result = await classifier.classify("Where is my order #12345?")
        print(result.intent, result.confidence)
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the intent classifier.

        Args:
            llm: A LangChain chat model. If None, creates one from config.
        """
        self._llm = llm
        self._keyword_only_mode = False

    def _get_llm(self) -> BaseChatModel:
        """Get or create the LLM instance."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    api_key=Config.llm.api_key,
                    base_url=Config.llm.base_url,
                    model=Config.llm.model_name,
                    temperature=0.0,  # Deterministic classification
                    max_tokens=200,
                )
            except Exception as e:
                logger.warning(f"Could not initialize LLM: {e}. Falling back to keyword-only mode.")
                self._keyword_only_mode = True
                return None
        return self._llm

    async def classify(self, message: str) -> IntentResult:
        """
        Classify a customer message into an intent category.

        Args:
            message: The raw customer message.

        Returns:
            IntentResult with intent, confidence, and metadata.
        """
        if not message or not message.strip():
            return IntentResult(
                intent="general_support",
                confidence=0.0,
                reasoning="Empty or whitespace-only input",
                keywords_detected=[],
                is_ambiguous=True,
            )

        # Stage 1: Keyword pre-screening
        kw_intent, kw_keywords = _keyword_pre_screen(message)

        # If very strong keyword match, use it directly (fast path)
        if kw_intent and len(kw_keywords) >= 3:
            logger.info(f"Fast-path classification: {kw_intent} (keywords: {kw_keywords})")
            return IntentResult(
                intent=kw_intent,
                confidence=min(0.7 + len(kw_keywords) * 0.05, 0.95),
                reasoning=f"Strong keyword match: {', '.join(kw_keywords[:5])}",
                keywords_detected=kw_keywords,
                is_ambiguous=False,
            )

        # Stage 2: LLM-based classification
        llm = self._get_llm()
        if self._keyword_only_mode or llm is None:
            return self._keyword_fallback(message, kw_intent, kw_keywords)

        try:
            response = await llm.ainvoke([
                SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM),
                HumanMessage(content=f"Classify this customer message:\n\n\"{message}\""),
            ])

            # Parse JSON response
            content = response.content.strip()
            # Strip markdown code blocks if present
            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            result_data = json.loads(content)

            intent = result_data.get("intent", "general_support")
            confidence = float(result_data.get("confidence", 0.5))
            reasoning = result_data.get("reasoning", "")
            keywords = result_data.get("keywords", [])

            # Validate intent
            valid_intents = Config.intent.categories
            if intent not in valid_intents:
                logger.warning(f"LLM returned invalid intent '{intent}', defaulting to general_support")
                intent = "general_support"
                confidence *= 0.5

            is_ambiguous = confidence < 0.6

            return IntentResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                keywords_detected=keywords,
                is_ambiguous=is_ambiguous,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM classification response: {e}")
            return self._keyword_fallback(message, kw_intent, kw_keywords)
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._keyword_fallback(message, kw_intent, kw_keywords)

    def _keyword_fallback(
        self, message: str,
        kw_intent: Optional[str],
        kw_keywords: list[str],
    ) -> IntentResult:
        """Fallback to keyword-only classification when LLM is unavailable."""
        if kw_intent:
            confidence = min(0.5 + len(kw_keywords) * 0.1, 0.8)
            return IntentResult(
                intent=kw_intent,
                confidence=confidence,
                reasoning=f"Keyword-based classification (LLM unavailable): {', '.join(kw_keywords[:5])}",
                keywords_detected=kw_keywords,
                is_ambiguous=len(kw_keywords) <= 1,
            )

        return IntentResult(
            intent="general_support",
            confidence=0.3,
            reasoning="No strong keyword match (LLM unavailable)",
            keywords_detected=[],
            is_ambiguous=True,
        )

    def classify_sync(self, message: str) -> IntentResult:
        """Synchronous wrapper for classify()."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.classify(message)).result()
        except RuntimeError:
            return asyncio.run(self.classify(message))

    def detect_escalation_keywords(self, message: str) -> list[str]:
        """
        Detect escalation-triggering keywords in a message.

        Returns:
            List of escalation keywords found in the message.
        """
        message_lower = message.lower()
        return [kw for kw in ESCALATION_KEYWORDS if kw in message_lower]