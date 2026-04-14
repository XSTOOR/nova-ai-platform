"""
NOVA AI Platform — Escalation Logic
=====================================
Determines when to escalate customer conversations to human agents.

Escalation triggers:
  1. LOW CONFIDENCE: Intent classifier returns confidence < threshold
  2. SENSITIVE TOPIC: Message contains sensitive topic keywords
  3. EXPLICIT REQUEST: Customer directly asks for a human
  4. COMPOUND ISSUE: Multiple intents detected in one message
  5. NEGATIVE SENTIMENT: High frustration/anger indicators

All escalation decisions are logged with full context for audit trails.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from config.settings import Config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Escalation Data Models
# ──────────────────────────────────────────────────────────────────────

class EscalationPriority(str, Enum):
    """Priority levels for escalated tickets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class EscalationReason(str, Enum):
    """Reasons for escalation."""
    LOW_CONFIDENCE = "low_confidence"
    SENSITIVE_TOPIC = "sensitive_topic"
    CUSTOMER_REQUEST = "customer_request"
    COMPOUND_INTENT = "compound_intent"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    FAILED_RESOLUTION = "failed_resolution"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class EscalationDecision:
    """Result of an escalation evaluation."""
    should_escalate: bool
    reason: Optional[EscalationReason]
    priority: EscalationPriority
    confidence: float
    explanation: str
    triggered_by: list[str]  # What triggered the escalation
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "should_escalate": self.should_escalate,
            "reason": self.reason.value if self.reason else None,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "triggered_by": self.triggered_by,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        esc = "ESCALATE" if self.should_escalate else "HANDLE"
        return f"EscalationDecision({esc}, reason={self.reason}, priority={self.priority})"


# ──────────────────────────────────────────────────────────────────────
# Negative Sentiment Indicators
# ──────────────────────────────────────────────────────────────────────

NEGATIVE_SENTIMENT_PATTERNS: list[tuple[str, float]] = [
    # (pattern, severity_score 0-1)
    (r"\b(?:unacceptable|ridiculous|absolutely not)\b", 0.9),
    (r"\b(?:furious|outraged|disgusted|appalled)\b", 0.95),
    (r"\b(?:terrible|horrible|worst|awful|pathetic)\b", 0.85),
    (r"\b(?:angry|pissed|mad|upset)\b", 0.7),
    (r"\b(?:disappointed|frustrated|annoyed)\b", 0.5),
    (r"\b(?:never again|never shopping|done with|finished with)\b", 0.9),
    (r"\b(?:waste of money|scam|rip.?off|stolen)\b", 0.85),
    (r"\b(?:bad service|poor quality|not acceptable)\b", 0.7),
    (r"!{3,}", 0.6),  # Multiple exclamation marks
    (r"(?:ALL CAPS.{10,})", 0.5),  # All-caps text (shouting)
]


# ──────────────────────────────────────────────────────────────────────
# Escalation Manager
# ──────────────────────────────────────────────────────────────────────

class EscalationManager:
    """
    Evaluates whether a customer conversation should be escalated to a human agent.

    Uses configurable thresholds from config/settings.py.

    Usage:
        manager = EscalationManager()
        decision = manager.evaluate(
            message="I want to speak to a manager right now!",
            intent="return_refund",
            confidence=0.85,
        )
        if decision.should_escalate:
            # Route to human agent
    """

    def __init__(
        self,
        confidence_threshold_low: Optional[float] = None,
        confidence_threshold_high: Optional[float] = None,
    ):
        """
        Initialize the EscalationManager.

        Args:
            confidence_threshold_low: Below this → always escalate. Default from Config.
            confidence_threshold_high: Below this → escalate if other factors present.
        """
        self._threshold_low = confidence_threshold_low or Config.escalation.confidence_threshold_low
        self._threshold_high = confidence_threshold_high or Config.escalation.confidence_threshold_high
        self._escalation_keywords = Config.escalation.auto_escalation_keywords
        self._sensitive_topics = Config.escalation.sensitive_topics

    def evaluate(
        self,
        message: str,
        intent: str,
        confidence: float,
        previous_turns: int = 0,
        tool_results: Optional[dict] = None,
    ) -> EscalationDecision:
        """
        Evaluate whether a conversation should be escalated.

        Args:
            message: The customer's message.
            intent: The classified intent category.
            confidence: The intent classification confidence (0.0–1.0).
            previous_turns: Number of previous conversation turns.
            tool_results: Results from any backend tool calls.

        Returns:
            EscalationDecision with escalation recommendation.
        """
        triggers: list[str] = []
        reasons: list[EscalationReason] = []
        max_priority = EscalationPriority.LOW

        # ── Check 1: Low confidence ──────────────────────────────
        if confidence < self._threshold_low:
            triggers.append(f"Very low confidence: {confidence:.2f}")
            reasons.append(EscalationReason.LOW_CONFIDENCE)
            max_priority = EscalationPriority.MEDIUM

        elif confidence < self._threshold_high:
            triggers.append(f"Moderate confidence: {confidence:.2f}")
            reasons.append(EscalationReason.LOW_CONFIDENCE)

        # ── Check 2: Sensitive topics ────────────────────────────
        sensitive_hits = self._check_sensitive_topics(message)
        if sensitive_hits:
            triggers.append(f"Sensitive topic: {', '.join(sensitive_hits)}")
            reasons.append(EscalationReason.SENSITIVE_TOPIC)
            max_priority = EscalationPriority.HIGH

        # ── Check 3: Explicit human request ──────────────────────
        human_request = self._check_human_request(message)
        if human_request:
            triggers.append(f"Customer requested human: '{human_request}'")
            reasons.append(EscalationReason.CUSTOMER_REQUEST)
            max_priority = max(max_priority, EscalationPriority.MEDIUM,
                              key=lambda p: list(EscalationPriority).index(p))

        # ── Check 4: Negative sentiment ──────────────────────────
        sentiment_score = self._check_negative_sentiment(message)
        if sentiment_score >= 0.7:
            triggers.append(f"High negative sentiment: {sentiment_score:.2f}")
            reasons.append(EscalationReason.NEGATIVE_SENTIMENT)
            max_priority = max(max_priority, EscalationPriority.HIGH,
                              key=lambda p: list(EscalationPriority).index(p))
        elif sentiment_score >= 0.5:
            triggers.append(f"Moderate negative sentiment: {sentiment_score:.2f}")
            reasons.append(EscalationReason.NEGATIVE_SENTIMENT)

        # ── Check 5: Repeated failures ───────────────────────────
        if previous_turns >= 5:
            triggers.append(f"Extended conversation: {previous_turns} turns")
            reasons.append(EscalationReason.FAILED_RESOLUTION)
            max_priority = max(max_priority, EscalationPriority.MEDIUM,
                              key=lambda p: list(EscalationPriority).index(p))

        # ── Check 6: Tool execution failure ──────────────────────
        if tool_results and not tool_results.get("success", True):
            triggers.append(f"Tool execution failure: {tool_results.get('error', 'unknown')}")
            reasons.append(EscalationReason.FAILED_RESOLUTION)

        # ── Build decision ───────────────────────────────────────
        should_escalate = len(reasons) > 0

        # Determine primary reason
        primary_reason = reasons[0] if reasons else None

        # Build explanation
        if should_escalate:
            explanation = f"Escalation triggered by: {'; '.join(triggers[:3])}"
        else:
            explanation = "No escalation triggers detected. Confidence is sufficient and no sensitive topics found."

        decision = EscalationDecision(
            should_escalate=should_escalate,
            reason=primary_reason,
            priority=max_priority if should_escalate else EscalationPriority.LOW,
            confidence=confidence,
            explanation=explanation,
            triggered_by=triggers,
        )

        # Log the decision
        if should_escalate:
            logger.warning(
                f"ESCALATION DECISION: reason={primary_reason}, "
                f"priority={max_priority.value}, intent={intent}, "
                f"triggers={triggers}"
            )
        else:
            logger.info(
                f"No escalation needed: intent={intent}, confidence={confidence:.2f}"
            )

        return decision

    def _check_sensitive_topics(self, message: str) -> list[str]:
        """Check for sensitive topic keywords."""
        message_lower = message.lower()
        return [topic for topic in self._sensitive_topics if topic in message_lower]

    def _check_human_request(self, message: str) -> Optional[str]:
        """Check if customer explicitly requests a human agent."""
        message_lower = message.lower()
        human_request_patterns = [
            "speak to a human",
            "talk to a person",
            "real person",
            "human agent",
            "live agent",
            "customer service rep",
            "manager",
            "supervisor",
            "someone real",
            "not a bot",
            "not a chatbot",
        ]
        for pattern in human_request_patterns:
            if pattern in message_lower:
                return pattern
        return None

    def _check_negative_sentiment(self, message: str) -> float:
        """
        Score negative sentiment intensity (0.0–1.0).

        Uses regex patterns with severity weights.
        """
        import re
        message_lower = message.lower()
        max_score = 0.0

        for pattern, severity in NEGATIVE_SENTIMENT_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                max_score = max(max_score, severity)

        return max_score

    def get_escalation_prompt_context(self, decision: EscalationDecision) -> str:
        """
        Generate context for the escalation agent's prompt.

        Args:
            decision: The escalation decision.

        Returns:
            Formatted context string for the escalation agent.
        """
        parts = [
            f"ESCALATION REQUIRED",
            f"Priority: {decision.priority.value.upper()}",
            f"Reason: {decision.reason.value if decision.reason else 'Unknown'}",
            f"Confidence Score: {decision.confidence:.2f}",
            f"Triggered By: {', '.join(decision.triggered_by)}",
            f"Timestamp: {decision.timestamp}",
        ]
        return "\n".join(parts)