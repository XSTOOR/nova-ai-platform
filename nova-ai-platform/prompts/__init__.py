"""
NOVA AI Platform — Prompts Module
===================================
Complete prompt engineering system for NOVA's AI support brain.

Components:
  - COSTARPromptBuilder: COSTAR framework prompt templates
  - IntentClassifier: 5-category intent classification with confidence
  - EscalationManager: Confidence + keyword escalation logic
  - InjectionDefender: Multi-layer prompt injection defense
"""

from .costar_templates import COSTARPromptBuilder, COSTARPrompt, NOVA_PERSONA
from .intent_classifier import IntentClassifier, IntentResult, ESCALATION_KEYWORDS
from .escalation_logic import (
    EscalationManager,
    EscalationDecision,
    EscalationPriority,
    EscalationReason,
)
from .injection_defender import (
    InjectionDefender,
    InjectionCheckResult,
    ThreatLevel,
    check_injection,
)

__all__ = [
    "COSTARPromptBuilder",
    "COSTARPrompt",
    "NOVA_PERSONA",
    "IntentClassifier",
    "IntentResult",
    "ESCALATION_KEYWORDS",
    "EscalationManager",
    "EscalationDecision",
    "EscalationPriority",
    "EscalationReason",
    "InjectionDefender",
    "InjectionCheckResult",
    "ThreatLevel",
    "check_injection",
]