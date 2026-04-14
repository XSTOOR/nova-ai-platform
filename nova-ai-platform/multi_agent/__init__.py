"""
NOVA AI Platform — Multi-Agent Module
========================================
LangGraph-based multi-agent orchestration for NOVA AI support.

Components:
  - NOVAAgentGraph: Main graph orchestrator
  - NOVAState: Shared state schema
  - TriageAgent: Intent classification + injection defense
  - SupportAgent: MCP tool orchestration
  - RecommendationAgent: RAG-powered product search
  - EscalationAgent: Human-in-the-loop escalation
"""

from .graph import NOVAAgentGraph
from .state import NOVAState, create_initial_state
from .agents import TriageAgent, SupportAgent, RecommendationAgent, EscalationAgent

__all__ = [
    "NOVAAgentGraph",
    "NOVAState",
    "create_initial_state",
    "TriageAgent",
    "SupportAgent",
    "RecommendationAgent",
    "EscalationAgent",
]