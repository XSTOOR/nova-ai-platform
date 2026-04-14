"""
NOVA AI Platform — LangGraph Orchestration
==============================================
Defines the multi-agent graph that routes customer messages through:
  Triage → Support/Recommendation → Escalation Check → Response

Uses LangGraph StateGraph for deterministic routing with conditional edges.
"""

import logging
import time
from typing import Optional

from langgraph.graph import StateGraph, END

from multi_agent.state import NOVAState, create_initial_state
from multi_agent.agents.triage_agent import TriageAgent
from multi_agent.agents.support_agent import SupportAgent
from multi_agent.agents.recommendation_agent import RecommendationAgent
from multi_agent.agents.escalation_agent import EscalationAgent
from fine_tuning.inference import NOVAInference

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Response Builder
# ──────────────────────────────────────────────────────────────────────

def _build_response(state: NOVAState) -> str:
    """
    Build a NOVA-branded response from tool results or RAG context.
    Uses Task 4 brand voice inference.
    """
    inference = NOVAInference(use_template=True)
    intent = state.get("intent", "general_support")
    name = state.get("customer_name", "there")
    tool_results = state.get("tool_results", {})
    rag_context = state.get("rag_context", [])
    message = state.get("current_message", "")

    # If we already have a response (e.g., injection blocked), keep it
    if state.get("response"):
        return state["response"]

    # Build response based on intent + tool results
    if intent == "order_status" and "order_lookup" in tool_results:
        result = tool_results["order_lookup"]
        if result.get("found"):
            order = result.get("order", {})
            return (
                f"Hey {name}! 📦 I found your order! "
                f"Order {order.get('order_id', 'N/A')} is currently "
                f"**{order.get('status', 'processing')}**. "
                f"Estimated delivery: {order.get('estimated_delivery', '3-5 business days')}. "
                f"Anything else I can help with? ✨"
            )
        else:
            return (
                f"Hey {name}! I couldn't find that order — could you double-check "
                f"the order number or email? It should look like ORD-2024-XXX. 💕"
            )

    elif intent == "return_refund" and "return_initiate" in tool_results:
        result = tool_results["return_initiate"]
        if result.get("eligible"):
            return (
                f"No worries, {name}! 💕 Your return has been initiated. "
                f"Return ID: {result.get('return_id', 'N/A')}. "
                f"Refund of ${result.get('refund_amount', '0.00')} will be processed "
                f"within {result.get('refund_timeline', '5-7 business days')}. "
                f"We'll send you a prepaid label via email! ✨"
            )
        else:
            return (
                f"Hey {name}, it looks like this item may not be eligible for return "
                f"({result.get('reason', 'outside return window')}). "
                f"Want me to connect you with a team member who can help? 💕"
            )

    elif intent == "loyalty_rewards" and "loyalty_check" in tool_results:
        result = tool_results["loyalty_check"]
        if result.get("found"):
            return (
                f"Hey {name}! ✨ Here are your NOVA Rewards details:\n"
                f"  • Tier: **{result.get('tier', 'Bronze')}**\n"
                f"  • Points: **{result.get('points', 0)}**\n"
                f"  • Redeemable: **${result.get('redeemable_value', 0):.2f}**\n"
                f"You're doing amazing! Keep shopping to unlock more perks! 💕"
            )
        else:
            return (
                f"Hey {name}! I couldn't find a loyalty account for that email. "
                f"Want to sign up? It's free and you get 100 points just for joining! ✨"
            )

    elif intent == "product_inquiry" and rag_context:
        # Build response from RAG results
        top_products = []
        for doc in rag_context[:3]:
            if doc.get("type") == "product":
                top_products.append(
                    f"**{doc.get('name', 'Product')}** "
                    f"(${doc.get('price', 'N/A')}) — {doc.get('description', '')[:80]}"
                )
            elif doc.get("type") == "faq":
                top_products.append(f"💡 {doc.get('question', '')}: {doc.get('answer', '')[:80]}")

        if top_products:
            product_list = "\n  • ".join(top_products)
            return (
                f"Hey {name}! ✨ Great question! Here's what I found:\n"
                f"  • {product_list}\n"
                f"Want more details on any of these? I'm here to help! 💕"
            )

    # Fallback — use inference template
    inf_result = inference.generate(message, customer_name=name)
    return inf_result.output_text


def _compute_brand_score(response: str) -> float:
    """Quick brand voice score computation."""
    from fine_tuning.train import compute_brand_voice_score
    return compute_brand_voice_score(response)["brand_voice_score"]


# ──────────────────────────────────────────────────────────────────────
# Graph Nodes
# ──────────────────────────────────────────────────────────────────────

def triage_node(state: NOVAState) -> NOVAState:
    """Triage: injection defense + intent classification + routing."""
    agent = TriageAgent()
    return agent.process(state)


def support_node(state: NOVAState) -> NOVAState:
    """Support: MCP tool orchestration."""
    agent = SupportAgent()
    return agent.process(state)


def recommendation_node(state: NOVAState) -> NOVAState:
    """Recommendation: RAG pipeline retrieval."""
    agent = RecommendationAgent()
    return agent.process(state)


def escalation_check_node(state: NOVAState) -> NOVAState:
    """Check if escalation is needed after agent processing."""
    agent = EscalationAgent()
    return agent.check_escalation(state)


def escalation_create_node(state: NOVAState) -> NOVAState:
    """Create escalation ticket and generate response."""
    agent = EscalationAgent()
    state = agent.create_ticket(state)
    return agent.generate_escalation_response(state)


def respond_node(state: NOVAState) -> NOVAState:
    """Build final NOVA-branded response with brand voice scoring."""
    start = time.perf_counter()

    response = _build_response(state)
    state["response"] = response
    state["brand_voice_score"] = _compute_brand_score(response)

    # Update conversation history
    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": response})
    state["messages"] = messages

    state["audit_trail"] = state.get("audit_trail", []) + [{
        "agent": "responder",
        "step": "respond",
        "action": "response_generated",
        "brand_voice_score": state["brand_voice_score"],
        "response_length": len(response),
        "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        "timestamp": time.time(),
    }]

    return state


# ──────────────────────────────────────────────────────────────────────
# Routing Functions
# ──────────────────────────────────────────────────────────────────────

def route_after_triage(state: NOVAState) -> str:
    """Route after triage: block, support, recommendation, or escalation."""
    route = state.get("route", "support")
    if route == "block":
        return "respond"
    if route == "escalation":
        return "escalation_check"
    return route


def route_after_agent(state: NOVAState) -> str:
    """Route after support/recommendation: escalation check or respond."""
    # Always check escalation after agent processing
    return "escalation_check"


def route_after_escalation_check(state: NOVAState) -> str:
    """Route after escalation check: escalate or respond."""
    if state.get("escalation_needed", False):
        return "escalation_create"
    return "respond"


# ──────────────────────────────────────────────────────────────────────
# Graph Builder
# ──────────────────────────────────────────────────────────────────────

class NOVAAgentGraph:
    """
    Complete LangGraph multi-agent orchestration for NOVA support.

    Usage:
        graph = NOVAAgentGraph()
        result = graph.run("Where is my order ORD-2024-001?")
        print(result["response"])
    """

    def __init__(self):
        self._graph = self._build_graph()
        self._compiled = self._graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph."""
        graph = StateGraph(NOVAState)

        # Add nodes
        graph.add_node("triage", triage_node)
        graph.add_node("support", support_node)
        graph.add_node("recommendation", recommendation_node)
        graph.add_node("escalation_check", escalation_check_node)
        graph.add_node("escalation_create", escalation_create_node)
        graph.add_node("respond", respond_node)

        # Set entry point
        graph.set_entry_point("triage")

        # Conditional routing after triage
        graph.add_conditional_edges(
            "triage",
            route_after_triage,
            {
                "support": "support",
                "recommendation": "recommendation",
                "escalation_check": "escalation_check",
                "respond": "respond",
            },
        )

        # After support → escalation check
        graph.add_edge("support", "escalation_check")

        # After recommendation → escalation check
        graph.add_edge("recommendation", "escalation_check")

        # After escalation check → escalate or respond
        graph.add_conditional_edges(
            "escalation_check",
            route_after_escalation_check,
            {
                "escalation_create": "escalation_create",
                "respond": "respond",
            },
        )

        # After escalation creation → respond
        graph.add_edge("escalation_create", "respond")

        # Respond → END
        graph.add_edge("respond", END)

        return graph

    def run(
        self,
        message: str,
        customer_id: str = "guest",
        customer_name: str = "Friend",
        session_id: str = "",
    ) -> dict:
        """
        Run a message through the full agent graph.

        Args:
            message: Customer message.
            customer_id: Customer identifier (email or ID).
            customer_name: Customer name for personalization.
            session_id: Optional session ID.

        Returns:
            Dict with response, audit_trail, and all state fields.
        """
        state = create_initial_state(
            message=message,
            customer_id=customer_id,
            customer_name=customer_name,
            session_id=session_id,
        )

        result = self._compiled.invoke(state)
        return dict(result)

    def run_conversation(
        self,
        messages: list[dict],
        customer_id: str = "guest",
        customer_name: str = "Friend",
    ) -> list[dict]:
        """
        Run a multi-turn conversation through the graph.

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}.
            customer_id: Customer identifier.
            customer_name: Customer name.

        Returns:
            List of results for each user message.
        """
        results = []
        for msg in messages:
            if msg["role"] == "user":
                result = self.run(
                    message=msg["content"],
                    customer_id=customer_id,
                    customer_name=customer_name,
                )
                results.append(result)
        return results

    def get_mermaid(self) -> str:
        """Export the graph as a Mermaid diagram string."""
        return (
            "graph TD\n"
            "    START([Customer Message]) --> Triage\n"
            "    \n"
            "    Triage -->|Injection Blocked| Respond\n"
            "    Triage -->|order_status, return_refund, loyalty| Support\n"
            "    Triage -->|product_inquiry| Recommendation\n"
            "    Triage -->|Low Confidence| EscalationCheck\n"
            "    \n"
            "    Support --> EscalationCheck\n"
            "    Recommendation --> EscalationCheck\n"
            "    \n"
            "    EscalationCheck -->|Escalation Needed| EscalationCreate\n"
            "    EscalationCheck -->|No Escalation| Respond\n"
            "    \n"
            "    EscalationCreate --> Respond\n"
            "    Respond --> END([Customer Response])\n"
            "    \n"
            "    classDef triage fill:#FF6B6B,stroke:#C0392B,color:#fff\n"
            "    classDef support fill:#4ECDC4,stroke:#16A085,color:#fff\n"
            "    classDef recommend fill:#45B7D1,stroke:#2980B9,color:#fff\n"
            "    classDef escalate fill:#F9CA24,stroke:#F39C12,color:#000\n"
            "    classDef respond fill:#6C5CE7,stroke:#5B2C6F,color:#fff\n"
            "    \n"
            "    class Triage triage\n"
            "    class Support support\n"
            "    class Recommendation recommend\n"
            "    class EscalationCheck,EscalationCreate escalate\n"
            "    class Respond respond\n"
        )

    def visualize(self, output_path: str = "nova_agent_graph.png") -> str:
        """
        Export graph visualization. Returns mermaid string (PNG needs graphviz).
        Falls back to mermaid if graphviz not available.
        """
        try:
            # Try to get the graph's mermaid representation
            mermaid_str = self.get_mermaid()

            # Save mermaid to file
            mmd_path = output_path.replace(".png", ".mmd")
            with open(mmd_path, "w") as f:
                f.write(mermaid_str)

            # Try PNG generation via graphviz
            try:
                from PIL import Image
                import subprocess
                # Use mermaid-cli if available
                result = subprocess.run(
                    ["mmdc", "-i", mmd_path, "-o", output_path],
                    capture_output=True, timeout=10,
                )
                if result.returncode == 0:
                    return output_path
            except (FileNotFoundError, ImportError, subprocess.TimeoutExpired):
                pass

            # Try graphviz approach from compiled graph
            try:
                img_bytes = self._compiled.get_graph().draw_mermaid_png()
                with open(output_path, "wb") as f:
                    f.write(img_bytes)
                return output_path
            except Exception:
                pass

            return mmd_path  # Return mermaid file as fallback

        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
            return ""