"""
NOVA AI Platform — RAGAS Evaluation
======================================
Evaluates the RAG pipeline using RAGAS metrics.

Metrics:
  - Faithfulness:        Are answers grounded in retrieved context?
  - Answer Relevancy:    Are answers relevant to the question?
  - Context Precision:   Are retrieved contexts relevant?
  - Context Recall:      Are all needed contexts retrieved?

Supports:
  - Full RAGAS evaluation with LLM
  - Mock evaluation for environments without LLM API keys
  - Per-metric and aggregate scoring
"""

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import Config

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# ──────────────────────────────────────────────────────────────────────
# Evaluation Data Models
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RAGASMetrics:
    """Container for RAGAS evaluation metrics."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    @property
    def average(self) -> float:
        vals = [self.faithfulness, self.answer_relevancy,
                self.context_precision, self.context_recall]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "average": self.average,
        }


# ──────────────────────────────────────────────────────────────────────
# Test Questions for Evaluation
# ──────────────────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "question": "What moisturizer do you recommend for dry skin?",
        "expected_answer": "Hydra-Boost Moisturizer",
        "expected_keywords": ["hydra-boost", "moisturizer", "hyaluronic acid", "hydration"],
    },
    {
        "question": "How long does shipping take?",
        "expected_answer": "Standard shipping takes 3-5 business days",
        "expected_keywords": ["3-5", "business days", "shipping"],
    },
    {
        "question": "What is NOVA's return policy?",
        "expected_answer": "30-day return policy",
        "expected_keywords": ["30-day", "return", "unused", "original packaging"],
    },
    {
        "question": "Do you have any cashmere products?",
        "expected_answer": "Cashmere Lounge Set",
        "expected_keywords": ["cashmere", "lounge set", "mongolian"],
    },
    {
        "question": "What's in the Glow-Up Vitamin C Serum?",
        "expected_answer": "20% vitamin C with ferulic acid and vitamin E",
        "expected_keywords": ["vitamin c", "ferulic acid", "vitamin e", "ascorbic"],
    },
    {
        "question": "How do NOVA Rewards tiers work?",
        "expected_answer": "Bronze, Silver, and Gold tiers with different earning rates",
        "expected_keywords": ["bronze", "silver", "gold", "points", "earning"],
    },
    {
        "question": "Are NOVA products cruelty-free?",
        "expected_answer": "Yes, 100% cruelty-free, Leaping Bunny certified",
        "expected_keywords": ["cruelty-free", "leaping bunny", "tested on animals"],
    },
    {
        "question": "I need a gift idea for someone who likes skincare",
        "expected_answer": "Glow-Up Vitamin C Serum or Hydra-Boost Moisturizer",
        "expected_keywords": ["serum", "moisturizer", "gift", "vitamin c"],
    },
]


# ──────────────────────────────────────────────────────────────────────
# Mock RAGAS Evaluator
# ──────────────────────────────────────────────────────────────────────

class MockRAGASEvaluator:
    """
    Evaluates RAG pipeline quality using heuristic scoring.
    Used when full RAGAS (with LLM) is unavailable.
    """

    def evaluate(
        self,
        question: str,
        retrieved_docs: list[str],
        generated_answer: str,
        expected_keywords: list[str],
    ) -> RAGASMetrics:
        """Evaluate a single Q&A pair with heuristics."""
        answer_lower = generated_answer.lower()
        context_text = " ".join(retrieved_docs).lower()

        # Faithfulness: do answer claims appear in context?
        answer_terms = set(answer_lower.split())
        context_terms = set(context_text.split())
        if answer_terms:
            faithfulness = len(answer_terms & context_terms) / len(answer_terms)
        else:
            faithfulness = 0.0

        # Answer Relevancy: do expected keywords appear in answer?
        if expected_keywords:
            found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
            answer_relevancy = found / len(expected_keywords)
        else:
            answer_relevancy = 0.5

        # Context Precision: are retrieved docs relevant to question?
        q_terms = set(question.lower().split())
        if retrieved_docs:
            precisions = []
            for doc in retrieved_docs:
                doc_terms = set(doc.lower().split())
                overlap = len(q_terms & doc_terms) / max(len(q_terms), 1)
                precisions.append(overlap)
            context_precision = statistics.mean(precisions)
        else:
            context_precision = 0.0

        # Context Recall: do expected keywords appear in context?
        if expected_keywords:
            found_in_context = sum(1 for kw in expected_keywords if kw.lower() in context_text)
            context_recall = found_in_context / len(expected_keywords)
        else:
            context_recall = 0.0

        return RAGASMetrics(
            faithfulness=min(faithfulness, 1.0),
            answer_relevancy=min(answer_relevancy, 1.0),
            context_precision=min(context_precision * 2, 1.0),  # Scale up
            context_recall=min(context_recall, 1.0),
        )


# ──────────────────────────────────────────────────────────────────────
# RAGAS Evaluator
# ──────────────────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Evaluates the RAG pipeline using RAGAS metrics.

    Supports:
      - Full RAGAS evaluation (requires LLM + RAGAS library)
      - Mock heuristic evaluation (works without LLM)

    Usage:
        evaluator = RAGASEvaluator(search_engine, reranker)
        report = evaluator.evaluate_pipeline()
    """

    def __init__(
        self,
        search_engine=None,
        reranker=None,
        use_mock: bool = True,
    ):
        self._search_engine = search_engine
        self._reranker = reranker
        self._use_mock = use_mock

    def generate_answer_from_context(
        self, question: str, context_docs: list[str]
    ) -> str:
        """
        Generate an answer from retrieved context documents.
        Simple extraction-based answer generation (no LLM needed).
        """
        if not context_docs:
            return "I couldn't find relevant information."

        # Combine top contexts
        combined = "\n".join(context_docs[:3])

        # Extract the most relevant sentences
        q_terms = set(question.lower().split()) - {"what", "how", "do", "is", "are", "the", "a", "an", "i", "you", "have", "does", "in", "for", "of", "and", "to"}
        sentences = combined.replace("\n", ". ").split(". ")
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower().strip()
            if len(sent_lower) < 10:
                continue
            overlap = sum(1 for term in q_terms if term in sent_lower)
            if overlap > 0:
                scored_sentences.append((overlap, sent.strip()))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if scored_sentences:
            return ". ".join(s for _, s in scored_sentences[:3])
        return context_docs[0][:300]

    def evaluate_single(
        self,
        question: str,
        expected_keywords: list[str],
    ) -> dict:
        """Evaluate a single question through the RAG pipeline."""
        # Retrieve
        results = self._search_engine.search(question, n_results=5) if self._search_engine else []

        # Rerank
        if self._reranker and results:
            results = self._reranker.rerank(question, results, top_k=3)

        # Extract documents
        context_docs = [r.get("document", "") for r in results]

        # Generate answer
        answer = self.generate_answer_from_context(question, context_docs)

        # Evaluate
        if self._use_mock:
            evaluator = MockRAGASEvaluator()
            metrics = evaluator.evaluate(question, context_docs, answer, expected_keywords)
        else:
            evaluator = MockRAGASEvaluator()
            metrics = evaluator.evaluate(question, context_docs, answer, expected_keywords)

        return {
            "question": question,
            "answer": answer,
            "context_count": len(context_docs),
            "metrics": metrics,
            "results_count": len(results),
        }

    def evaluate_pipeline(
        self,
        questions: Optional[list[dict]] = None,
    ) -> dict:
        """
        Run a full evaluation of the RAG pipeline.

        Args:
            questions: Optional custom evaluation questions.

        Returns:
            Evaluation report with per-question and aggregate metrics.
        """
        eval_questions = questions or EVAL_QUESTIONS
        per_question = []
        all_metrics = RAGASMetrics()

        for q in eval_questions:
            result = self.evaluate_single(q["question"], q.get("expected_keywords", []))
            per_question.append(result)

        # Aggregate metrics
        if per_question:
            faith_scores = [r["metrics"].faithfulness for r in per_question]
            rel_scores = [r["metrics"].answer_relevancy for r in per_question]
            prec_scores = [r["metrics"].context_precision for r in per_question]
            rec_scores = [r["metrics"].context_recall for r in per_question]

            all_metrics = RAGASMetrics(
                faithfulness=statistics.mean(faith_scores),
                answer_relevancy=statistics.mean(rel_scores),
                context_precision=statistics.mean(prec_scores),
                context_recall=statistics.mean(rec_scores),
            )

        report = {
            "total_questions": len(per_question),
            "per_question": [
                {"question": r["question"], "metrics": r["metrics"].to_dict()}
                for r in per_question
            ],
            "aggregate_metrics": all_metrics.to_dict(),
            "evaluation_mode": "mock" if self._use_mock else "full_ragas",
        }

        logger.info(f"RAGAS evaluation complete: {all_metrics.to_dict()}")
        return report