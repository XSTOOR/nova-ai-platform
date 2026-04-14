"""
NOVA AI Platform — Inference Pipeline
=======================================
Loads the fine-tuned NOVA brand voice model and generates responses.

Supports:
  - Full inference with fine-tuned model (GPU)
  - Template-based fallback inference (CPU, no model)
  - Brand voice evaluation on generated outputs
  - Modular interface for Task 5 integration
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fine_tuning.train import compute_bleu, compute_rouge_l, compute_brand_voice_score

logger = logging.getLogger(__name__)

NOVA_SYSTEM_PROMPT = (
    "You are NOVA's AI Style & Support Concierge. You speak like a stylish "
    "best friend: warm, enthusiastic, helpful, and never robotic. Use emojis "
    "sparingly but effectively. Be proactive with suggestions. Address "
    "customers by name when available. Acknowledge emotions before solving "
    "problems. Keep responses concise (2-4 paragraphs). Never say 'I am an AI'."
)


# ──────────────────────────────────────────────────────────────────────
# Template-based Inference (Fallback)
# ──────────────────────────────────────────────────────────────────────

_RESPONSE_TEMPLATES = {
    "greeting": (
        "Hey there! 💕 Welcome to NOVA support! I'd love to help you out. "
        "What can I do for you today?"
    ),
    "order_status": (
        "I'd be happy to help you track that down! 📦 Could you share your "
        "order number (starts with ORD-) or the email you used at checkout? "
        "Once I have that, I'll give you a full update!"
    ),
    "product_inquiry": (
        "Great question! ✨ Based on what you're looking for, I'd recommend "
        "checking out our bestsellers. Want me to help you find the perfect match?"
    ),
    "return_refund": (
        "No worries at all — let's get that sorted for you! 💕 I can help "
        "process a return or exchange. Could you share your order number so I "
        "can pull up the details?"
    ),
    "loyalty_rewards": (
        "Let's check those points! ✨ I can look that up for you. Could you "
        "share the email address on your NOVA account? Remember, 100 points = $5 off!"
    ),
    "escalation": (
        "Absolutely, I'll connect you with one of our team members right away! 🙋‍♀️ "
        "Before I hand you over, could you briefly let me know what you need help with?"
    ),
    "complaint": (
        "I'm really sorry about that experience — that's not okay, and I want to "
        "make it right. 😔 Could you share more details so I can help resolve this?"
    ),
    "shipping": (
        "Great question! 🚚 We offer free standard shipping on orders over $75. "
        "Standard: 3-5 days, Express: 2-3 days ($9.99), Overnight: $19.99. "
        "NOVA Rewards members get even better perks! ✨"
    ),
    "default": (
        "Hey there! 💕 Thanks for reaching out to NOVA. I'd love to help you "
        "with that! Could you give me a bit more detail so I can assist you better?"
    ),
}


def _detect_category(text: str) -> str:
    """Detect the response category from input text."""
    text_lower = text.lower()
    categories = {
        "greeting": ["hello", "hey", "good morning", "good afternoon", "help me"],
        "order_status": ["order", "track", "shipping", "delivery", "where is"],
        "product_inquiry": ["recommend", "product", "looking for", "suggest", "best for"],
        "return_refund": ["return", "refund", "exchange", "send back"],
        "loyalty_rewards": ["points", "rewards", "loyalty", "tier", "redeem"],
        "escalation": ["manager", "supervisor", "human", "real person"],
        "complaint": ["unacceptable", "terrible", "worst", "angry", "furious"],
        "shipping": ["ship", "delivery", "free shipping", "how long"],
    }
    best_cat = "default"
    best_score = 0
    for cat, keywords in categories.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


# ──────────────────────────────────────────────────────────────────────
# Inference Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Result from model inference."""
    input_text: str
    output_text: str
    category: str
    brand_voice_score: float
    bleu_score: Optional[float] = None
    rouge_l_score: Optional[float] = None
    latency_ms: float = 0.0
    model_type: str = "template"  # "template" or "fine-tuned"

    def to_dict(self) -> dict:
        return {
            "input": self.input_text,
            "output": self.output_text,
            "category": self.category,
            "brand_voice_score": self.brand_voice_score,
            "bleu_score": self.bleu_score,
            "rouge_l_score": self.rouge_l_score,
            "latency_ms": self.latency_ms,
            "model_type": self.model_type,
        }


# ──────────────────────────────────────────────────────────────────────
# NOVA Inference Pipeline
# ──────────────────────────────────────────────────────────────────────

class NOVAInference:
    """
    NOVA brand voice inference pipeline.

    Supports:
      - Template-based inference (no GPU needed, always available)
      - Fine-tuned model inference (requires GPU + saved checkpoint)

    Usage:
        inference = NOVAInference()
        result = inference.generate("Where is my order?")
        print(result.output_text)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        use_template: bool = True,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model_path: Path to fine-tuned LoRA adapter.
            base_model: Base model name for loading.
            use_template: If True, use template-based generation (no GPU).
        """
        self._model_path = model_path
        self._base_model = base_model
        self._use_template = use_template
        self._model = None
        self._tokenizer = None

        if not use_template and model_path:
            self._load_model()

    def _load_model(self):
        """Load the fine-tuned model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            logger.info(f"Loading base model: {self._base_model}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._base_model)

            self._model = AutoModelForCausalLM.from_pretrained(
                self._base_model,
                device_map="auto",
                torch_dtype=torch.float16,
            )

            logger.info(f"Loading LoRA adapter: {self._model_path}")
            self._model = PeftModel.from_pretrained(self._model, self._model_path)
            self._model.eval()

            logger.info("Fine-tuned model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load fine-tuned model: {e}. Falling back to templates.")
            self._use_template = True
            self._model = None
            self._tokenizer = None

    def generate(
        self,
        input_text: str,
        customer_name: Optional[str] = None,
        max_new_tokens: int = 250,
        temperature: float = 0.7,
    ) -> InferenceResult:
        """
        Generate a brand voice response.

        Args:
            input_text: Customer input text.
            customer_name: Optional customer name for personalization.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            InferenceResult with generated text and metrics.
        """
        import time
        start = time.perf_counter()

        category = _detect_category(input_text)

        if self._use_template or self._model is None:
            output = self._template_generate(input_text, category, customer_name)
            model_type = "template"
        else:
            output = self._model_generate(
                input_text, customer_name, max_new_tokens, temperature
            )
            model_type = "fine-tuned"

        latency = (time.perf_counter() - start) * 1000

        # Evaluate brand voice
        bv_score = compute_brand_voice_score(output)["brand_voice_score"]

        return InferenceResult(
            input_text=input_text,
            output_text=output,
            category=category,
            brand_voice_score=bv_score,
            latency_ms=round(latency, 2),
            model_type=model_type,
        )

    def _template_generate(
        self, input_text: str, category: str, customer_name: Optional[str]
    ) -> str:
        """Generate response using templates."""
        template = _RESPONSE_TEMPLATES.get(category, _RESPONSE_TEMPLATES["default"])
        response = template

        # Personalize with name
        if customer_name and "!" in response:
            response = response.replace("!", f", {customer_name}!", 1)

        return response

    def _model_generate(
        self,
        input_text: str,
        customer_name: Optional[str],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generate response using the fine-tuned model."""
        import torch

        name_context = f"Customer name: {customer_name}\n" if customer_name else ""

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{NOVA_SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{name_context}{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        return response

    def compare_models(
        self, input_text: str, expected_output: Optional[str] = None
    ) -> dict:
        """
        Compare template vs fine-tuned model outputs.

        Args:
            input_text: Customer input.
            expected_output: Optional expected output for scoring.

        Returns:
            Dict with both outputs and comparison metrics.
        """
        template_result = self.generate(input_text)
        template_result.model_type = "template"

        if self._model is not None and not self._use_template:
            ft_result = self.generate(input_text)
            ft_result.model_type = "fine-tuned"
        else:
            ft_result = None

        comparison = {
            "input": input_text,
            "template_output": template_result.output_text,
            "template_brand_score": template_result.brand_voice_score,
            "fine_tuned_output": ft_result.output_text if ft_result else "Model not available",
            "fine_tuned_brand_score": ft_result.brand_voice_score if ft_result else None,
        }

        if expected_output:
            comparison["expected"] = expected_output
            comparison["template_bleu"] = compute_bleu(
                expected_output, template_result.output_text
            )
            comparison["template_rouge_l"] = compute_rouge_l(
                expected_output, template_result.output_text
            )
            if ft_result:
                comparison["ft_bleu"] = compute_bleu(
                    expected_output, ft_result.output_text
                )
                comparison["ft_rouge_l"] = compute_rouge_l(
                    expected_output, ft_result.output_text
                )

        return comparison