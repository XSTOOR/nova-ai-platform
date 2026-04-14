"""
NOVA AI Platform — Dataset Preparation
=========================================
Prepares the brand voice fine-tuning dataset from brand_voice_samples.json.

Pipeline:
  1. Load 12 raw brand voice samples
  2. Augment to 100+ examples via paraphrasing and context variations
  3. Format as instruction-response pairs (chat template)
  4. Split into train/validation (80/20)
  5. Export as list of dicts ready for HuggingFace Dataset
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# ──────────────────────────────────────────────────────────────────────
# NOVA System Prompt (for training)
# ──────────────────────────────────────────────────────────────────────

NOVA_SYSTEM_PROMPT = (
    "You are NOVA's AI Style & Support Concierge. You speak like a stylish "
    "best friend: warm, enthusiastic, helpful, and never robotic. Use emojis "
    "sparingly but effectively. Be proactive with suggestions. Address "
    "customers by name when available. Acknowledge emotions before solving "
    "problems. Keep responses concise (2-4 paragraphs). Never say 'I am an AI'."
)

# ──────────────────────────────────────────────────────────────────────
# Augmentation Helpers
# ──────────────────────────────────────────────────────────────────────

_CONTEXT_VARIATIONS = {
    "greeting": [
        "I need some help please.",
        "Can you assist me?",
        "Hey there!",
        "Hi NOVA!",
        "Hello, I'm reaching out for support.",
    ],
    "order_status": [
        "I'm waiting for my package and want to know the status.",
        "Has my order shipped yet?",
        "When will my order arrive?",
        "Can you check on my delivery?",
        "I placed an order and want an update.",
    ],
    "product_recommendation": [
        "What would you suggest for me?",
        "I'm looking for something new to try.",
        "Can you help me find the right product?",
        "I need some recommendations please.",
        "What are your best sellers?",
    ],
    "return_request": [
        "I'd like to send this back.",
        "This doesn't work for me, can I return it?",
        "How do I return an item?",
        "I need a refund.",
        "The product isn't what I expected.",
    ],
    "complaint": [
        "I'm really not happy with this.",
        "This is not okay.",
        "I have a serious issue with my order.",
        "This experience has been frustrating.",
        "I need to escalate this problem.",
    ],
    "loyalty_inquiry": [
        "What benefits do I get as a member?",
        "Tell me about the rewards program.",
        "Am I close to the next tier?",
        "Can I use my points for a discount?",
        "What perks do I have?",
    ],
    "escalation": [
        "I need to talk to someone who can actually help.",
        "Get me a real person please.",
        "I want to speak to your supervisor.",
        "Connect me with a human agent.",
        "I don't want to deal with automated support.",
    ],
    "shipping_inquiry": [
        "What are the shipping options?",
        "How much does shipping cost?",
        "Do you offer free delivery?",
        "Can I get expedited shipping?",
        "How fast can I get my order?",
    ],
    "product_question": [
        "Can you tell me more about this product?",
        "Is this suitable for my needs?",
        "What are the ingredients/materials?",
        "How do I use this correctly?",
        "What makes this product special?",
    ],
}


def _paraphrase_input(text: str) -> list[str]:
    """Generate paraphrased variations of a customer input."""
    variations = [text]

    # Add context variations from the same category
    words = text.lower().split()
    for category, cat_variations in _CONTEXT_VARIATIONS.items():
        cat_words = category.replace("_", " ").split()
        if any(cw in words for cw in cat_words):
            variations.extend(cat_variations[:2])
            break

    # Simple rephrasings
    if text.endswith("?"):
        variations.append(f"Can you tell me {text.lower().rstrip('?')}")
        variations.append(f"I'd like to know {text.lower().rstrip('?')}")
    elif not text.endswith("!"):
        variations.append(f"Hi, {text.lower()}")
        variations.append(f"Hello! {text}")

    return list(set(variations))


def _extract_brand_markers(response: str) -> dict:
    """Extract brand voice markers from a response for evaluation."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+"
    )
    emojis_found = emoji_pattern.findall(response)
    warmth_words = [
        w for w in ["love", "obsessed", "amazing", "perfect", "sweet",
                     "gorgeous", "beautiful", "darling", "hey", "welcome"]
        if w in response.lower()
    ]
    proactive_markers = [
        m for m in ["want me to", "i can help", "let me", "pro tip",
                     "heads up", "would you like", "shall i"]
        if m in response.lower()
    ]
    return {
        "emoji_count": len(emojis_found),
        "warmth_word_count": len(warmth_words),
        "proactive_marker_count": len(proactive_markers),
        "has_follow_up": any(
            q in response
            for q in ["?", "want me to", "would you", "shall i", "can i"]
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Dataset Builder
# ──────────────────────────────────────────────────────────────────────

class BrandVoiceDatasetBuilder:
    """
    Builds a fine-tuning dataset from NOVA brand voice samples.

    Usage:
        builder = BrandVoiceDatasetBuilder()
        dataset = builder.build()
        train, val = builder.split(dataset)
    """

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._samples: list[dict] = []
        self._dataset: list[dict] = []

    def load_samples(self) -> list[dict]:
        """Load raw brand voice samples from JSON."""
        samples_file = self._data_dir / "brand_voice_samples.json"
        if not samples_file.exists():
            raise FileNotFoundError(f"Brand voice samples not found: {samples_file}")

        with open(samples_file) as f:
            self._samples = json.load(f)

        logger.info(f"Loaded {len(self._samples)} brand voice samples")
        return self._samples

    def augment(self, target_count: int = 120) -> list[dict]:
        """
        Augment samples to reach target_count using paraphrasing
        and context variations.

        Args:
            target_count: Target number of training examples.

        Returns:
            List of augmented training examples.
        """
        if not self._samples:
            self.load_samples()

        augmented = []

        # First pass: add all original samples
        for sample in self._samples:
            augmented.append({
                "input": sample["input"],
                "output": sample["ideal_response"],
                "category": sample["category"],
                "source": "original",
            })

        # Second pass: create variations by rephrasing input
        random.seed(42)  # Reproducible
        attempts = 0
        max_attempts = target_count * 5

        while len(augmented) < target_count and attempts < max_attempts:
            sample = random.choice(self._samples)
            input_variations = _paraphrase_input(sample["input"])

            for var_input in input_variations:
                if len(augmented) >= target_count:
                    break
                if var_input != sample["input"]:
                    augmented.append({
                        "input": var_input,
                        "output": sample["ideal_response"],
                        "category": sample["category"],
                        "source": "augmented_paraphrase",
                    })
            attempts += 1

        # Third pass: add context-prefixed variations
        for sample in self._samples:
            if len(augmented) >= target_count:
                break
            category = sample.get("category", "general_support")
            if category in _CONTEXT_VARIATIONS:
                for ctx in _CONTEXT_VARIATIONS[category][:2]:
                    if len(augmented) >= target_count:
                        break
                    combined_input = f"{ctx} Also, {sample['input'].lower()}"
                    augmented.append({
                        "input": combined_input,
                        "output": sample["ideal_response"],
                        "category": category,
                        "source": "augmented_context",
                    })

        self._dataset = augmented[:target_count]
        logger.info(f"Augmented to {len(self._dataset)} examples")
        return self._dataset

    def format_for_training(self, dataset: Optional[list[dict]] = None) -> list[dict]:
        """
        Format dataset as instruction-response pairs with chat template.

        Returns:
            List of dicts with 'text' field containing formatted conversations.
        """
        data = dataset or self._dataset
        if not data:
            raise ValueError("No dataset to format. Run augment() first.")

        formatted = []
        for example in data:
            # Llama 3.2 chat format
            conversation = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{NOVA_SYSTEM_PROMPT}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{example['input']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{example['output']}<|eot_id|>"
            )
            formatted.append({
                "text": conversation,
                "input": example["input"],
                "output": example["output"],
                "category": example.get("category", "unknown"),
                "source": example.get("source", "unknown"),
            })

        logger.info(f"Formatted {len(formatted)} examples for training")
        return formatted

    def split(
        self, dataset: Optional[list[dict]] = None, val_ratio: float = 0.2, seed: int = 42
    ) -> tuple[list[dict], list[dict]]:
        """
        Split dataset into train and validation sets.

        Args:
            dataset: Dataset to split. Uses self._dataset if None.
            val_ratio: Fraction for validation (default 0.2).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_set, val_set).
        """
        data = dataset or self._dataset
        if not data:
            raise ValueError("No dataset to split.")

        random.seed(seed)
        shuffled = list(data)
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - val_ratio))
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]

        logger.info(f"Split: {len(train)} train, {len(val)} validation")
        return train, val

    def build(self, target_count: int = 120) -> list[dict]:
        """
        Full pipeline: load → augment → format.

        Args:
            target_count: Target number of examples.

        Returns:
            Formatted dataset ready for training.
        """
        self.load_samples()
        self.augment(target_count)
        formatted = self.format_for_training()
        return formatted

    def get_brand_markers(self, text: str) -> dict:
        """Extract brand voice markers from text."""
        return _extract_brand_markers(text)

    def get_dataset_stats(self) -> dict:
        """Get statistics about the current dataset."""
        if not self._dataset:
            return {"total": 0}

        categories = {}
        sources = {}
        input_lengths = []
        output_lengths = []

        for ex in self._dataset:
            cat = ex.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            src = ex.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            input_lengths.append(len(ex["input"].split()))
            output_lengths.append(len(ex["output"].split()))

        return {
            "total": len(self._dataset),
            "categories": categories,
            "sources": sources,
            "avg_input_tokens": sum(input_lengths) / len(input_lengths),
            "avg_output_tokens": sum(output_lengths) / len(output_lengths),
            "total_original": sources.get("original", 0),
            "total_augmented": len(self._dataset) - sources.get("original", 0),
        }