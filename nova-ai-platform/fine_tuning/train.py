"""
NOVA AI Platform — Training Script
=====================================
QLoRA fine-tuning script for NOVA brand voice model.

Supports:
  - Full training on GPU (Colab T4)
  - Mock training for testing (CPU, no model download)
  - W&B experiment tracking
  - Evaluation metrics (loss, BLEU, ROUGE-L)

Usage (Colab):
    from fine_tuning.train import NOVATrainer
    trainer = NOVATrainer()
    results = trainer.train()

Usage (Mock for testing):
    trainer = NOVATrainer(mock=True)
    results = trainer.train()
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fine_tuning.dataset_prep import BrandVoiceDatasetBuilder
from fine_tuning.qlora_config import QLoRAConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute a simplified BLEU score (up to 4-gram).

    Args:
        reference: Expected output text.
        hypothesis: Generated output text.

    Returns:
        BLEU score (0.0 to 1.0).
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # N-gram precisions
    precisions = []
    for n in range(1, min(5, len(hyp_tokens) + 1)):
        ref_ngrams = {}
        for i in range(len(ref_tokens) - n + 1):
            ng = tuple(ref_tokens[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        matches = 0
        total = 0
        for i in range(len(hyp_tokens) - n + 1):
            ng = tuple(hyp_tokens[i:i + n])
            total += 1
            if ng in ref_ngrams and ref_ngrams[ng] > 0:
                matches += 1
                ref_ngrams[ng] -= 1

        precision = matches / max(total, 1)
        precisions.append(max(precision, 1e-10))

    # Geometric mean
    log_avg = sum(math.log(p) for p in precisions) / max(len(precisions), 1)
    return round(bp * math.exp(log_avg), 4)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Compute ROUGE-L (Longest Common Subsequence) score.

    Args:
        reference: Expected output text.
        hypothesis: Generated output text.

    Returns:
        ROUGE-L F1 score (0.0 to 1.0).
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS length
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]

    precision = lcs / max(n, 1)
    recall = lcs / max(m, 1)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def compute_brand_voice_score(text: str) -> dict:
    """
    Evaluate brand voice consistency in generated text.

    Returns:
        Dict with brand voice metrics.
    """
    text_lower = text.lower()
    emoji_count = 0
    for c in text:
        if ord(c) > 0x1F000:
            emoji_count += 1

    warmth_words = [
        "love", "obsessed", "amazing", "perfect", "sweet",
        "gorgeous", "hey", "welcome", "hey there", "no worries",
    ]
    warmth_count = sum(1 for w in warmth_words if w in text_lower)

    proactive_phrases = [
        "want me to", "i can help", "let me", "pro tip",
        "would you like", "shall i", "how about",
    ]
    proactive_count = sum(1 for p in proactive_phrases if p in text_lower)

    return {
        "emoji_count": emoji_count,
        "warmth_markers": warmth_count,
        "proactive_markers": proactive_count,
        "has_follow_up": "?" in text,
        "response_length": len(text.split()),
        "brand_voice_score": round(
            min(1.0, (warmth_count * 0.2 + proactive_count * 0.15
                      + min(emoji_count, 3) * 0.1 + (0.1 if "?" in text else 0))),
            3,
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Training Result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    """Result from a training run."""
    success: bool = True
    training_time_seconds: float = 0.0
    final_train_loss: float = 0.0
    final_eval_loss: float = 0.0
    perplexity: float = 0.0
    epochs_completed: int = 0
    steps_completed: int = 0
    best_model_path: Optional[str] = None
    bleu_score: float = 0.0
    rouge_l_score: float = 0.0
    brand_voice_score: float = 0.0
    wandb_url: Optional[str] = None
    config: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "training_time_seconds": round(self.training_time_seconds, 2),
            "final_train_loss": round(self.final_train_loss, 4),
            "final_eval_loss": round(self.final_eval_loss, 4),
            "perplexity": round(self.perplexity, 2),
            "epochs_completed": self.epochs_completed,
            "steps_completed": self.steps_completed,
            "best_model_path": self.best_model_path,
            "bleu_score": self.bleu_score,
            "rouge_l_score": self.rouge_l_score,
            "brand_voice_score": self.brand_voice_score,
            "wandb_url": self.wandb_url,
        }


# ──────────────────────────────────────────────────────────────────────
# NOVA Trainer
# ──────────────────────────────────────────────────────────────────────

class NOVATrainer:
    """
    QLoRA fine-tuning trainer for NOVA brand voice model.

    Supports:
      - Full training with GPU (Colab T4)
      - Mock training for testing without GPU

    Usage:
        trainer = NOVATrainer(mock=True)  # or mock=False for real training
        results = trainer.train()
    """

    def __init__(
        self,
        config: Optional[QLoRAConfig] = None,
        mock: bool = False,
    ):
        self._config = config or QLoRAConfig()
        self._mock = mock
        self._dataset_builder = BrandVoiceDatasetBuilder()

    def prepare_dataset(self, target_count: int = 120) -> tuple[list[dict], list[dict]]:
        """Prepare and split the training dataset."""
        formatted = self._dataset_builder.build(target_count)
        train, val = self._dataset_builder.split(formatted)
        return train, val

    def train(self, target_count: int = 120) -> TrainingResult:
        """
        Run the fine-tuning process.

        Args:
            target_count: Number of training examples to generate.

        Returns:
            TrainingResult with metrics.
        """
        if self._mock:
            return self._mock_train(target_count)
        else:
            return self._real_train(target_count)

    def _mock_train(self, target_count: int) -> TrainingResult:
        """Simulate training for testing environments (no GPU)."""
        logger.info("Running mock training (no GPU required)")

        start = time.time()
        train, val = self.prepare_dataset(target_count)

        # Simulate training metrics
        epochs = self._config.training.num_train_epochs
        base_loss = 2.5
        loss_decay = 0.3  # Loss reduction per epoch

        train_loss = base_loss
        for epoch in range(epochs):
            train_loss -= loss_decay * (0.8 + 0.4 * (epoch / epochs))

        eval_loss = train_loss * 1.1  # Eval loss slightly higher
        perplexity = math.exp(min(eval_loss, 10))  # Cap to avoid overflow

        # Evaluate on validation set
        bleu_scores = []
        rouge_scores = []
        brand_scores = []

        for example in val[:5]:
            output = example["output"]
            # Simulate a "generated" output (slightly modified)
            gen_words = output.split()
            if len(gen_words) > 10:
                gen_output = " ".join(gen_words[:int(len(gen_words) * 0.85)])
            else:
                gen_output = output

            bleu_scores.append(compute_bleu(output, gen_output))
            rouge_scores.append(compute_rouge_l(output, gen_output))
            brand_scores.append(compute_brand_voice_score(output)["brand_voice_score"])

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        avg_brand = sum(brand_scores) / len(brand_scores) if brand_scores else 0

        elapsed = time.time() - start

        return TrainingResult(
            success=True,
            training_time_seconds=elapsed,
            final_train_loss=round(max(train_loss, 0.1), 4),
            final_eval_loss=round(max(eval_loss, 0.1), 4),
            perplexity=round(perplexity, 2),
            epochs_completed=epochs,
            steps_completed=len(train) * epochs,
            best_model_path="./models/nova-brand-voice-mock",
            bleu_score=round(avg_bleu, 4),
            rouge_l_score=round(avg_rouge, 4),
            brand_voice_score=round(avg_brand, 3),
            config=self._config.to_dict(),
        )

    def _real_train(self, target_count: int) -> TrainingResult:
        """Real QLoRA training on GPU (for Colab)."""
        start = time.time()

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from trl import SFTTrainer
            from datasets import Dataset
        except ImportError as e:
            logger.error(f"Missing dependencies for real training: {e}")
            logger.info("Falling back to mock training.")
            return self._mock_train(target_count)

        # Prepare dataset
        train_data, val_data = self.prepare_dataset(target_count)

        train_dataset = Dataset.from_list(
            [{"text": ex["text"]} for ex in train_data]
        )
        val_dataset = Dataset.from_list(
            [{"text": ex["text"]} for ex in val_data]
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self._config.model.model_name,
            token=self._config.model.token,
            trust_remote_code=self._config.model.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with quantization
        from transformers import BitsAndBytesConfig as BnbConfig

        bnb_config = BnbConfig(**self._config.get_bnb_config_dict())

        model = AutoModelForCausalLM.from_pretrained(
            self._config.model.model_name,
            quantization_config=bnb_config,
            device_map=self._config.model.device_map,
            token=self._config.model.token,
            trust_remote_code=self._config.model.trust_remote_code,
        )

        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        # Apply LoRA
        peft_config = LoraConfig(**self._config.get_peft_config_dict())
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(
            **self._config.get_transformers_training_args_dict()
        )

        # Initialize W&B
        try:
            import wandb
            wandb.init(
                project=self._config.wandb.project,
                entity=self._config.wandb.entity,
                name=self._config.wandb.name,
                tags=self._config.wandb.tags,
                config=self._config.to_dict(),
            )
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

        # Train
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            max_seq_length=self._config.training.max_seq_length,
            packing=self._config.training.packing,
        )

        train_result = trainer.train()
        eval_result = trainer.evaluate()

        # Save model
        output_dir = self._config.training.output_dir
        trainer.save_model(output_dir)

        elapsed = time.time() - start
        eval_loss = eval_result.get("eval_loss", 0)
        perplexity = math.exp(min(eval_loss, 20))

        # Evaluate brand voice
        bleu_scores = []
        rouge_scores = []
        brand_scores = []

        for example in val_data[:5]:
            output = example["output"]
            gen_output = self._generate_sample(model, tokenizer, example["input"])
            bleu_scores.append(compute_bleu(output, gen_output))
            rouge_scores.append(compute_rouge_l(output, gen_output))
            brand_scores.append(compute_brand_voice_score(gen_output)["brand_voice_score"])

        # Finish W&B
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

        return TrainingResult(
            success=True,
            training_time_seconds=elapsed,
            final_train_loss=train_result.training_loss,
            final_eval_loss=eval_loss,
            perplexity=round(perplexity, 2),
            epochs_completed=self._config.training.num_train_epochs,
            steps_completed=train_result.global_step,
            best_model_path=output_dir,
            bleu_score=round(sum(bleu_scores) / max(len(bleu_scores), 1), 4),
            rouge_l_score=round(sum(rouge_scores) / max(len(rouge_scores), 1), 4),
            brand_voice_score=round(sum(brand_scores) / max(len(brand_scores), 1), 3),
            config=self._config.to_dict(),
        )

    def _generate_sample(self, model, tokenizer, input_text: str) -> str:
        """Generate a sample response for evaluation."""
        import torch
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        return response