"""
NOVA AI Platform — Fine-Tuning Module
========================================
QLoRA fine-tuning for NOVA brand voice model.

Components:
  - BrandVoiceDatasetBuilder: Dataset prep from brand voice samples
  - QLoRAConfig: Complete configuration bundle
  - NOVATrainer: Training with W&B tracking
  - NOVAInference: Inference pipeline for Task 5
  - Metrics: BLEU, ROUGE-L, brand voice scoring
"""

from .dataset_prep import BrandVoiceDatasetBuilder
from .qlora_config import QLoRAConfig
from .train import (
    NOVATrainer, TrainingResult,
    compute_bleu, compute_rouge_l, compute_brand_voice_score,
)
from .inference import NOVAInference, InferenceResult

__all__ = [
    "BrandVoiceDatasetBuilder",
    "QLoRAConfig",
    "NOVATrainer", "TrainingResult",
    "NOVAInference", "InferenceResult",
    "compute_bleu", "compute_rouge_l", "compute_brand_voice_score",
]