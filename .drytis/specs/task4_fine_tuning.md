# Task 4: Fine-Tuning — NOVA Brand Voice Model

## Objective
QLoRA fine-tune a small Llama model to adopt NOVA's brand voice (warm, friendly,
emoji-enhanced, proactive). Track with W&B, evaluate brand consistency, and prepare
the model for inference in Task 5.

## Files to Create/Modify
- `fine_tuning/dataset_prep.py` — Dataset prep from brand_voice_samples.json
- `fine_tuning/qlora_config.py` — QLoRA hyperparameters & config
- `fine_tuning/train.py` — Training script with W&B tracking
- `fine_tuning/inference.py` — Inference pipeline for Task 5
- `fine_tuning/__init__.py` — Updated module exports
- `tests/test_fine_tuning.py` — Full test suite
- `notebooks/task4_fine_tuning.ipynb` — Colab notebook

## Acceptance Criteria

### Dataset Preparation
- [ ] Load 12 brand voice samples from data/brand_voice_samples.json
- [ ] Augment to 100+ training examples (paraphrase, context variations)
- [ ] Format as instruction-response pairs for chat template
- [ ] Train/validation split (80/20)
- [ ] Export as HuggingFace Dataset

### QLoRA Configuration
- [ ] 4-bit quantization (bitsandbytes NF4)
- [ ] LoRA rank 16, alpha 32, dropout 0.05
- [ ] Target modules: q_proj, k_proj, v_proj, o_proj
- [ ] Colab T4 GPU compatible (max 16GB VRAM)

### Training
- [ ] SFTTrainer from trl library
- [ ] W&B experiment tracking (loss, perplexity)
- [ ] Gradient checkpointing for memory efficiency
- [ ] Evaluation during training
- [ ] Save best checkpoint

### Evaluation
- [ ] BLEU score against expected brand voice outputs
- [ ] ROUGE-L score for response similarity
- [ ] Brand voice keyword analysis (emojis, warmth markers)
- [ ] Perplexity on held-out examples

### Inference
- [ ] Load fine-tuned model from checkpoint
- [ ] Generate brand-voice responses for test inputs
- [ ] Compare base vs fine-tuned model outputs
- [ ] Modular interface for Task 5 integration

### Tests
- [ ] Dataset prep tests (loading, augmentation, formatting)
- [ ] Config tests (parameter validation)
- [ ] Evaluation metric tests
- [ ] Inference mock tests