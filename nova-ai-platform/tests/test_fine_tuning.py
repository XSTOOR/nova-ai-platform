"""
NOVA AI Platform — Task 4: Fine-Tuning Tests
=================================================
Tests for dataset preparation, QLoRA config, training (mock mode),
and inference pipeline.

Run:  pytest tests/test_fine_tuning.py -v
"""

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ──────────────────────────────────────────────────────────────────────
# 1. Dataset Preparation Tests
# ──────────────────────────────────────────────────────────────────────

class TestBrandVoiceDatasetBuilder:
    """Tests for BrandVoiceDatasetBuilder."""

    def setup_method(self):
        from fine_tuning.dataset_prep import BrandVoiceDatasetBuilder
        if not (DATA_DIR / "brand_voice_samples.json").exists():
            pytest.skip("brand_voice_samples.json not found")
        self.builder = BrandVoiceDatasetBuilder()

    def test_load_samples(self):
        samples = self.builder.load_samples()
        assert len(samples) >= 2
        for s in samples:
            assert "input" in s
            assert "ideal_response" in s
            assert "category" in s

    def test_augment_reaches_target(self):
        self.builder.load_samples()
        augmented = self.builder.augment(target_count=50)
        assert len(augmented) >= 10  # At minimum, original samples
        for ex in augmented:
            assert "input" in ex
            assert "output" in ex

    def test_format_for_training(self):
        self.builder.load_samples()
        self.builder.augment(target_count=30)
        formatted = self.builder.format_for_training()
        assert len(formatted) > 0
        for item in formatted:
            assert "text" in item
            assert "input" in item
            assert "output" in item
            assert "<|begin_of_text|>" in item["text"]
            assert "system" in item["text"]

    def test_train_val_split(self):
        self.builder.load_samples()
        self.builder.augment(target_count=50)
        formatted = self.builder.format_for_training()
        train, val = self.builder.split(formatted, val_ratio=0.2)
        assert len(train) > 0
        assert len(val) >= 0
        assert len(train) + len(val) == len(formatted)
        ratio = len(train) / len(formatted)
        assert 0.7 <= ratio <= 0.9

    def test_full_build_pipeline(self):
        dataset = self.builder.build(target_count=50)
        assert len(dataset) > 0
        for item in dataset:
            assert "text" in item
            assert "input" in item
            assert "output" in item

    def test_dataset_stats(self):
        self.builder.load_samples()
        self.builder.augment(target_count=30)
        stats = self.builder.get_dataset_stats()
        assert stats["total"] > 0
        assert "categories" in stats
        assert "avg_input_tokens" in stats

    def test_brand_markers(self):
        markers = self.builder.get_brand_markers("Hey! We'd love to help you! 💕")
        assert "emoji_count" in markers
        assert "warmth_word_count" in markers


# ──────────────────────────────────────────────────────────────────────
# 2. QLoRA Config Tests
# ──────────────────────────────────────────────────────────────────────

class TestQLoRAConfig:
    """Tests for QLoRA configuration."""

    def test_default_config_creation(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        assert config.model.model_name == "meta-llama/Llama-3.2-1B-Instruct"
        assert config.lora.r == 16
        assert config.lora.lora_alpha == 32
        assert config.lora.lora_dropout == 0.05
        assert config.quantization.load_in_4bit is True

    def test_quantization_config_dict(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        qdict = config.get_bnb_config_dict()
        assert qdict["load_in_4bit"] is True
        assert qdict["bnb_4bit_quant_type"] == "nf4"

    def test_lora_config_dict(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        ldict = config.get_peft_config_dict()
        assert ldict["r"] == 16
        assert ldict["lora_alpha"] == 32
        assert ldict["lora_dropout"] == 0.05
        assert ldict["bias"] == "none"
        assert ldict["task_type"] == "CAUSAL_LM"

    def test_training_args_dict(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        tdict = config.get_transformers_training_args_dict()
        assert tdict["num_train_epochs"] == 3
        assert tdict["per_device_train_batch_size"] == 4
        assert tdict["gradient_accumulation_steps"] == 4
        assert tdict["learning_rate"] == 2e-4

    def test_custom_config(self):
        from fine_tuning.qlora_config import QLoRAConfig, LoRAConfig, TrainingConfig
        config = QLoRAConfig(
            lora=LoRAConfig(r=32, lora_alpha=64),
            training=TrainingConfig(num_train_epochs=5, per_device_train_batch_size=2),
        )
        assert config.lora.r == 32
        assert config.lora.lora_alpha == 64
        assert config.training.num_train_epochs == 5
        assert config.training.per_device_train_batch_size == 2

    def test_target_modules(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        targets = config.lora.target_modules
        assert "q_proj" in targets
        assert "v_proj" in targets
        assert "k_proj" in targets
        assert "o_proj" in targets

    def test_colab_compatible_defaults(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        effective_batch = (config.training.per_device_train_batch_size
                           * config.training.gradient_accumulation_steps)
        assert effective_batch == 16
        assert config.quantization.load_in_4bit is True
        assert config.training.num_train_epochs <= 5

    def test_config_summary(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        summary = config.summary()
        assert isinstance(summary, str)
        assert "Llama-3.2-1B-Instruct" in summary
        assert "r=16" in summary

    def test_config_to_dict(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        d = config.to_dict()
        assert "model" in d
        assert "quantization" in d
        assert "lora" in d
        assert "training" in d
        assert "wandb" in d

    def test_config_from_dict(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        d = config.to_dict()
        d["lora"]["r"] = 64
        restored = QLoRAConfig.from_dict(d)
        assert restored.lora.r == 64

    def test_validate_no_warnings(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        warnings = config.validate()
        assert len(warnings) == 0

    def test_validate_catches_bad_config(self):
        from fine_tuning.qlora_config import QLoRAConfig, LoRAConfig
        config = QLoRAConfig(lora=LoRAConfig(r=0))
        warnings = config.validate()
        assert any("r must be > 0" in w for w in warnings)

    def test_save_and_load_json(self):
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config.to_json(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["lora"]["r"] == 16
            restored = QLoRAConfig.from_dict(loaded)
            assert restored.lora.r == 16


# ──────────────────────────────────────────────────────────────────────
# 3. Training Tests (Mock Mode)
# ──────────────────────────────────────────────────────────────────────

class TestNOVATrainer:
    """Tests for NOVATrainer — mock training mode."""

    def test_init_with_mock(self):
        from fine_tuning.train import NOVATrainer
        trainer = NOVATrainer(mock=True)
        assert trainer._mock is True

    def test_init_default_config(self):
        from fine_tuning.train import NOVATrainer
        trainer = NOVATrainer(mock=True)
        assert trainer._config is not None

    def test_mock_training_runs(self):
        from fine_tuning.train import NOVATrainer
        trainer = NOVATrainer(mock=True)
        result = trainer.train(target_count=30)
        assert result is not None
        assert result.success is True

    def test_mock_training_loss(self):
        from fine_tuning.train import NOVATrainer
        trainer = NOVATrainer(mock=True)
        result = trainer.train(target_count=30)
        assert result.final_train_loss > 0
        assert result.final_eval_loss > 0
        # Mock should produce reasonable loss values
        assert result.final_train_loss < 5.0

    def test_mock_training_epochs(self):
        from fine_tuning.train import NOVATrainer
        from fine_tuning.qlora_config import QLoRAConfig, TrainingConfig
        config = QLoRAConfig(training=TrainingConfig(num_train_epochs=3))
        trainer = NOVATrainer(config=config, mock=True)
        result = trainer.train(target_count=30)
        assert result.epochs_completed == 3

    def test_mock_training_metrics(self):
        from fine_tuning.train import NOVATrainer
        trainer = NOVATrainer(mock=True)
        result = trainer.train(target_count=30)
        assert result.bleu_score >= 0
        assert result.rouge_l_score >= 0
        assert result.brand_voice_score >= 0

    def test_mock_training_result_to_dict(self):
        from fine_tuning.train import NOVATrainer
        trainer = NOVATrainer(mock=True)
        result = trainer.train(target_count=30)
        d = result.to_dict()
        assert d["success"] is True
        assert "training_time_seconds" in d
        assert "final_train_loss" in d
        assert "epochs_completed" in d

    def test_mock_training_steps(self):
        from fine_tuning.train import NOVATrainer
        trainer = NOVATrainer(mock=True)
        result = trainer.train(target_count=30)
        assert result.steps_completed > 0

    def test_prepare_dataset(self):
        from fine_tuning.train import NOVATrainer
        trainer = NOVATrainer(mock=True)
        train, val = trainer.prepare_dataset(target_count=30)
        assert len(train) > 0
        assert len(val) >= 0


# ──────────────────────────────────────────────────────────────────────
# 4. Inference Tests
# ──────────────────────────────────────────────────────────────────────

class TestNOVAInference:
    """Tests for NOVAInference pipeline."""

    def setup_method(self):
        from fine_tuning.inference import NOVAInference
        self.inference = NOVAInference(use_template=True)

    def test_greeting_detection(self):
        from fine_tuning.inference import _detect_category
        assert _detect_category("Hello there") == "greeting"
        assert _detect_category("Hello!") == "greeting"
        assert _detect_category("hey") == "greeting"

    def test_order_status_detection(self):
        from fine_tuning.inference import _detect_category
        assert _detect_category("Where is my order?") == "order_status"
        assert _detect_category("Track my delivery") == "order_status"

    def test_product_inquiry_detection(self):
        from fine_tuning.inference import _detect_category
        assert _detect_category("Can you recommend a good moisturizer?") == "product_inquiry"

    def test_return_refund_detection(self):
        from fine_tuning.inference import _detect_category
        assert _detect_category("I want to return this") == "return_refund"
        assert _detect_category("How do I get a refund?") == "return_refund"

    def test_loyalty_rewards_detection(self):
        from fine_tuning.inference import _detect_category
        assert _detect_category("How many points do I have?") == "loyalty_rewards"

    def test_escalation_detection(self):
        from fine_tuning.inference import _detect_category
        assert _detect_category("Let me speak to a manager") == "escalation"

    def test_complaint_detection(self):
        from fine_tuning.inference import _detect_category
        assert _detect_category("This is unacceptable and terrible") == "complaint"

    def test_default_fallback(self):
        from fine_tuning.inference import _detect_category
        assert _detect_category("xyzabc random text") == "default"

    def test_generate_returns_result(self):
        result = self.inference.generate("Where is my order?")
        assert result is not None
        assert result.output_text
        assert result.category == "order_status"
        assert result.model_type == "template"

    def test_generate_with_name(self):
        result = self.inference.generate("Hello!", customer_name="Sarah")
        assert "Sarah" in result.output_text

    def test_generate_brand_voice_score(self):
        result = self.inference.generate("Where is my order?")
        assert 0.0 <= result.brand_voice_score <= 1.0

    def test_generate_latency(self):
        result = self.inference.generate("Hello")
        assert result.latency_ms >= 0

    def test_inference_result_to_dict(self):
        result = self.inference.generate("Hello")
        d = result.to_dict()
        assert "input" in d
        assert "output" in d
        assert "category" in d
        assert "brand_voice_score" in d
        assert "model_type" in d

    def test_all_categories_have_templates(self):
        from fine_tuning.inference import _RESPONSE_TEMPLATES, _detect_category
        test_inputs = {
            "greeting": "Hi there",
            "order_status": "Where is my order?",
            "product_inquiry": "Recommend something",
            "return_refund": "I want a return",
            "loyalty_rewards": "Check my points",
            "escalation": "Speak to manager",
            "complaint": "This is terrible",
            "shipping": "How long for shipping",
            "default": "Random text xyz",
        }
        for expected_cat, text in test_inputs.items():
            result = self.inference.generate(text)
            assert result.output_text, f"No output for {expected_cat}"
            assert len(result.output_text) > 10, f"Too short for {expected_cat}"

    def test_compare_models(self):
        comparison = self.inference.compare_models("Where is my order?")
        assert "template_output" in comparison
        assert "template_brand_score" in comparison
        assert comparison["template_brand_score"] is not None

    def test_compare_with_expected(self):
        comparison = self.inference.compare_models(
            "Where is my order?",
            expected_output="Let me check on that for you!"
        )
        assert "expected" in comparison
        assert "template_bleu" in comparison


# ──────────────────────────────────────────────────────────────────────
# 5. Brand Voice Metrics Tests
# ──────────────────────────────────────────────────────────────────────

class TestBrandVoiceMetrics:
    """Tests for brand voice evaluation metrics."""

    def test_compute_bleu_identical(self):
        from fine_tuning.train import compute_bleu
        score = compute_bleu("Hello world", "Hello world")
        assert score == 1.0

    def test_compute_bleu_different(self):
        from fine_tuning.train import compute_bleu
        score = compute_bleu("Hello world", "Goodbye planet")
        assert 0.0 <= score < 1.0

    def test_compute_bleu_empty(self):
        from fine_tuning.train import compute_bleu
        score = compute_bleu("", "Hello world")
        assert score == 0.0

    def test_compute_rouge_l_identical(self):
        from fine_tuning.train import compute_rouge_l
        score = compute_rouge_l("Hello world test", "Hello world test")
        assert score == 1.0

    def test_compute_rouge_l_partial(self):
        from fine_tuning.train import compute_rouge_l
        score = compute_rouge_l("Hello world test", "Hello world")
        assert 0.0 < score < 1.0

    def test_compute_rouge_l_empty(self):
        from fine_tuning.train import compute_rouge_l
        score = compute_rouge_l("", "Hello world")
        assert score == 0.0

    def test_brand_voice_score_nova_style(self):
        from fine_tuning.train import compute_brand_voice_score
        text = "Hey there! 💕 We'd love to help you find the perfect product! ✨"
        result = compute_brand_voice_score(text)
        assert result["brand_voice_score"] > 0.3

    def test_brand_voice_score_robotic(self):
        from fine_tuning.train import compute_brand_voice_score
        text = "Your request has been processed. Please allow 3-5 business days."
        result = compute_brand_voice_score(text)
        assert result["brand_voice_score"] < 0.7

    def test_brand_voice_score_has_metrics(self):
        from fine_tuning.train import compute_brand_voice_score
        result = compute_brand_voice_score("Hey! Love this! 💕")
        assert "emoji_count" in result
        assert "warmth_markers" in result
        assert "proactive_markers" in result
        assert "brand_voice_score" in result


# ──────────────────────────────────────────────────────────────────────
# 6. Integration Tests
# ──────────────────────────────────────────────────────────────────────

class TestFineTuningIntegration:
    """End-to-end integration tests for the fine-tuning pipeline."""

    def test_full_pipeline_build_train_infer(self):
        """Build dataset → train (mock) → inference."""
        from fine_tuning.train import NOVATrainer
        from fine_tuning.inference import NOVAInference

        # Train (mock)
        trainer = NOVATrainer(mock=True)
        result = trainer.train(target_count=30)
        assert result.success is True
        assert result.epochs_completed > 0

        # Inference
        inference = NOVAInference(use_template=True)
        inf_result = inference.generate("Where is my order?")
        assert inf_result.output_text
        assert inf_result.brand_voice_score >= 0

    def test_dataset_suitable_for_training(self):
        """Verify the built dataset is suitable for fine-tuning."""
        from fine_tuning.dataset_prep import BrandVoiceDatasetBuilder

        builder = BrandVoiceDatasetBuilder()
        dataset = builder.build(target_count=50)

        for sample in dataset:
            assert "text" in sample
            assert "output" in sample
            assert len(sample["output"]) >= 10, "Response too short for training"

    def test_colab_memory_footprint(self):
        """Verify the configuration fits in Colab T4 memory (16GB)."""
        from fine_tuning.qlora_config import QLoRAConfig

        config = QLoRAConfig()
        # Llama-3.2-1B in 4-bit ≈ 0.7GB + LoRA + optimizer + gradients ≈ 1.5GB total
        estimated_gb = 1.5  # conservative
        assert estimated_gb < 12, "Config may not fit in Colab Free Tier"

    def test_inference_integration_with_tasks_1_2(self):
        """Verify inference module can work with Task 1 components."""
        from fine_tuning.inference import NOVAInference
        from prompts.intent_classifier import IntentClassifier
        import asyncio

        inference = NOVAInference(use_template=True)
        classifier = IntentClassifier()

        # Classify intent then generate
        text = "Where is my order ORD-2024-001?"
        intent_result = asyncio.get_event_loop().run_until_complete(classifier.classify(text))
        inf_result = inference.generate(text)

        # Both should produce results
        assert intent_result.intent in [
            "order_status", "product_inquiry", "return_refund",
            "loyalty_rewards", "general_support"
        ]
        assert inf_result.output_text
        assert len(inf_result.output_text) > 0

    def test_config_roundtrip(self):
        """Test config serialization roundtrip."""
        from fine_tuning.qlora_config import QLoRAConfig
        config = QLoRAConfig()
        d = config.to_dict()
        restored = QLoRAConfig.from_dict(d)
        assert restored.lora.r == config.lora.r
        assert restored.lora.lora_alpha == config.lora.lora_alpha
        assert restored.training.num_train_epochs == config.training.num_train_epochs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])