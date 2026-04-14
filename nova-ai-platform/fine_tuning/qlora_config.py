"""
NOVA AI Platform — QLoRA Configuration
=========================================
All hyperparameters and configuration for QLoRA fine-tuning.

Designed for Google Colab Free Tier (T4 GPU, 16GB VRAM).
Uses 4-bit NF4 quantization to fit Llama-3.2-1B in memory.
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class BitsAndBytesConfig:
    """4-bit quantization configuration."""
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 512
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    report_to: str = "wandb"
    run_name: Optional[str] = None
    output_dir: str = "./models/nova-brand-voice"
    seed: int = 42
    group_by_length: bool = True
    packing: bool = False


@dataclass
class ModelConfig:
    """Base model configuration."""
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    token: Optional[str] = None
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_cache: bool = False  # Must be False for gradient checkpointing


@dataclass
class WandbConfig:
    """Weights & Biases tracking configuration."""
    project: str = "nova-ai-platform"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list = field(default_factory=lambda: [
        "qlora", "brand-voice", "nova", "llama-3.2-1b"
    ])
    notes: Optional[str] = None
    log_model: str = "checkpoint"


# ──────────────────────────────────────────────────────────────────────
# Complete QLoRA Config Bundle
# ──────────────────────────────────────────────────────────────────────

@dataclass
class QLoRAConfig:
    """
    Complete configuration bundle for QLoRA fine-tuning.

    Usage:
        config = QLoRAConfig()
        print(config.summary())
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: BitsAndBytesConfig = field(default_factory=BitsAndBytesConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            "╔══════════════════════════════════════════╗",
            "║    NOVA QLoRA Fine-Tuning Configuration  ║",
            "╚══════════════════════════════════════════╝",
            "",
            f"Model: {self.model.model_name}",
            f"Quantization: {self.quantization.bnb_4bit_quant_type} (4-bit: {self.quantization.load_in_4bit})",
            f"LoRA: r={self.lora.r}, alpha={self.lora.lora_alpha}, dropout={self.lora.lora_dropout}",
            f"Targets: {', '.join(self.lora.target_modules)}",
            f"Epochs: {self.training.num_train_epochs}",
            f"Batch size: {self.training.per_device_train_batch_size} × {self.training.gradient_accumulation_steps} accumulation",
            f"Effective batch size: {self.training.per_device_train_batch_size * self.training.gradient_accumulation_steps}",
            f"Learning rate: {self.training.learning_rate}",
            f"Scheduler: {self.training.lr_scheduler_type}",
            f"Max seq length: {self.training.max_seq_length}",
            f"Gradient checkpointing: {self.training.gradient_checkpointing}",
            f"Optimizer: {self.training.optim}",
            f"FP16: {self.training.fp16}, BF16: {self.training.bf16}",
            f"W&B project: {self.wandb.project}",
            f"Output: {self.training.output_dir}",
            "",
            f"Estimated VRAM: ~8-10GB (fits Colab T4)",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export all config as nested dict."""
        return {
            "model": self.model.__dict__,
            "quantization": self.quantization.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "wandb": self.wandb.__dict__,
        }

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "QLoRAConfig":
        """Create config from dict."""
        config = cls()
        if "model" in d:
            for k, v in d["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        if "quantization" in d:
            for k, v in d["quantization"].items():
                if hasattr(config.quantization, k):
                    setattr(config.quantization, k, v)
        if "lora" in d:
            for k, v in d["lora"].items():
                if hasattr(config.lora, k):
                    setattr(config.lora, k, v)
        if "training" in d:
            for k, v in d["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        if "wandb" in d:
            for k, v in d["wandb"].items():
                if hasattr(config.wandb, k):
                    setattr(config.wandb, k, v)
        return config

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of warnings."""
        warnings = []

        if self.lora.r <= 0:
            warnings.append("LoRA r must be > 0")
        if self.lora.lora_alpha <= 0:
            warnings.append("LoRA alpha must be > 0")
        if self.training.learning_rate <= 0:
            warnings.append("Learning rate must be > 0")
        if self.training.num_train_epochs <= 0:
            warnings.append("Must have at least 1 epoch")
        if self.training.per_device_train_batch_size <= 0:
            warnings.append("Batch size must be > 0")
        if self.training.max_seq_length <= 0:
            warnings.append("Max sequence length must be > 0")
        if not self.lora.target_modules:
            warnings.append("Must specify at least one target module")

        # Colab T4 warnings
        effective_batch = (self.training.per_device_train_batch_size
                           * self.training.gradient_accumulation_steps)
        if effective_batch > 32:
            warnings.append(f"Effective batch size {effective_batch} may be too large for T4")

        if self.training.max_seq_length > 1024:
            warnings.append(f"Max seq length {self.training.max_seq_length} may cause OOM on T4")

        return warnings

    def get_transformers_training_args_dict(self) -> dict:
        """Export training params compatible with transformers TrainingArguments."""
        d = {}
        t = self.training
        for attr in [
            "num_train_epochs", "per_device_train_batch_size",
            "per_device_eval_batch_size", "gradient_accumulation_steps",
            "learning_rate", "weight_decay", "warmup_ratio",
            "lr_scheduler_type", "max_grad_norm", "fp16", "bf16",
            "logging_steps", "eval_strategy", "eval_steps",
            "save_strategy", "save_steps", "save_total_limit",
            "load_best_model_at_end", "metric_for_best_model",
            "greater_is_better", "gradient_checkpointing",
            "optim", "report_to", "output_dir", "seed",
            "group_by_length",
        ]:
            if hasattr(t, attr):
                d[attr] = getattr(t, attr)
        if t.run_name:
            d["run_name"] = t.run_name
        return d

    def get_peft_config_dict(self) -> dict:
        """Export LoRA params compatible with PEFT LoraConfig."""
        return {
            "r": self.lora.r,
            "lora_alpha": self.lora.lora_alpha,
            "lora_dropout": self.lora.lora_dropout,
            "bias": self.lora.bias,
            "task_type": self.lora.task_type,
            "target_modules": self.lora.target_modules,
        }

    def get_bnb_config_dict(self) -> dict:
        """Export quantization params for bitsandbytes."""
        return {
            "load_in_4bit": self.quantization.load_in_4bit,
            "bnb_4bit_quant_type": self.quantization.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.quantization.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.quantization.bnb_4bit_use_double_quant,
        }