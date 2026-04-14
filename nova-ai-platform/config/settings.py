"""
NOVA AI Platform - Central Configuration
==========================================
All settings, API keys, and configuration constants.
Uses environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    # Primary: OpenRouter (free models available)
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

    # Alternative: Groq (fast, free tier)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Active provider: "openrouter" or "groq"
    active_provider: str = os.getenv("LLM_PROVIDER", "openrouter")

    # Generation parameters
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9

    @property
    def api_key(self) -> str:
        if self.active_provider == "groq":
            return self.groq_api_key
        return self.openrouter_api_key

    @property
    def model_name(self) -> str:
        if self.active_provider == "groq":
            return self.groq_model
        return self.openrouter_model

    @property
    def base_url(self) -> str:
        if self.active_provider == "groq":
            return "https://api.groq.com/openai/v1"
        return self.openrouter_base_url


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""
    persist_directory: str = str(CHROMA_DIR)
    collection_name: str = "nova_products"
    distance_function: str = "cosine"


@dataclass
class RerankerConfig:
    """Re-ranking model configuration."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


@dataclass
class RAGASConfig:
    """RAGAS evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall"
    ])
    sample_size: int = 20


@dataclass
class QLoRAConfig:
    """QLoRA fine-tuning configuration."""
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bits: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


@dataclass
class WandbConfig:
    """Weights & Biases tracking configuration."""
    project: str = "nova-ai-platform"
    entity: str = os.getenv("WANDB_ENTITY", "")
    api_key: str = os.getenv("WANDB_API_KEY", "")
    tags: List[str] = field(default_factory=lambda: ["qlora", "brand-voice", "nova"])


@dataclass
class EscalationConfig:
    """Escalation thresholds and rules."""
    confidence_threshold_low: float = 0.4
    confidence_threshold_high: float = 0.7
    auto_escalation_keywords: List[str] = field(default_factory=lambda: [
        "manager", "supervisor", "complaint", "lawsuit",
        "legal", "refund", "unacceptable", "furious",
        "better business bureau", "attorney"
    ])
    sensitive_topics: List[str] = field(default_factory=lambda: [
        "medical reaction", "allergic reaction", "skin irritation",
        "data breach", "privacy concern", "billing fraud"
    ])


@dataclass
class IntentConfig:
    """Intent classification categories and settings."""
    categories: List[str] = field(default_factory=lambda: [
        "order_status",        # Order tracking, delivery inquiries
        "product_inquiry",     # Product details, recommendations, availability
        "return_refund",       # Returns, exchanges, refunds
        "loyalty_rewards",     # Points, tiers, reward redemptions
        "general_support"      # Shipping, account, general questions
    ])
    escalation_category: str = "escalation"


@dataclass
class InjectionDefenseConfig:
    """Prompt injection defense configuration."""
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+a",
        r"forget\s+(everything|all|your\s+instructions)",
        r"system\s*:\s*",
        r"jailbreak",
        r"pretend\s+you\s+are",
        r"act\s+as\s+if",
        r"new\s+directive",
        r"override\s+(your|all|safety)",
        r"reveal\s+(your|the)\s+(prompt|instructions|system)"
    ])
    max_input_length: int = 2000
    suspicion_threshold: float = 0.6


@dataclass
class AuditConfig:
    """Audit logging configuration."""
    log_dir: str = str(LOGS_DIR / "audit")
    max_log_size_mb: int = 50
    retention_days: int = 90
    log_tool_calls: bool = True
    log_agent_decisions: bool = True
    log_escalations: bool = True


# Singleton configuration instance
class Config:
    """Central configuration singleton."""
    llm = LLMConfig()
    embedding = EmbeddingConfig()
    chroma = ChromaConfig()
    reranker = RerankerConfig()
    ragas = RAGASConfig()
    qlora = QLoRAConfig()
    wandb = WandbConfig()
    escalation = EscalationConfig()
    intent = IntentConfig()
    injection_defense = InjectionDefenseConfig()
    audit = AuditConfig()

    @classmethod
    def to_dict(cls) -> dict:
        """Export all config as dictionary."""
        import dataclasses
        result = {}
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if dataclasses.is_dataclass(attr) and not isinstance(attr, type):
                result[attr_name] = dataclasses.asdict(attr)
        return result