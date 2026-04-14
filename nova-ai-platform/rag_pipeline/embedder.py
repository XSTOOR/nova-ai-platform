"""
NOVA AI Platform — Embedder
==============================
Wraps HuggingFace sentence-transformers for document and query embeddings.
Falls back to a deterministic mock embedder when models can't be downloaded.

Default model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
"""

import hashlib
import logging
import math
from typing import Union

from config.settings import Config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Mock Embedder (fallback when sentence-transformers unavailable)
# ──────────────────────────────────────────────────────────────────────

class MockEmbedder:
    """
    Deterministic mock embedder that generates consistent vectors from text.
    Used when the real model can't be downloaded (e.g., no internet).
    Produces L2-normalized 384-dim vectors.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._model_name = "mock-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _text_to_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic pseudo-random vector."""
        vector = []
        for i in range(self._dimension):
            seed = hashlib.md5(f"{text}:{i}".encode()).hexdigest()
            val = int(seed[:8], 16) / 0xFFFFFFFF  # 0.0 to 1.0
            vector.append(val * 2 - 1)  # -1.0 to 1.0

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._text_to_vector(text)


# ──────────────────────────────────────────────────────────────────────
# Real HuggingFace Embedder
# ──────────────────────────────────────────────────────────────────────

class HuggingFaceEmbedder:
    """
    Real HuggingFace sentence-transformers embedder.

    Uses all-MiniLM-L6-v2 by default (384 dimensions, fast, small).
    """

    def __init__(self, model_name: str = None):
        self._model_name = model_name or Config.embedding.model_name
        self._model = None
        self._dimension = Config.embedding.dimension

    def _load_model(self):
        """Lazy-load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        self._load_model()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        self._load_model()
        embedding = self._model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


# ──────────────────────────────────────────────────────────────────────
# Embedder Factory
# ──────────────────────────────────────────────────────────────────────

class NOVAEmbedder:
    """
    NOVA's embedder with automatic fallback.

    Tries to load the real HuggingFace model. Falls back to MockEmbedder
    if the model can't be loaded (no internet, no disk space, etc.)

    Usage:
        embedder = NOVAEmbedder()
        vectors = embedder.embed_documents(["Hello world", "Test query"])
        query_vec = embedder.embed_query("Find me a moisturizer")
    """

    def __init__(self, model_name: str = None, force_mock: bool = False):
        """
        Initialize the embedder.

        Args:
            model_name: Model name to use. Defaults to config.
            force_mock: If True, always use the mock embedder.
        """
        self._model_name = model_name or Config.embedding.model_name
        self._inner = None
        self._is_mock = False

        if force_mock:
            self._inner = MockEmbedder(Config.embedding.dimension)
            self._is_mock = True
            logger.info("Using MockEmbedder (forced)")
            return

        try:
            self._inner = HuggingFaceEmbedder(self._model_name)
            # Test embed to verify model loads
            test = self._inner.embed_query("test")
            if test and len(test) > 0:
                logger.info(f"Using HuggingFace embedder: {self._model_name}")
            else:
                raise RuntimeError("Empty embedding returned")
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace model: {e}. Falling back to MockEmbedder.")
            self._inner = MockEmbedder(Config.embedding.dimension)
            self._is_mock = True

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._inner.dimension

    @property
    def is_mock(self) -> bool:
        """Whether using mock embedder."""
        return self._is_mock

    @property
    def model_name(self) -> str:
        return "mock-embedder" if self._is_mock else self._model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document texts.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            text: The query text.

        Returns:
            Embedding vector.
        """
        if not text:
            return [0.0] * self.dimension
        return self._inner.embed_query(text)