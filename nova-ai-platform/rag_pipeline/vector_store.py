"""
NOVA AI Platform — ChromaDB Vector Store
==========================================
Manages the ChromaDB collection for NOVA's product and FAQ knowledge base.

Features:
  - Build collection from products.json + faqs.json
  - Persist to disk (chroma_db/ directory)
  - Add, query, delete documents
  - Rich metadata filtering
  - Batch operations for efficiency
"""

import json
import logging
from pathlib import Path
from typing import Optional

import chromadb

from config.settings import Config
from rag_pipeline.embedder import NOVAEmbedder

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# ──────────────────────────────────────────────────────────────────────
# Document Preparation
# ──────────────────────────────────────────────────────────────────────

def _load_and_prepare_documents() -> tuple[list[str], list[dict], list[str]]:
    """
    Load products and FAQs from JSON files and prepare for ChromaDB.

    Returns:
        Tuple of (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []

    # Load products
    products_file = DATA_DIR / "products.json"
    if products_file.exists():
        with open(products_file) as f:
            products = json.load(f)

        for product in products:
            # Create rich document text for better retrieval
            doc_parts = [
                f"Product: {product['name']}",
                f"Category: {product['category']} > {product.get('subcategory', '')}",
                f"Price: ${product['price']:.2f}",
                f"Description: {product.get('description', '')}",
            ]
            if product.get("ingredients"):
                doc_parts.append(f"Ingredients: {product['ingredients']}")
            if product.get("materials"):
                doc_parts.append(f"Materials: {product['materials']}")
            if product.get("skin_type"):
                doc_parts.append(f"Skin type: {product['skin_type']}")
            if product.get("skin_concerns"):
                doc_parts.append(f"Skin concerns: {', '.join(product['skin_concerns'])}")
            if product.get("finish"):
                doc_parts.append(f"Finish: {product['finish']}")
            if product.get("tags"):
                doc_parts.append(f"Tags: {', '.join(product['tags'])}")

            doc_text = "\n".join(doc_parts)
            documents.append(doc_text)

            meta = {
                "source": "product",
                "product_id": product["id"],
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "rating": product.get("rating", 0),
                "bestseller": product.get("bestseller", False),
            }
            if product.get("in_stock") is not None:
                meta["in_stock"] = product["in_stock"]

            metadatas.append(meta)
            ids.append(f"product_{product['id']}")

    # Load FAQs
    faqs_file = DATA_DIR / "faqs.json"
    if faqs_file.exists():
        with open(faqs_file) as f:
            faqs = json.load(f)

        for faq in faqs:
            doc_text = f"FAQ: {faq['question']}\nAnswer: {faq['answer']}\nCategory: {faq['category']}"
            if faq.get("keywords"):
                doc_text += f"\nKeywords: {', '.join(faq['keywords'])}"

            documents.append(doc_text)
            metadatas.append({
                "source": "faq",
                "faq_id": faq["id"],
                "faq_category": faq["category"],
                "name": faq["question"][:100],
            })
            ids.append(f"faq_{faq['id']}")

    return documents, metadatas, ids


# ──────────────────────────────────────────────────────────────────────
# ChromaDB Vector Store
# ──────────────────────────────────────────────────────────────────────

class NovAVectorStore:
    """
    ChromaDB-backed vector store for NOVA's product and FAQ knowledge.

    Usage:
        store = NovAVectorStore()
        store.build()  # Build from data files (first time)
        results = store.query("moisturizer for dry skin", n_results=5)
    """

    def __init__(
        self,
        embedder: Optional[NOVAEmbedder] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self._embedder = embedder or NOVAEmbedder(force_mock=True)
        self._persist_dir = persist_directory or Config.chroma.persist_directory
        self._collection_name = collection_name or Config.chroma.collection_name
        self._client = None
        self._collection = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Get or create the ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        """Get or create the collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": Config.chroma.distance_function},
            )
        return self._collection

    def build(self, force_rebuild: bool = False) -> dict:
        """
        Build the vector store from data files.

        Args:
            force_rebuild: If True, delete existing collection and rebuild.

        Returns:
            Dict with build statistics.
        """
        collection = self._get_collection()

        # Check if already populated
        if collection.count() > 0 and not force_rebuild:
            logger.info(f"Collection already has {collection.count()} documents. Use force_rebuild=True to rebuild.")
            return {
                "status": "skipped",
                "existing_count": collection.count(),
                "message": "Collection already populated. Use force_rebuild=True to rebuild.",
            }

        # Force rebuild: delete and recreate
        if force_rebuild and collection.count() > 0:
            client = self._get_client()
            client.delete_collection(self._collection_name)
            self._collection = None
            collection = self._get_collection()

        # Load and prepare documents
        documents, metadatas, ids = _load_and_prepare_documents()
        logger.info(f"Prepared {len(documents)} documents for embedding")

        # Generate embeddings
        embeddings = self._embedder.embed_documents(documents)
        logger.info(f"Generated {len(embeddings)} embeddings (dim={self._embedder.dimension})")

        # Add to collection in batches
        batch_size = Config.embedding.batch_size
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
            )

        stats = {
            "status": "built",
            "total_documents": collection.count(),
            "products": sum(1 for m in metadatas if m["source"] == "product"),
            "faqs": sum(1 for m in metadatas if m["source"] == "faq"),
            "embedding_dimension": self._embedder.dimension,
            "embedder": self._embedder.model_name,
        }
        logger.info(f"Vector store built: {stats}")
        return stats

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> dict:
        """
        Query the vector store for similar documents.

        Args:
            query_text: The search query.
            n_results: Number of results to return.
            where: Metadata filter (e.g., {"source": "product"}).
            where_document: Document content filter.

        Returns:
            Dict with documents, metadatas, distances, ids.
        """
        if not query_text or not query_text.strip():
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

        collection = self._get_collection()
        query_embedding = self._embedder.embed_query(query_text)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, max(1, collection.count())),
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = collection.query(**kwargs)
        return results

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Add custom documents to the collection."""
        collection = self._get_collection()
        embeddings = self._embedder.embed_documents(documents)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        collection = self._get_collection()
        collection.delete(ids=ids)

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self._get_collection().count()

    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        collection = self._get_collection()
        return {
            "name": self._collection_name,
            "count": collection.count(),
            "persist_directory": self._persist_dir,
            "embedder": self._embedder.model_name,
            "dimension": self._embedder.dimension,
        }