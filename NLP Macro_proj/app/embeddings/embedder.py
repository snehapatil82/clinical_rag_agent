"""
Sentence Transformers embedding wrapper for Medical RAG.
Generates dense L2-normalized vector embeddings for both queries and document chunks
to enable true cosine similarity search in FAISS.
"""

from typing import List, Optional, Union
import numpy as np

from app.config import settings
from app.utils.logging_config import setup_logger

logger = setup_logger("embeddings.embedder")


class MedicalEmbedder:
    """
    Embedding engine backed by Sentence Transformers.
    Computes unit-normalized embeddings to allow Inner Product (IP) to compute Cosine Similarity.
    """

    _instance: Optional["MedicalEmbedder"] = None
    _model = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if MedicalEmbedder._model is None or getattr(self, "_current_model_name", None) != self.model_name:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' on device '{self.device}'...")
            from sentence_transformers import SentenceTransformer
            MedicalEmbedder._model = SentenceTransformer(self.model_name, device=self.device)
            self._current_model_name = self.model_name
            logger.info(f"SentenceTransformer '{self.model_name}' loaded successfully. Embedding dimension: {self.dimension}")

    @property
    def model(self):
        if MedicalEmbedder._model is None:
            self._load_model()
        return MedicalEmbedder._model

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimensionality (e.g., 384 for all-MiniLM-L6-v2)."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a 1D normalized float32 numpy vector.
        """
        if not text or not text.strip():
            # Return zero vector of model dimension
            return np.zeros((self.dimension,), dtype=np.float32)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user search query into a 1D normalized float32 numpy vector.
        Ensures the query resides in the exact same semantic vector space as documents.
        """
        return self.embed_text(query)

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Embed a list of text chunks in batches.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per forward pass.
            show_progress_bar: Whether to display a tqdm progress bar.

        Returns:
            2D numpy array of shape (N, dimension) with float32 dtype.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        return embeddings.astype(np.float32)


# Module-level singleton helper
def get_embedder() -> MedicalEmbedder:
    """Returns a singleton instance of MedicalEmbedder."""
    if MedicalEmbedder._instance is None:
        MedicalEmbedder._instance = MedicalEmbedder()
    return MedicalEmbedder._instance
