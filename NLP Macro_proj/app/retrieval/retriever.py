"""
Clinical retriever module.
Transforms natural language clinical queries into embeddings, executes FAISS similarity search,
and applies relevance thresholding to guarantee evidence quality before generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import settings
from app.embeddings.embedder import MedicalEmbedder, get_embedder
from app.vectorstore.faiss_store import FAISSVectorStore
from app.utils.logging_config import setup_logger

logger = setup_logger("retrieval.retriever")


@dataclass
class RetrievalResult:
    """Represents an individual retrieved evidence chunk."""
    rank: int
    score: float
    chunk_id: str
    text: str
    document_name: str
    source: str
    page: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> str:
        """Format chunk source as a clean academic citation string."""
        return f"{self.document_name} (Page {self.page})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_name": self.document_name,
            "source": self.source,
            "page": self.page,
            "metadata": self.metadata,
        }


class ClinicalRetriever:
    """
    Semantic retrieval coordinator for clinical question answering.
    Connects query embedding with FAISS vector lookup and relevance thresholding.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: Optional[MedicalEmbedder] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder or get_embedder()
        self.top_k = top_k if top_k is not None else settings.TOP_K
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_threshold: bool = False,
    ) -> List[RetrievalResult]:
        """
        Retrieve the top-K relevant clinical chunks for a query.

        Args:
            query: Clinical question text.
            top_k: Optional override for number of chunks to retrieve.
            threshold: Optional override for similarity cutoff.
            filter_threshold: If True, drops chunks with similarity < threshold.

        Returns:
            List of RetrievalResult objects sorted by descending similarity.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to retrieve().")
            return []

        k = top_k if top_k is not None else self.top_k
        cutoff = threshold if threshold is not None else self.similarity_threshold

        # Step 1: Embed query
        query_vector = self.embedder.embed_query(query.strip())

        # Step 2: Search FAISS
        raw_results = self.vector_store.search(query_vector, top_k=k)

        # Step 3: Wrap into structured RetrievalResult objects
        results: List[RetrievalResult] = []
        for r in raw_results:
            res_obj = RetrievalResult(
                rank=r["rank"],
                score=r["score"],
                chunk_id=r["chunk_id"],
                text=r["text"],
                document_name=r["document_name"],
                source=r["source"],
                page=r["page"],
                metadata=r["metadata"],
            )
            if filter_threshold and res_obj.score < cutoff:
                continue
            results.append(res_obj)

        top_s = results[0].score if results else 0.0
        logger.info(
            f"Query '{query[:40]}...' retrieved {len(results)} chunks (top_score={top_s:.4f})"
        )
        return results

    def is_evidence_sufficient(
        self,
        results: List[RetrievalResult],
        threshold: Optional[float] = None
    ) -> bool:
        """
        Determines whether the retrieved results contain sufficient relevance.
        Safeguard against hallucinating answers on out-of-domain / unsupported queries.
        """
        if not results:
            return False

        cutoff = threshold if threshold is not None else self.similarity_threshold
        # Check if the top retrieved chunk passes the similarity threshold
        top_score = max(r.score for r in results)
        return top_score >= cutoff

    def format_context_for_llm(self, results: List[RetrievalResult]) -> str:
        """
        Formats retrieved chunks into an unambiguous, source-tagged context block for the prompt.
        """
        if not results:
            return "No medical context retrieved."

        context_blocks = []
        for r in results:
            block = (
                f"[DOCUMENT: {r.document_name} | PAGE: {r.page} | CHUNK: {r.chunk_id} | SIMILARITY: {r.score:.2f}]\n"
                f"{r.text.strip()}"
            )
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)
