"""
FAISS Vector Store implementation for Medical RAG.
Persists dense embeddings in an IndexFlatIP index and stores metadata in a synchronized JSON catalog.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import faiss

from app.config import settings
from app.ingestion.chunker import TextChunk
from app.utils.logging_config import setup_logger

logger = setup_logger("vectorstore.faiss_store")


class FAISSVectorStore:
    """
    Manages FAISS vector indexing, serialization, and high-performance similarity search.
    Maps internal FAISS vector indices directly to source document chunk metadata.
    """

    INDEX_FILENAME = "medical.index"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, store_dir: Optional[Union[str, Path]] = None, dimension: Optional[int] = None):
        self.store_dir = Path(store_dir) if store_dir else settings.VECTOR_STORE_DIR
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks_metadata: List[Dict[str, Any]] = []

    @property
    def index_path(self) -> Path:
        return self.store_dir / self.INDEX_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.store_dir / self.METADATA_FILENAME

    def is_indexed(self) -> bool:
        """Check if both the FAISS index and metadata files exist on disk."""
        return self.index_path.exists() and self.metadata_path.exists()

    def build_index(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """
        Construct a new FAISS IndexFlatIP index from chunks and their precomputed embeddings.

        Args:
            chunks: List of TextChunk objects.
            embeddings: 2D numpy array of shape (N, D) with normalized float32 vectors.
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            logger.warning("Empty chunks or embeddings provided to build_index.")
            return

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but embeddings have {embeddings.shape[0]} rows."
            )

        dim = embeddings.shape[1]
        self.dimension = dim
        logger.info(f"Building FAISS IndexFlatIP with dimension {dim} for {len(chunks)} vectors...")

        # Ensure float32 and contiguous memory
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # IndexFlatIP calculates inner product: for unit vectors, IP == Cosine Similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

        # Store metadata aligned with vector IDs (0, 1, ..., N-1)
        self.chunks_metadata = [chunk.to_dict() for chunk in chunks]
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def save(self, target_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Persist the FAISS index and chunk metadata catalog to disk.
        """
        out_dir = Path(target_dir) if target_dir else self.store_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise RuntimeError("Cannot save uninitialized or empty FAISS index.")

        idx_path = out_dir / self.INDEX_FILENAME
        meta_path = out_dir / self.METADATA_FILENAME

        logger.info(f"Saving FAISS index to {idx_path}...")
        faiss.write_index(self.index, str(idx_path))

        logger.info(f"Saving metadata catalog ({len(self.chunks_metadata)} entries) to {meta_path}...")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, indent=2, ensure_ascii=False)

        logger.info("Vector store persisted successfully.")

    def load(self, target_dir: Optional[Union[str, Path]] = None) -> bool:
        """
        Load the FAISS index and metadata catalog from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        load_dir = Path(target_dir) if target_dir else self.store_dir
        idx_path = load_dir / self.INDEX_FILENAME
        meta_path = load_dir / self.METADATA_FILENAME

        if not idx_path.exists() or not meta_path.exists():
            logger.warning(
                f"FAISS index files not found in {load_dir}. Please run 'python scripts/build_index.py' first."
            )
            return False

        try:
            logger.info(f"Loading FAISS index from {idx_path}...")
            self.index = faiss.read_index(str(idx_path))
            self.dimension = self.index.d

            logger.info(f"Loading metadata catalog from {meta_path}...")
            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunks_metadata = json.load(f)

            if self.index.ntotal != len(self.chunks_metadata):
                logger.warning(
                    f"Count mismatch: Index has {self.index.ntotal} vectors, metadata has {len(self.chunks_metadata)} entries."
                )

            logger.info(
                f"FAISS vector store loaded successfully: {self.index.ntotal} vectors, dimension={self.dimension}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {load_dir}: {e}", exc_info=True)
            return False

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Perform nearest neighbor semantic search in FAISS.

        Args:
            query_vector: 1D or 2D numpy array of shape (D,) or (1, D).
            top_k: Number of nearest chunks to retrieve.

        Returns:
            List of result dictionaries containing rank, score, text, and metadata.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search invoked on empty or uninitialized FAISS index.")
            return []

        # Prepare query shape (1, D) and float32
        if query_vector.ndim == 1:
            q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        else:
            q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, actual_k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue
            chunk_data = self.chunks_metadata[idx]
            meta = chunk_data.get("metadata", {})
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk_data.get("chunk_id", f"chunk_{idx}"),
                "text": chunk_data.get("text", ""),
                "document_name": meta.get("document_name", "unknown"),
                "source": meta.get("source", meta.get("document_name", "Knowledge Base")),
                "page": meta.get("page_number", 1),
                "metadata": meta,
            })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics about the current vector store."""
        if self.index is None:
            return {
                "indexed": False,
                "total_chunks": 0,
                "total_documents": 0,
                "dimension": self.dimension,
                "index_type": "None",
            }

        unique_docs = {
            m.get("metadata", {}).get("document_name", "")
            for m in self.chunks_metadata
            if m.get("metadata", {}).get("document_name")
        }

        return {
            "indexed": True,
            "total_chunks": self.index.ntotal,
            "total_documents": len(unique_docs),
            "documents_list": sorted(list(unique_docs)),
            "dimension": self.dimension,
            "index_type": "IndexFlatIP (Cosine Similarity)",
        }

    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """
        Incrementally append new chunks and embeddings to the existing index and persist.

        Args:
            chunks: List of newly generated TextChunk objects.
            embeddings: 2D numpy array of shape (M, D) with normalized float32 vectors.
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            logger.warning("No chunks or embeddings provided to add_chunks.")
            return

        if self.index is None:
            self.build_index(chunks, embeddings)
        else:
            vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
            self.index.add(vectors)
            self.chunks_metadata.extend([c.to_dict() for c in chunks])
            logger.info(
                f"Added {len(chunks)} chunks to FAISS index. Total now: {self.index.ntotal}"
            )

        self.save()

    def get_documents_summary(self) -> List[Dict[str, Any]]:
        """
        Return structured per-document breakdown for all indexed documents,
        including automated clinical disease categorization and tags.

        Returns:
            List of dicts with doc_name, category, chunk_count, sample_text, etc.
        """
        if not self.chunks_metadata:
            return []

        from app.ingestion.categorizer import DiseaseCategorizer
        categorizer = DiseaseCategorizer()

        doc_map: Dict[str, Dict[str, Any]] = {}
        for chunk_data in self.chunks_metadata:
            meta = chunk_data.get("metadata", {})
            doc_name = meta.get("document_name", "unknown_document")
            chunk_text = chunk_data.get("text", "")
            
            if doc_name not in doc_map:
                doc_map[doc_name] = {
                    "document_name": doc_name,
                    "chunk_count": 0,
                    "total_chars": 0,
                    "file_type": meta.get("file_type", Path(doc_name).suffix.lstrip(".") or "txt"),
                    "sample_text": chunk_text[:300] + ("..." if len(chunk_text) > 300 else ""),
                    "full_text_sample": chunk_text,
                    "pages": set(),
                }
            
            doc_map[doc_name]["chunk_count"] += 1
            doc_map[doc_name]["total_chars"] += len(chunk_text)
            if len(doc_map[doc_name]["full_text_sample"]) < 15000:
                doc_map[doc_name]["full_text_sample"] += " " + chunk_text
            if "page_number" in meta:
                doc_map[doc_name]["pages"].add(meta["page_number"])

        result = []
        for doc_name, info in sorted(doc_map.items()):
            pages_list = sorted(list(info["pages"]))
            info["pages_count"] = len(pages_list) if pages_list else 1
            del info["pages"]

            # Add disease categorization
            cat_meta = categorizer.categorize_document(doc_name, sample_text=info["full_text_sample"])
            info["category"] = cat_meta["category_name"]
            info["category_icon"] = cat_meta["category_icon"]
            info["badge_bg"] = cat_meta["badge_bg"]
            info["badge_text"] = cat_meta["badge_text"]
            info["badge_border"] = cat_meta["badge_border"]
            info["disease_tags"] = cat_meta["tags"]
            info["confidence"] = cat_meta["confidence"]
            del info["full_text_sample"]

            result.append(info)

        return result

    def delete_document(self, document_name: str, embedder: Optional[Any] = None) -> bool:
        """
        Remove a document and its chunks from the vector store and re-index remaining.

        Args:
            document_name: The name of the document to delete.
            embedder: MedicalEmbedder instance to re-embed remaining chunks if needed.

        Returns:
            True if removed, False if document was not present.
        """
        if not self.chunks_metadata:
            return False

        remaining = [
            c for c in self.chunks_metadata
            if c.get("metadata", {}).get("document_name") != document_name
        ]

        if len(remaining) == len(self.chunks_metadata):
            logger.warning(f"Document '{document_name}' not found in vector store.")
            return False

        logger.info(f"Removing document '{document_name}'. Remaining chunks: {len(remaining)}")

        if not remaining:
            self.clear_index()
            return True

        if embedder is not None:
            # Reconstruct TextChunk objects
            chunk_objs = [
                TextChunk(
                    text=c.get("text", ""),
                    chunk_id=c.get("chunk_id", ""),
                    metadata=c.get("metadata", {}),
                )
                for c in remaining
            ]
            texts = [c.text for c in chunk_objs]
            new_embeddings = embedder.embed_documents(texts)
            self.build_index(chunk_objs, new_embeddings)
            self.save()
            return True
        else:
            # If no embedder passed, update metadata catalog and recreate index if possible
            self.chunks_metadata = remaining
            self.save()
            return True

    def clear_index(self) -> None:
        """Completely reset the FAISS index and metadata on disk."""
        self.index = None
        self.chunks_metadata = []
        if self.index_path.exists():
            try:
                self.index_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete index file {self.index_path}: {e}")
        if self.metadata_path.exists():
            try:
                self.metadata_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete metadata file {self.metadata_path}: {e}")
        logger.info("FAISS vector store successfully cleared.")
