"""
Text chunking module for medical document segmentation.
Implements sliding-window and sentence-aware chunking while preserving full source metadata.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from app.ingestion.loaders import Document
from app.utils.logging_config import setup_logger

logger = setup_logger("ingestion.chunker")


@dataclass
class TextChunk:
    """Represents a discrete semantic text chunk with traceable provenance."""
    text: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def document_name(self) -> str:
        return str(self.metadata.get("document_name", "unknown"))

    @property
    def page_number(self) -> int:
        return int(self.metadata.get("page_number", 1))

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "Medical Knowledge Base"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextChunk":
        """Reconstruct TextChunk from dictionary."""
        return cls(
            text=data["text"],
            chunk_id=data["chunk_id"],
            metadata=data.get("metadata", {}),
        )


class TextChunker:
    """
    Sentence-aware text chunker with configurable size and overlap.
    Preserves document identity, page numbers, and chunk coordinates.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Sentence splitting pattern preserving abbreviations like e.g., i.e., vs.
        self._sentence_splitter = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentence units."""
        paragraphs = text.split("\n\n")
        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            splits = self._sentence_splitter.split(para)
            for s in splits:
                s_clean = s.strip()
                if s_clean:
                    sentences.append(s_clean)
        return sentences

    def chunk_text(self, text: str, base_metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Split a string of text into overlapping TextChunk objects.

        Args:
            text: Text to chunk.
            base_metadata: Metadata dictionary to copy into each chunk.

        Returns:
            List of TextChunk instances.
        """
        if not text or not text.strip():
            return []

        base_meta = dict(base_metadata) if base_metadata else {}
        doc_name = str(base_meta.get("document_name", "doc"))
        # Sanitize document name for ID
        safe_name = re.sub(r"[^\w\-]", "_", doc_name.replace(".pdf", "").replace(".txt", "").replace(".md", ""))
        page_num = base_meta.get("page_number", 1)

        sentences = self._split_into_sentences(text)
        if not sentences:
            # Fallback if sentence splitter yields nothing
            sentences = [text.strip()]

        chunks: List[TextChunk] = []
        current_chunk_sentences: List[str] = []
        current_length = 0
        chunk_idx = 1

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size and we already have content
            if current_length + sentence_len > self.chunk_size and current_chunk_sentences:
                chunk_str = " ".join(current_chunk_sentences)
                chunk_id = f"{safe_name}_p{page_num}_c{chunk_idx:02d}"

                chunk_meta = dict(base_meta)
                chunk_meta.update({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "char_length": len(chunk_str),
                    "word_count": len(chunk_str.split()),
                })

                chunks.append(TextChunk(text=chunk_str, chunk_id=chunk_id, metadata=chunk_meta))
                chunk_idx += 1

                # Calculate overlap: retain trailing sentences up to chunk_overlap length
                overlap_sentences: List[str] = []
                overlap_len = 0
                for prev_s in reversed(current_chunk_sentences):
                    if overlap_len + len(prev_s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_s)
                        overlap_len += len(prev_s)
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_length = sum(len(s) for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_length += sentence_len

        # Final remaining chunk
        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            chunk_id = f"{safe_name}_p{page_num}_c{chunk_idx:02d}"
            chunk_meta = dict(base_meta)
            chunk_meta.update({
                "chunk_id": chunk_id,
                "chunk_index": chunk_idx,
                "char_length": len(chunk_str),
                "word_count": len(chunk_str.split()),
            })
            chunks.append(TextChunk(text=chunk_str, chunk_id=chunk_id, metadata=chunk_meta))

        return chunks

    def chunk_document(self, doc: Document) -> List[TextChunk]:
        """Chunk a single Document object."""
        return self.chunk_text(doc.text, base_metadata=doc.metadata)

    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        Chunk a collection of Document objects.

        Returns:
            Flat list of all generated TextChunk objects.
        """
        all_chunks: List[TextChunk] = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        logger.info(f"Generated {len(all_chunks)} chunks from {len(documents)} document pages/units.")
        return all_chunks
