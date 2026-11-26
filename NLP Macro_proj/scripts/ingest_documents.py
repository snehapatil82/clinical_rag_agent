"""
Document Ingestion CLI Script for Medical RAG.
Scans data/raw, extracts text from PDF/TXT/MD, cleans text, and generates chunks.
"""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.ingestion.loaders import DocumentLoader
from app.ingestion.cleaner import MedicalTextCleaner
from app.ingestion.chunker import TextChunker
from app.utils.logging_config import setup_logger

logger = setup_logger("scripts.ingest_documents")


def ingest_documents(raw_dir: Path, chunk_size: int, chunk_overlap: int):
    """Run full ingestion pipeline from raw documents to text chunks."""
    logger.info("=" * 60)
    logger.info("Starting Medical Document Ingestion Pipeline")
    logger.info(f"Source directory : {raw_dir}")
    logger.info(f"Chunk size       : {chunk_size} chars")
    logger.info(f"Chunk overlap    : {chunk_overlap} chars")
    logger.info("=" * 60)

    # 1. Load raw files
    loader = DocumentLoader(raw_dir=raw_dir)
    documents = loader.load_directory()
    if not documents:
        logger.error(f"No valid documents found in {raw_dir}. Please add PDF/TXT/MD files.")
        return []

    logger.info(f"Extracted {len(documents)} document pages/sections.")

    # 2. Clean text
    cleaner = MedicalTextCleaner()
    cleaned_docs = cleaner.clean_documents(documents)

    # 3. Chunk text
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(cleaned_docs)

    logger.info("=" * 60)
    logger.info(f"Ingestion complete: Created {len(chunks)} chunks from {len(documents)} document units.")
    logger.info("=" * 60)
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and chunk clinical documents.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=settings.DATA_RAW_DIR,
        help="Path to directory containing raw documents (PDF/TXT/MD)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.CHUNK_SIZE,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.CHUNK_OVERLAP,
        help="Chunk overlap in characters",
    )
    args = parser.parse_args()

    ingest_documents(args.raw_dir, args.chunk_size, args.chunk_overlap)
