"""
FAISS Vector Index Builder CLI Script.
Loads documents from data/raw, creates normalized embeddings with Sentence Transformers,
and serializes the FAISS index + metadata to vector_store/.
"""

import argparse
from pathlib import Path
import sys
import time

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.ingestion.loaders import DocumentLoader
from app.ingestion.cleaner import MedicalTextCleaner
from app.ingestion.chunker import TextChunker
from app.embeddings.embedder import MedicalEmbedder
from app.vectorstore.faiss_store import FAISSVectorStore
from app.utils.logging_config import setup_logger

logger = setup_logger("scripts.build_index")


def build_faiss_index(
    raw_dir: Path,
    vector_store_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
):
    """Orchestrates end-to-end vector index construction."""
    start_time = time.perf_counter()

    logger.info("=" * 65)
    logger.info("MEDICAL RAG: BUILDING FAISS VECTOR INDEX")
    logger.info(f"Raw Documents Dir : {raw_dir}")
    logger.info(f"Vector Store Dir  : {vector_store_dir}")
    logger.info(f"Embedding Model   : {model_name}")
    logger.info(f"Chunk Config      : size={chunk_size}, overlap={chunk_overlap}")
    logger.info("=" * 65)

    # Step 1: Load documents
    loader = DocumentLoader(raw_dir=raw_dir)
    documents = loader.load_directory()
    if not documents:
        logger.error(f"No documents found in {raw_dir}. Aborting index build.")
        sys.exit(1)

    # Step 2: Clean documents
    cleaner = MedicalTextCleaner()
    cleaned_docs = cleaner.clean_documents(documents)

    # Step 3: Chunk documents
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(cleaned_docs)
    if not chunks:
        logger.error("Chunking generated 0 chunks. Aborting index build.")
        sys.exit(1)

    logger.info(f"Generated {len(chunks)} total text chunks to be embedded.")

    # Step 4: Generate Sentence Transformer embeddings
    embedder = MedicalEmbedder(model_name=model_name)
    chunk_texts = [c.text for c in chunks]

    logger.info("Computing dense embeddings with L2 normalization...")
    embeddings = embedder.embed_documents(chunk_texts, batch_size=32, show_progress_bar=True)
    logger.info(f"Embeddings shape: {embeddings.shape} | Dtype: {embeddings.dtype}")

    # Step 5: Construct FAISS Index & Save
    vector_store = FAISSVectorStore(store_dir=vector_store_dir, dimension=embedder.dimension)
    vector_store.build_index(chunks=chunks, embeddings=embeddings)
    vector_store.save()

    elapsed = time.perf_counter() - start_time
    logger.info("=" * 65)
    logger.info("FAISS VECTOR INDEX BUILD COMPLETED SUCCESSFULLY")
    logger.info(f"Total Vectors Indexed : {embeddings.shape[0]}")
    logger.info(f"Vector Dimension      : {embeddings.shape[1]}")
    logger.info(f"Index File            : {vector_store.index_path}")
    logger.info(f"Metadata File         : {vector_store.metadata_path}")
    logger.info(f"Total Processing Time : {elapsed:.2f} seconds")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build persistent FAISS index from medical documents.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=settings.DATA_RAW_DIR,
        help="Path to directory containing source documents",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=settings.VECTOR_STORE_DIR,
        help="Path to directory where FAISS index will be saved",
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
    parser.add_argument(
        "--model-name",
        type=str,
        default=settings.EMBEDDING_MODEL_NAME,
        help="SentenceTransformer model name",
    )
    args = parser.parse_args()

    build_faiss_index(
        raw_dir=args.raw_dir,
        vector_store_dir=args.vector_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model_name,
    )
