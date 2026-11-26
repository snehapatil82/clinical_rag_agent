"""
Unit tests for Vector Store and Semantic Retrieval.
"""

from pathlib import Path
import pytest
from app.embeddings.embedder import MedicalEmbedder
from app.ingestion.chunker import TextChunk
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.retriever import ClinicalRetriever


@pytest.fixture(scope="module")
def populated_vector_store(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("test_vector_store")
    embedder = MedicalEmbedder()

    chunks = [
        TextChunk(
            text="Common asthma symptoms include wheezing, breathlessness, and nocturnal cough.",
            chunk_id="asthma_c1",
            metadata={"document_name": "asthma.pdf", "page_number": 1}
        ),
        TextChunk(
            text="Hypertension is defined as blood pressure >= 140/90 mmHg.",
            chunk_id="htn_c1",
            metadata={"document_name": "hypertension.pdf", "page_number": 1}
        ),
        TextChunk(
            text="Type 2 diabetes diagnostic criteria include fasting plasma glucose >= 126 mg/dL.",
            chunk_id="t2d_c1",
            metadata={"document_name": "diabetes.pdf", "page_number": 1}
        ),
    ]

    embeddings = embedder.embed_documents([c.text for c in chunks])
    store = FAISSVectorStore(store_dir=temp_dir, dimension=embedder.dimension)
    store.build_index(chunks, embeddings)
    store.save()
    return store, embedder


def test_retrieval_ranking(populated_vector_store):
    store, embedder = populated_vector_store
    retriever = ClinicalRetriever(vector_store=store, embedder=embedder, top_k=2)

    results = retriever.retrieve("What are the cardinal symptoms of asthma?")
    assert len(results) == 2
    assert results[0].document_name == "asthma.pdf"
    assert results[0].score > 0.50
    assert "wheezing" in results[0].text


def test_insufficient_evidence_threshold(populated_vector_store):
    store, embedder = populated_vector_store
    retriever = ClinicalRetriever(vector_store=store, embedder=embedder, similarity_threshold=0.45)

    ood_results = retriever.retrieve("How to bake chocolate chip cookies?")
    is_sufficient = retriever.is_evidence_sufficient(ood_results, threshold=0.45)
    assert not is_sufficient


def test_add_chunks_and_summary(populated_vector_store):
    store, embedder = populated_vector_store
    initial_total = store.index.ntotal

    new_chunk = TextChunk(
        text="Pneumonia treatment involves amoxicillin or macrolides for outpatients.",
        chunk_id="pneu_c1",
        metadata={"document_name": "pneumonia.pdf", "page_number": 2, "file_type": "pdf"}
    )
    new_vec = embedder.embed_documents([new_chunk.text])
    store.add_chunks([new_chunk], new_vec)

    assert store.index.ntotal == initial_total + 1
    summary = store.get_documents_summary()
    doc_names = [d["document_name"] for d in summary]
    assert "pneumonia.pdf" in doc_names

    # Test deletion
    deleted = store.delete_document("pneumonia.pdf", embedder=embedder)
    assert deleted is True
    assert store.index.ntotal == initial_total

