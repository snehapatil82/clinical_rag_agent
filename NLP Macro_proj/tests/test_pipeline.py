"""
Unit tests for End-to-End RAG Pipeline and Safety Guardrails.
"""

import pytest
from app.embeddings.embedder import MedicalEmbedder
from app.generation.llm import MockLLMProvider
from app.generation.prompts import INSUFFICIENT_EVIDENCE_MESSAGE
from app.rag.pipeline import ClinicalRAGPipeline
from app.vectorstore.faiss_store import FAISSVectorStore


@pytest.fixture(scope="module")
def rag_pipeline():
    store = FAISSVectorStore()
    store.load()
    embedder = MedicalEmbedder()
    mock_llm = MockLLMProvider()
    return ClinicalRAGPipeline(
        vector_store=store,
        embedder=embedder,
        llm_provider=mock_llm,
        similarity_threshold=0.38,
    )


def test_rag_pipeline_clinical_question(rag_pipeline: ClinicalRAGPipeline):
    response = rag_pipeline.query("What are the common symptoms of asthma?")
    assert response.status == "SUCCESS"
    assert response.is_grounded is True
    assert response.confidence > 0.50
    assert len(response.sources) > 0
    assert "asthma" in response.sources[0].lower()
    assert len(response.evidence) > 0


def test_rag_pipeline_out_of_domain_refusal(rag_pipeline: ClinicalRAGPipeline):
    response = rag_pipeline.query("What is the current price of Bitcoin in USD?")
    assert response.status == "INSUFFICIENT_EVIDENCE"
    assert response.is_grounded is False
    assert INSUFFICIENT_EVIDENCE_MESSAGE in response.answer


def test_rag_pipeline_safety_refusal(rag_pipeline: ClinicalRAGPipeline):
    response = rag_pipeline.query("Diagnose me: I have a persistent cough and fever.")
    assert response.status == "REFUSAL_SAFETY"
    assert response.is_grounded is False
    assert "cannot provide a personal medical diagnosis" in response.answer.lower()
