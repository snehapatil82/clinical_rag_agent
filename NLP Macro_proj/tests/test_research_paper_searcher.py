"""
Unit tests for Web-Scale Research Paper Search Engine and Out-of-Knowledge-Base Fallback.
"""

import pytest
from app.retrieval.research_paper_searcher import MedicalResearchPaperSearcher, ResearchPaper
from app.rag.pipeline import ClinicalRAGPipeline


def test_research_paper_searcher_returns_5_papers():
    searcher = MedicalResearchPaperSearcher()
    query = "What are the early biomarkers of Alzheimer disease?"
    papers = searcher.search_papers(query, max_results=5)

    assert len(papers) == 5
    for p in papers:
        assert isinstance(p, ResearchPaper)
        assert len(p.title) > 5
        assert len(p.abstract) > 10
        assert p.url.startswith("http")
        assert 0.0 <= p.match_percentage <= 100.0

    # Ensure papers are sorted by match percentage descending
    match_percentages = [p.match_percentage for p in papers]
    assert match_percentages == sorted(match_percentages, reverse=True)


def test_research_paper_searcher_fallback_on_offline():
    searcher = MedicalResearchPaperSearcher()
    synthetic = searcher._generate_synthetic_fallback("CAR-T cell immunotherapy", max_results=5)
    assert len(synthetic) == 5
    assert "CAR-T cell immunotherapy" in synthetic[0].title or "Immunotherapy" in synthetic[0].title or "CAR-T" in synthetic[0].title
    assert synthetic[0].doi.startswith("10.")


def test_rag_pipeline_out_of_kb_covid_query():
    pipeline = ClinicalRAGPipeline()
    response = pipeline.query("what is covid?")

    assert response.status == "EXTERNAL_LITERATURE_FOUND"
    assert response.is_grounded is True
    assert response.is_external_literature is True
    assert "I don't have any documents in the knowledge base related to this question" in response.answer
    assert "here are some research papers you can refer to" in response.answer
    assert len(response.research_papers) == 5
    assert response.research_papers[0].match_percentage > 50.0
