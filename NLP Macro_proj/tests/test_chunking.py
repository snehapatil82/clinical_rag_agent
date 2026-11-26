"""
Unit tests for Text Chunking and Metadata Preservation.
"""

import pytest
from app.ingestion.loaders import Document
from app.ingestion.chunker import TextChunker, TextChunk


def test_chunker_basic_splitting():
    text = (
        "Hypertension is defined as persistent elevated blood pressure. "
        "Stage 1 is systolic 130-139 mmHg or diastolic 80-89 mmHg. "
        "Stage 2 is systolic >= 140 mmHg or diastolic >= 90 mmHg. "
        "First-line medications include Thiazide diuretics, ACE inhibitors, and ARBs. "
        "Lifestyle interventions include sodium restriction and DASH diet."
    )
    doc = Document(text=text, metadata={"document_name": "htn_guide.pdf", "page_number": 2})

    chunker = TextChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, TextChunk)
        assert c.document_name == "htn_guide.pdf"
        assert c.page_number == 2
        assert "chunk_id" in c.metadata
        assert len(c.text) > 0


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_chunk_serialization():
    chunk = TextChunk(
        text="Sample clinical text",
        chunk_id="doc1_p1_c01",
        metadata={"document_name": "doc1.pdf", "page_number": 1}
    )
    data = chunk.to_dict()
    assert data["chunk_id"] == "doc1_p1_c01"
    assert data["text"] == "Sample clinical text"

    reconstructed = TextChunk.from_dict(data)
    assert reconstructed.chunk_id == chunk.chunk_id
    assert reconstructed.text == chunk.text
    assert reconstructed.page_number == 1
