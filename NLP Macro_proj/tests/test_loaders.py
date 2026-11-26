"""
Unit tests for Document Ingestion and Loading.
"""

from pathlib import Path
import pytest
from app.ingestion.loaders import DocumentLoader, Document
from app.ingestion.cleaner import MedicalTextCleaner


def test_markdown_loader(tmp_path: Path):
    sample_file = tmp_path / "sample_guide.md"
    sample_file.write_text("# Clinical Guideline\n\nPatient with hypertension must follow DASH diet.", encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load_file(sample_file)

    assert len(docs) == 1
    assert "DASH diet" in docs[0].text
    assert docs[0].document_name == "sample_guide.md"
    assert docs[0].page_number == 1
    assert docs[0].metadata["file_type"] == "md"


def test_text_cleaner_preserves_medical_terms():
    cleaner = MedicalTextCleaner()
    raw = "Patient  BP:   140/90 mmHg  \n\n\n  FPG  >= 126 mg/dL  (HbA1c 6.5%).\x00"
    cleaned = cleaner.clean_text(raw)

    assert "140/90 mmHg" in cleaned
    assert "126 mg/dL" in cleaned
    assert "HbA1c 6.5%" in cleaned
    assert "\x00" not in cleaned
    assert "   " not in cleaned


def test_unsupported_file_format(tmp_path: Path):
    dummy_file = tmp_path / "unsupported.exe"
    dummy_file.write_text("binary content", encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load_file(dummy_file)
    assert len(docs) == 0


def test_load_text_direct():
    loader = DocumentLoader()
    content = "Asthma clinical protocol: Albuterol inhaler 2 puffs every 4-6 hours as needed."
    docs = loader.load_text(content, document_name="asthma_protocol.txt", source="Emergency Dept")

    assert len(docs) == 1
    assert "Albuterol inhaler" in docs[0].text
    assert docs[0].document_name == "asthma_protocol.txt"
    assert docs[0].metadata["source"] == "Emergency Dept"
    assert docs[0].metadata["file_type"] == "txt"


def test_load_bytes_markdown():
    loader = DocumentLoader()
    raw_bytes = b"# Cardiology Protocol\n\nAspirin 81mg daily for secondary prevention."
    docs = loader.load_bytes(raw_bytes, "cardiology.md")

    assert len(docs) == 1
    assert "Aspirin 81mg" in docs[0].text
    assert docs[0].document_name == "cardiology.md"
    assert docs[0].metadata["file_type"] == "md"

