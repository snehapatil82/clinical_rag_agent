"""
Unit tests for Sentence Transformers Embedding Engine.
"""

import numpy as np
import pytest
from app.embeddings.embedder import MedicalEmbedder


@pytest.fixture(scope="module")
def embedder():
    return MedicalEmbedder()


def test_embedder_dimension_and_shape(embedder: MedicalEmbedder):
    assert embedder.dimension == 384

    vec = embedder.embed_text("Clinical symptoms of acute asthma exacerbation.")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_embedding_normalization(embedder: MedicalEmbedder):
    vec = embedder.embed_text("Diagnostic criteria for diabetes mellitus.")
    # Check L2 unit norm: ||v||_2 approx 1.0
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4


def test_embed_batch_documents(embedder: MedicalEmbedder):
    texts = [
        "First line antihypertensive agents include ACE inhibitors.",
        "Spirometry confirms reversible airflow limitation in asthma.",
        "Atherosclerosis is driven by subendothelial retention of LDL.",
    ]
    batch_vecs = embedder.embed_documents(texts)
    assert batch_vecs.shape == (3, 384)
    assert batch_vecs.dtype == np.float32
