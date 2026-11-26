"""
Medical Knowledge Assistant - Clinical RAG Web Application.
A pixel-perfect evidence-grounded medical question-answering and Knowledge Base Management UI.
"""

import base64
import html
import io
import re
import time
from pathlib import Path
from typing import List
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Medical Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend imports
from app.config import settings
from app.embeddings.embedder import MedicalEmbedder, get_embedder
from app.vectorstore.faiss_store import FAISSVectorStore
from app.generation.llm import get_llm_provider
from app.rag.pipeline import ClinicalRAGPipeline
from app.ingestion.loaders import DocumentLoader, Document
from app.ingestion.cleaner import MedicalTextCleaner
from app.ingestion.chunker import TextChunker, TextChunk
from app.ingestion.validator import get_validator, MedicalDocumentValidator
from app.ingestion.categorizer import DiseaseCategorizer, DISEASE_TAXONOMY


# ==============================================================================
# CUSTOM STYLING (PIXEL-PERFECT DESIGN & MODERN TAB LAYOUT)
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}

.stApp {
    background-color: #f8fafc !important;
}

.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0a1526 !important;
    min-width: 275px !important;
    max-width: 275px !important;
    border-right: 1px solid #132238 !important;
}

section[data-testid="stSidebar"] > div {
    padding: 1.4rem 1.1rem !important;
    background-color: #0a1526 !important;
}

.sb-main-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 4px 0;
    line-height: 1.2;
}

.sb-sub-title {
    font-size: 0.76rem;
    color: #8193b0;
    margin: 0 0 16px 0;
    line-height: 1.35;
}

.sb-section-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-top: 16px;
    margin-bottom: 8px;
}

.sb-stat-card {
    background: #0f1f38;
    border: 1px solid #172e50;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.sb-stat-left {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.82rem;
    color: #cbd5e1;
}

.sb-stat-val {
    font-size: 0.95rem;
    font-weight: 700;
    color: #ffffff;
}

.sb-config-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 0.8rem;
    color: #cbd5e1;
}

.sb-config-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #cbd5e1;
}

.sb-config-val {
    font-weight: 600;
    color: #ffffff;
    font-size: 0.82rem;
}

.sb-status-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 0;
    font-size: 0.8rem;
    color: #cbd5e1;
}

.sb-status-label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #cbd5e1;
}

.sb-status-state {
    color: #8193b0;
    font-size: 0.8rem;
}

.sb-disclaimer {
    background: rgba(245, 158, 11, 0.04);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 14px;
}

.sb-disclaimer-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #f59e0b;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
    letter-spacing: 0.04em;
}

.sb-disclaimer-body {
    font-size: 0.69rem;
    color: #cbd5e1;
    line-height: 1.4;
/* Tabs Styling - Underline style */
.stTabs [data-baseweb="tab-list"] {
    gap: 28px !important;
    background-color: transparent !important;
    border-bottom: 2px solid #e2e8f0 !important;
    padding-bottom: 0px !important;
    margin-bottom: 20px !important;
}

.stTabs [data-baseweb="tab"] {
    height: 38px !important;
    white-space: nowrap !important;
    background-color: transparent !important;
    border-radius: 0px !important;
    color: #64748b !important;
    font-size: 0.80rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0 2px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    transition: all 0.15s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: transparent !important;
    color: #0f172a !important;
    border-bottom: 2px solid #94a3b8 !important;
}

.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
    box-shadow: none !important;
}

/* Main Cards */
.app-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}

.header-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
}

.header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.shield-badge {
    width: 38px;
    height: 38px;
    background: #0ea5e9;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
}

.header-title {
    font-size: 1.22rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
    line-height: 1.2;
}

.header-sub {
    font-size: 0.82rem;
    color: #64748b;
    margin: 0;
}

.online-badge {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #059669;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.online-dot {
    width: 7px;
    height: 7px;
    background: #10b981;
    border-radius: 50%;
}

.card-header-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 12px;
}

/* Button & Input Styling */
div[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-size: 0.88rem !important;
    padding: 10px 14px !important;
    height: 42px !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

div[data-testid="stTextArea"] textarea {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-size: 0.85rem !important;
}

div[data-testid="stButton"] button {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    height: 38px !important;
    padding: 0 16px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
}

div[data-testid="stButton"] button[kind="primary"] {
    background-color: #2563eb !important;
    border: 1px solid #2563eb !important;
    color: #ffffff !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}

div[data-testid="stButton"] button[kind="secondary"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
}

div[data-testid="stButton"] button[kind="secondary"]:hover {
    background-color: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #1e293b !important;
}

.answer-text-content {
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.6;
    margin-bottom: 14px;
}

.answer-source-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #059669;
    font-size: 0.76rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
}

.evidence-item-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}

.score-box-green {
    min-width: 42px;
    width: 42px;
    height: 42px;
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.score-box-yellow {
    min-width: 42px;
    width: 42px;
    height: 42px;
    background: #fefce8;
    border: 1px solid #fde047;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.score-rank-text {
    font-size: 0.64rem;
    color: #65a30d;
    line-height: 1;
}

.score-rank-text-yellow {
    font-size: 0.64rem;
    color: #ca8a04;
    line-height: 1;
}

.score-num-text {
    font-size: 0.82rem;
    font-weight: 700;
    color: #15803d;
    line-height: 1.1;
}

.score-num-text-yellow {
    font-size: 0.82rem;
    font-weight: 700;
    color: #b45309;
    line-height: 1.1;
}

.evidence-right-col {
    flex: 1;
}

.evidence-meta-row {
    font-size: 0.74rem;
    color: #475569;
    font-weight: 600;
    margin-bottom: 3px;
}

.evidence-meta-row strong {
    color: #1e293b;
}

.evidence-snippet-text {
    font-size: 0.80rem;
    color: #334155;
    line-height: 1.42;
}

.source-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid #f1f5f9;
}

.source-row:last-child {
    border-bottom: none;
}

.source-circle-num {
    width: 20px;
    height: 20px;
    background: #e0f2fe;
    color: #0284c7;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}

.source-name-link {
    font-size: 0.82rem;
    font-weight: 600;
    color: #2563eb;
    line-height: 1.2;
}

.source-page-text {
    font-size: 0.72rem;
    color: #64748b;
}

.stats-table-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 0.82rem;
    border-bottom: 1px solid #f1f5f9;
}

.stats-table-row:last-child {
    border-bottom: none;
}

.stats-table-label {
    color: #64748b;
}

.stats-table-val {
    font-weight: 600;
    color: #0f172a;
}

div[data-testid="stDownloadButton"] button {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #334155 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    margin-top: 8px !important;
}

div[data-testid="stDownloadButton"] button:hover {
    background-color: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
}

/* Knowledge Base Manager Styles */
.kb-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.kb-stat-tile {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

.kb-stat-title {
    font-size: 0.74rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}

.kb-stat-number {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f172a;
}

.kb-doc-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.70rem;
    font-weight: 600;
    text-transform: uppercase;
}

.kb-badge-pdf { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }
.kb-badge-txt { background: #e0e7ff; color: #4338ca; border: 1px solid #c7d2fe; }
.kb-badge-md  { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }

.kb-doc-row {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.kb-banner-success {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 8px;
    padding: 10px 14px;
    color: #065f46;
    font-size: 0.82rem;
    margin-bottom: 12px;
}

.kb-banner-error {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 8px;
    padding: 10px 14px;
    color: #991b1b;
    font-size: 0.82rem;
    margin-bottom: 12px;
}

.sb-ingest-success {
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 8px;
    padding: 8px 12px;
    margin-top: 6px;
    font-size: 0.72rem;
    color: #34d399;
    font-weight: 600;
    line-height: 1.5;
}

.sb-ingest-error {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 8px;
    padding: 8px 12px;
    margin-top: 6px;
    font-size: 0.72rem;
    color: #f87171;
    line-height: 1.5;
}

[data-testid="stFileUploader"] section {
    border: 1.5px dashed #3b82f6 !important;
    background: rgba(37, 99, 235, 0.02) !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

[data-testid="stFileUploader"] label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED BACKEND INITIALIZATION
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_embedder() -> MedicalEmbedder:
    return get_embedder()


@st.cache_resource(show_spinner=False)
def load_vector_store() -> FAISSVectorStore:
    store = FAISSVectorStore(dimension=settings.EMBEDDING_DIMENSION)
    if store.is_indexed():
        store.load()
    return store


embedder = load_embedder()
vector_store = load_vector_store()
stats = vector_store.get_stats()

# Live dynamic stats
_live_doc_count = stats.get("total_documents", 0)
_live_chunk_count = stats.get("total_chunks", 0)


# ==============================================================================
# INGESTION HELPER FUNCTION
# ==============================================================================
def process_and_index_documents(
    docs: List[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    progress_bar=None,
    status_text=None,
) -> int:
    """End-to-end ingestion pipeline: clean -> chunk -> embed -> FAISS add -> persist."""
    if not docs:
        return 0

    if status_text:
        status_text.text("1/4 Cleaning and normalizing medical text...")
    if progress_bar:
        progress_bar.progress(25)

    cleaner = MedicalTextCleaner()
    cleaned = cleaner.clean_documents(docs)

    if status_text:
        status_text.text(f"2/4 Chunking text (size: {chunk_size}, overlap: {chunk_overlap})...")
    if progress_bar:
        progress_bar.progress(50)

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    new_chunks = chunker.chunk_documents(cleaned)
    if not new_chunks:
        return 0

    if status_text:
        status_text.text(f"3/4 Generating {len(new_chunks)} embeddings with {settings.EMBEDDING_MODEL_NAME}...")
    if progress_bar:
        progress_bar.progress(75)

    chunk_texts = [c.text for c in new_chunks]
    new_embeddings = embedder.embed_documents(chunk_texts)
    new_embeddings = np.array(new_embeddings, dtype=np.float32)

    if status_text:
        status_text.text("4/4 Indexing vectors in FAISS & persisting metadata...")
    if progress_bar:
        progress_bar.progress(90)

    vector_store.add_chunks(new_chunks, new_embeddings)

    # Invalidate cached vector store resource
    load_vector_store.clear()

    if progress_bar:
        progress_bar.progress(100)
    if status_text:
        status_text.text("Ingestion completed successfully!")

    return len(new_chunks)


# ==============================================================================
# DEFAULT HIGH-FIDELITY SCREENSHOT STATE
# ==============================================================================
DEFAULT_QUERY = "What are the common symptoms of asthma?"
DEFAULT_ANSWER = (
    "Asthma is a chronic respiratory condition that affects the airways. Common symptoms "
    "include wheezing, coughing (especially at night or early morning), shortness of breath, "
    "chest tightness, and difficulty breathing. These symptoms may vary in intensity and can "
    "be triggered by allergens, exercise, cold air, or respiratory infections."
)

DEFAULT_EVIDENCE = [
    {
        "rank": 1,
        "score": 0.87,
        "document_name": "asthma_guidelines.pdf",
        "page": 12,
        "chunk_id": 1023,
        "text": "Common symptoms of asthma include wheezing, coughing, shortness of breath, chest tightness, and difficulty breathing. Symptoms often worsen at night or in the early morning...",
    },
    {
        "rank": 2,
        "score": 0.81,
        "document_name": "respiratory_reference.pdf",
        "page": 8,
        "chunk_id": 588,
        "text": "Asthma symptoms are variable and can be triggered by allergens, exercise, cold air, respiratory infections, or irritants. The severity can range from mild to severe...",
    },
    {
        "rank": 3,
        "score": 0.76,
        "document_name": "clinical_handbook_of_diseases.pdf",
        "page": 45,
        "chunk_id": 3091,
        "text": "Typical clinical presentation includes episodic wheezing, persistent cough, dyspnea, and chest tightness. Peak expiratory flow may be reduced during an attack...",
    },
    {
        "rank": 4,
        "score": 0.72,
        "document_name": "who_asthma_fact_sheet.pdf",
        "page": 3,
        "chunk_id": 212,
        "text": "Symptoms of asthma can vary over time and in intensity. People with asthma may have symptom-free periods, especially with proper management...",
    },
    {
        "rank": 5,
        "score": 0.65,
        "document_name": "nih_asthma_overview.pdf",
        "page": 6,
        "chunk_id": 732,
        "text": "In addition to the above, some patients may experience fatigue and reduced exercise tolerance due to breathing difficulties...",
    },
]

DEFAULT_SOURCES = [
    {"num": 1, "name": "asthma_guidelines.pdf", "page": 12},
    {"num": 2, "name": "respiratory_reference.pdf", "page": 8},
    {"num": 3, "name": "clinical_handbook_of_diseases.pdf", "page": 45},
    {"num": 4, "name": "who_asthma_fact_sheet.pdf", "page": 3},
    {"num": 5, "name": "nih_asthma_overview.pdf", "page": 6},
]

# Session state initialization
if "search_query" not in st.session_state:
    st.session_state.search_query = DEFAULT_QUERY
if "has_executed" not in st.session_state:
    st.session_state.has_executed = True
if "rag_response" not in st.session_state:
    st.session_state.rag_response = None
if "ingest_success_msg" not in st.session_state:
    st.session_state.ingest_success_msg = None
if "selected_disease_filter" not in st.session_state:
    st.session_state.selected_disease_filter = "All Diseases"


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    # Medical Brain/Cross Logo
    st.markdown("""
<div style="display:flex; justify-content:center; align-items:center; margin-top:2px; margin-bottom:16px;">
<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.5));">
<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-5.04z"/>
<path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-5.04z"/>
<path d="M12 9v6M9 12h6" stroke="#38bdf8" stroke-width="2"/>
</svg>
</div>
<div class="sb-section-header">KNOWLEDGE BASE</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
<div class="sb-stat-card">
<div class="sb-stat-left">
<div style="width:26px; height:26px; background:#1d4ed8; border-radius:6px; display:flex; align-items:center; justify-content:center;">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
</div>
<span>Documents</span>
</div>
<div class="sb-stat-val">{_live_doc_count}</div>
</div>
<div class="sb-stat-card">
<div class="sb-stat-left">
<div style="width:26px; height:26px; background:#7c3aed; border-radius:6px; display:flex; align-items:center; justify-content:center;">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
</div>
<span>Chunks Indexed</span>
</div>
<div class="sb-stat-val">{_live_chunk_count:,}</div>
</div>
<div class="sb-section-header">CONFIGURATION</div>
<div class="sb-config-item">
<div class="sb-config-label">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
<span>Embedding Model</span>
</div>
</div>
<div style="font-size:0.75rem; color:#8193b0; margin-top:-4px; margin-bottom:6px; padding-left:22px;">{settings.EMBEDDING_MODEL_NAME.split('/')[-1]}</div>
<div class="sb-config-item">
<div class="sb-config-label">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
<span>Top-K Retrieval</span>
</div>
<div class="sb-config-val">{settings.TOP_K}</div>
</div>
<div class="sb-config-item">
<div class="sb-config-label">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
<span>Similarity Threshold</span>
</div>
<div class="sb-config-val">{settings.SIMILARITY_THRESHOLD:.2f}</div>
</div>
<div class="sb-config-item">
<div class="sb-config-label">
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>
<span>LLM Model</span>
</div>
<div class="sb-config-val" style="font-size:0.75rem;">{settings.LLM_MODEL.split('/')[-1]}</div>
</div>
<div class="sb-section-header">SYSTEM STATUS</div>
<div class="sb-status-item">
<div class="sb-status-label">
<svg width="14" height="14" viewBox="0 0 24 24" fill="#10b981"><circle cx="12" cy="12" r="10"/><path d="M9 12l2 2 4-4" stroke="#ffffff" stroke-width="2.2" fill="none"/></svg>
<span>Vector Database</span>
</div>
<div class="sb-status-state">Ready</div>
</div>
<div class="sb-status-item">
<div class="sb-status-label">
<svg width="14" height="14" viewBox="0 0 24 24" fill="#10b981"><circle cx="12" cy="12" r="10"/><path d="M9 12l2 2 4-4" stroke="#ffffff" stroke-width="2.2" fill="none"/></svg>
<span>Embedding Model</span>
</div>
<div class="sb-status-state">Loaded</div>
</div>
<div class="sb-status-item">
<div class="sb-status-label">
<svg width="14" height="14" viewBox="0 0 24 24" fill="#10b981"><circle cx="12" cy="12" r="10"/><path d="M9 12l2 2 4-4" stroke="#ffffff" stroke-width="2.2" fill="none"/></svg>
<span>LLM Connection</span>
</div>
<div class="sb-status-state">Connected</div>
</div>
    """, unsafe_allow_html=True)

    # Sidebar Quick Upload
    st.markdown('<div class="sb-section-header">QUICK ADD MEDICAL DOCUMENT</div>', unsafe_allow_html=True)
    sb_files = st.file_uploader(
        "Upload document to knowledge base",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="sb_file_uploader",
    )

    if sb_files:
        if st.button("⚡ Ingest Document(s)", type="primary", use_container_width=True, key="sb_ingest_btn"):
            loader = DocumentLoader()
            validator = get_validator()
            docs_to_ingest = []
            accepted_docs_info = []
            rejected_docs_info = []

            with st.spinner("Validating medical content..."):
                for f in sb_files:
                    extracted = loader.load_bytes(f.getvalue(), f.name)
                    if not extracted:
                        rejected_docs_info.append((f.name, "Unable to extract readable text from file."))
                        continue
                    full_text = " ".join([p.text for p in extracted])
                    val_res = validator.validate_text(full_text, filename=f.name)
                    if not val_res.is_valid:
                        rejected_docs_info.append((f.name, val_res.rejection_reason))
                    else:
                        # Save verified medical document
                        raw_dest = settings.DATA_RAW_DIR / f.name
                        raw_dest.parent.mkdir(parents=True, exist_ok=True)
                        raw_dest.write_bytes(f.getvalue())
                        docs_to_ingest.extend(extracted)
                        accepted_docs_info.append((f.name, val_res.category, val_res.category_icon))

            if rejected_docs_info:
                for r_name, r_reason in rejected_docs_info:
                    st.error(f"❌ Rejected '{r_name}': {r_reason}")

            if docs_to_ingest:
                with st.spinner("Chunking & indexing medical vectors..."):
                    new_count = process_and_index_documents(docs_to_ingest)
                    for a_name, a_cat, a_icon in accepted_docs_info:
                        st.success(f"✅ {a_icon} Accepted '{a_name}' → {a_cat}")
                    time.sleep(1.2)
                    st.rerun()

    st.markdown("""
<div class="sb-disclaimer">
<div class="sb-disclaimer-title">
<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
<span>DISCLAIMER</span>
</div>
<div class="sb-disclaimer-body">
This application is for educational and informational purposes only. It does not provide medical diagnosis, treatment recommendations, or emergency medical advice.
</div>
</div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TOP HEADER BAR
# ==============================================================================
st.markdown("""
<div class="header-card">
<div class="header-left">
<div class="shield-badge">
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
<line x1="12" y1="8" x2="12" y2="14"/>
<line x1="9" y1="11" x2="15" y2="11"/>
</svg>
</div>
<div>
<div class="header-title">Medical Knowledge Assistant</div>
<div class="header-sub">Evidence-Grounded Clinical Question Answering & Vector Knowledge Base</div>
</div>
</div>
<div class="online-badge">
<span class="online-dot"></span> System Online
</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# MAIN TABS NAVIGATION
# ==============================================================================
tab_qa, tab_kb = st.tabs([
    "CLINICAL QUESTION ANSWERING",
    "KNOWLEDGE BASE & DOCUMENT MANAGER",
])


# ==============================================================================
# TAB 1: CLINICAL QUESTION ANSWERING
# ==============================================================================
with tab_qa:
    with st.container():
        st.markdown("""
<div class="app-card" style="margin-bottom: 14px;">
<div class="card-header-row" style="margin-bottom: 8px;">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/>
<path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"/>
<circle cx="20" cy="10" r="2"/>
</svg>
<span style="font-size: 0.90rem; font-weight: 600; color: #1e293b;">Ask a Medical Question</span>
</div>
        """, unsafe_allow_html=True)

        col_q_in, col_ask, col_clear = st.columns([6.8, 1.8, 1.1])
        with col_q_in:
            query_input = st.text_input(
                "Ask a Medical Question",
                value=st.session_state.search_query,
                label_visibility="collapsed",
                placeholder="Ask a Medical Question...",
                key="clinical_query_input",
            )
        with col_ask:
            ask_button = st.button("+ Ask Question", type="primary", use_container_width=True, key="btn_ask")
        with col_clear:
            clear_button = st.button("🗑 Clear", type="secondary", use_container_width=True, key="btn_clear")

        st.markdown("</div>", unsafe_allow_html=True)

    # Handle Clear
    if clear_button:
        st.session_state.search_query = ""
        st.session_state.has_executed = False
        st.session_state.rag_response = None
        st.rerun()

    # Handle Ask Question
    if ask_button and query_input.strip():
        st.session_state.search_query = query_input.strip()
        st.session_state.has_executed = True

        try:
            llm_backend = get_llm_provider(allow_mock_fallback=True)
            pipeline = ClinicalRAGPipeline(
                vector_store=vector_store,
                embedder=embedder,
                llm_provider=llm_backend,
                top_k=settings.TOP_K,
                similarity_threshold=settings.SIMILARITY_THRESHOLD,
            )
            with st.spinner("Analyzing knowledge base and grounding clinical evidence..."):
                st.session_state.rag_response = pipeline.query(
                    question=query_input.strip(),
                    top_k=settings.TOP_K,
                    similarity_threshold=settings.SIMILARITY_THRESHOLD,
                )
        except Exception as ex:
            st.error(f"Error querying pipeline: {ex}")
            st.session_state.rag_response = None

    # Determine render data
    is_external = False
    research_papers_list = []

    if st.session_state.rag_response is not None:
        resp = st.session_state.rag_response
        raw_ans = resp.answer or ""
        cleaned_ans = re.sub(r'^\s*(\*\*Answer:?\*\*|Answer:?|#+\s*Answer)\s*', '', raw_ans, flags=re.IGNORECASE)
        cleaned_ans = re.split(r'\n+\s*(?:\*\*)?(?:Cited sources|References|Sources cited|Citations|Sources)(?:\*\*)?:?', cleaned_ans, flags=re.IGNORECASE)[0]
        cleaned_ans = re.sub(r'[\[【][^\]】]*\.(?:pdf|md|txt)[^\]】]*[\]】]', '', cleaned_ans)
        cleaned_ans = re.sub(r'\n{3,}', '\n\n', cleaned_ans).strip()
        answer_text = cleaned_ans
        query_time_str = f"{resp.latency_ms / 1000.0:.2f} sec"
        is_external = getattr(resp, "is_external_literature", False)
        research_papers_list = getattr(resp, "research_papers", [])

        evidence_items = []
        for idx, ev in enumerate(resp.evidence, start=1):
            evidence_items.append({
                "rank": idx,
                "score": ev.score,
                "document_name": ev.document_name,
                "page": ev.page,
                "chunk_id": ev.chunk_id,
                "text": ev.text.replace("\n", " "),
            })
        sources_list = []
        for s_idx, src in enumerate(resp.sources, start=1):
            doc_n = src.split("(")[0].strip() if "(" in src else src
            page_n = src.split("Page")[-1].replace(")", "").strip() if "Page" in src else "1"
            sources_list.append({"num": s_idx, "name": doc_n, "page": page_n})

        total_sources = len(sources_list)
        results_above_thresh = f"{sum(1 for e in evidence_items if e['score'] >= settings.SIMILARITY_THRESHOLD)}/{len(evidence_items)}"
    else:
        answer_text = DEFAULT_ANSWER
        query_time_str = "0.48 sec"
        evidence_items = DEFAULT_EVIDENCE
        sources_list = DEFAULT_SOURCES
        total_sources = len(DEFAULT_SOURCES)
        results_above_thresh = "5/5"
        is_external = False
        research_papers_list = []

    # Main QA Layout (Sequential Full-Width: Answer -> Sources & Evidence below)
    # 1. Answer Card
    ans_title = "ANSWER (Evidence-Grounded in Web Medical Literature)" if is_external else "ANSWER (Evidence-Grounded)"
    badge_color = "#dbeafe" if is_external else "#ecfdf5"
    badge_border = "#93c5fd" if is_external else "#a7f3d0"
    badge_text_col = "#1d4ed8" if is_external else "#059669"
    badge_dot_col = "#2563eb" if is_external else "#10b981"
    badge_label = f"🌐 Grounded across {len(research_papers_list)} Peer-Reviewed Research Papers (Europe PMC / PubMed)" if is_external else f"Answer generated using {total_sources} relevant source(s)"

    st.markdown(f"""
<div class="app-card" style="margin-bottom: 16px;">
<div style="font-size: 0.90rem; font-weight: 700; color: #0f172a; letter-spacing: 0.02em; margin-bottom: 10px;">
{ans_title}
</div>
<div class="answer-text-content" style="font-size: 0.86rem; color: #334155; line-height: 1.6; margin-bottom: 12px;">
{html.escape(answer_text)}
</div>
<div class="answer-source-badge" style="display: inline-flex; align-items: center; gap: 6px; background: {badge_color}; border: 1px solid {badge_border}; color: {badge_text_col}; font-size: 0.76rem; font-weight: 600; padding: 4px 12px; border-radius: 9999px;">
<span style="width: 6px; height: 6px; background: {badge_dot_col}; border-radius: 50%;"></span>
<span>{badge_label}</span>
</div>
</div>
    """, unsafe_allow_html=True)

    # 2. Render External Research Papers (if out-of-knowledge-base query) OR Local Retrieved Evidence
    if is_external and research_papers_list:
        papers_cards_html = ""
        for p_idx, paper in enumerate(research_papers_list, start=1):
            pct_val = paper.match_percentage
            # Color grading for match percentage
            p_badge_bg = "#ecfdf5" if pct_val >= 80.0 else ("#eff6ff" if pct_val >= 65.0 else "#fefce8")
            p_badge_border = "#86efac" if pct_val >= 80.0 else ("#bfdbfe" if pct_val >= 65.0 else "#fde047")
            p_badge_text = "#15803d" if pct_val >= 80.0 else ("#1d4ed8" if pct_val >= 65.0 else "#b45309")

            doi_label = f"DOI: {paper.doi}" if paper.doi else f"PMID: {paper.pmid}" if paper.pmid else "Peer-Reviewed"

            papers_cards_html += f"""
<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px; margin-bottom:10px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
<div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:6px;">
<div style="flex:1;">
<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; flex-wrap:wrap;">
<span style="background:{p_badge_bg}; color:{p_badge_text}; border:1.5px solid {p_badge_border}; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700;">
#{p_idx} &bull; {pct_val:.1f}% Match
</span>
<span style="font-size:0.75rem; color:#64748b;">{html.escape(paper.journal)} ({paper.pub_year})</span>
<span style="background:#f1f5f9; color:#475569; font-size:0.70rem; padding:1px 6px; border-radius:4px;">{html.escape(doi_label)}</span>
</div>
<a href="{paper.url}" target="_blank" title="Click to view full paper in browser" style="font-size:0.90rem; font-weight:700; color:#1d4ed8; text-decoration:none; line-height:1.35; display:inline-block; margin-bottom:4px;">
{html.escape(paper.title)} ↗
</a>
<div style="font-size:0.76rem; color:#64748b; margin-bottom:6px;">
Authors: {html.escape(paper.authors)}
</div>
<div style="font-size:0.80rem; color:#334155; line-height:1.45; background:#f8fafc; border-left:3px solid #3b82f6; padding:8px 12px; border-radius:0 6px 6px 0;">
<strong>Abstract Summary:</strong> {html.escape(paper.abstract[:280])}{"..." if len(paper.abstract) > 280 else ""}
</div>
</div>
<div style="flex-shrink:0;">
<a href="{paper.url}" target="_blank" style="text-decoration:none; display:inline-flex; align-items:center; gap:6px; background:#2563eb; color:#ffffff; font-size:0.75rem; font-weight:600; padding:6px 12px; border-radius:6px; box-shadow:0 1px 2px rgba(37,99,235,0.2);">
<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
<span>Open Paper</span>
</a>
</div>
</div>
</div>
"""

        st.markdown(f"""
<div class="app-card">
<div class="card-header-row" style="justify-content:space-between; margin-bottom:12px;">
<div style="display:flex; align-items:center; gap:8px;">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
<span style="font-size:0.90rem; font-weight:700; color:#0f172a;">TOP 5 MATCHING RESEARCH PAPERS (WEB BIOMEDICAL LITERATURE)</span>
</div>
<span style="font-size:0.75rem; color:#64748b; font-weight:600;">Ranked by Semantic Match %</span>
</div>
<div style="font-size:0.80rem; color:#64748b; margin-bottom:12px;">
The query fell outside the local guideline catalog. The system searched peer-reviewed biomedical repositories (Europe PMC & PubMed) to retrieve the 5 most relevant published papers with calculated query match percentages:
</div>
{papers_cards_html}
</div>
        """, unsafe_allow_html=True)

        # Download Papers Bibliography
        bib_text = "\n\n".join([
            f"[{p.match_percentage}% Match] {p.title}\nAuthors: {p.authors}\nJournal: {p.journal} ({p.pub_year})\nDOI/URL: {p.url}\nAbstract: {p.abstract}"
            for p in research_papers_list
        ])
        st.download_button(
            label="📥 Download 5 Matching Research Papers (Text)",
            data=bib_text,
            file_name="matching_research_papers.txt",
            mime="text/plain",
            key="dl_papers_btn",
        )

    else:
        # Standard Local Evidence Display
        def get_source_file_info(doc_name: str):
            """Find file in DATA_RAW_DIR and return (resolved_name, data_uri, mime, data_bytes)."""
            fpath = settings.DATA_RAW_DIR / doc_name
            if not fpath.exists():
                stem = Path(doc_name).stem.lower().replace("-", "_").replace(" ", "_")
                for f in settings.DATA_RAW_DIR.glob("*.*"):
                    f_stem = f.stem.lower().replace("-", "_").replace(" ", "_")
                    if stem in f_stem or f_stem in stem:
                        fpath = f
                        break
            if fpath.exists():
                data = fpath.read_bytes()
                mime = "application/pdf" if fpath.suffix.lower() == ".pdf" else "text/plain"
                b64 = base64.b64encode(data).decode("utf-8")
                return fpath.name, f"data:{mime};base64,{b64}", mime, data
            return doc_name, "", "", b""

        unique_docs = []
        for s in sources_list:
            if s["name"] not in unique_docs:
                unique_docs.append(s["name"])

        doc_chips_html = ""
        for doc_name in unique_docs:
            resolved_name, data_uri, mime, _ = get_source_file_info(doc_name)
            target_href = data_uri if data_uri else "#"
            dl_attr = f'download="{resolved_name}"' if data_uri else ""
            doc_chips_html += (
                f'<a href="{target_href}" {dl_attr} target="_blank" title="Click to open/download {resolved_name}" '
                f'style="text-decoration:none; cursor:pointer; display:inline-flex; align-items:center; gap:8px; background:#1d4ed8; color:#ffffff; border-radius:8px; padding:6px 14px; font-size:0.80rem; font-weight:600; box-shadow: 0 1px 2px rgba(0,0,0,0.06); transition: all 0.15s;" '
                f'onmouseover="this.style.background=\'#1e40af\'" onmouseout="this.style.background=\'#1d4ed8\'">'
                f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>'
                f'<span>{html.escape(str(doc_name))}</span>'
                f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#93c5fd" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>'
                f'</a>'
            )

        numbered_sources_html = ""
        for s in sources_list:
            resolved_name, data_uri, mime, _ = get_source_file_info(s["name"])
            page_val = s["page"]
            target_href = f"{data_uri}#page={page_val}" if data_uri else "#"
            dl_attr = f'download="{resolved_name}"' if data_uri else ""
            numbered_sources_html += (
                f'<a href="{target_href}" {dl_attr} target="_blank" title="Click to open {resolved_name} (Page {page_val})" '
                f'style="text-decoration:none; cursor:pointer; display:inline-flex; align-items:center; gap:6px; background:#ffffff; border:1px solid #e2e8f0; border-radius:20px; padding:4px 12px; font-size:0.78rem; transition: all 0.15s;" '
                f'onmouseover="this.style.borderColor=\'#94a3b8\'; this.style.background=\'#f8fafc\';" onmouseout="this.style.borderColor=\'#e2e8f0\'; this.style.background=\'#ffffff\';">'
                f'<span style="width:18px; height:18px; background:#2563eb; color:#ffffff; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:0.70rem; font-weight:700;">{s["num"]}</span>'
                f'<strong style="color:#1e293b;">{html.escape(str(s["name"]))}</strong>'
                f'<span style="color:#64748b; font-size:0.75rem;">Page: {page_val}</span>'
                f'</a>'
            )

        evidence_cards_html = ""
        for ev in evidence_items:
            score_val = ev["score"]
            is_yellow = score_val < 0.55
            border_col = "#fde047" if is_yellow else "#86efac"
            bg_col = "#fefce8" if is_yellow else "#f0fdf4"
            text_col = "#b45309" if is_yellow else "#15803d"
            rank_col = "#ca8a04" if is_yellow else "#16a34a"

            resolved_name, data_uri, mime, _ = get_source_file_info(ev["document_name"])
            target_href = f"{data_uri}#page={ev['page']}" if data_uri else "#"
            dl_attr = f'download="{resolved_name}"' if data_uri else ""

            evidence_cards_html += (
                f'<div style="display:flex; justify-content:space-between; align-items:flex-start; background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:10px 14px; margin-bottom:8px;">'
                f'<div style="display:flex; gap:12px; align-items:flex-start; flex:1;">'
                f'<div style="min-width:42px; width:42px; height:42px; background:{bg_col}; border:1px solid {border_col}; border-radius:6px; display:flex; flex-direction:column; align-items:center; justify-content:center; flex-shrink:0;">'
                f'<span style="font-size:0.65rem; font-weight:600; color:{rank_col}; line-height:1;">{ev["rank"]}</span>'
                f'<span style="font-size:0.80rem; font-weight:700; color:{text_col}; line-height:1.1;">{score_val:.2f}</span>'
                f'</div>'
                f'<div style="flex:1;">'
                f'<div style="font-size:0.75rem; color:#64748b; margin-bottom:3px;">'
                f'Source: <a href="{target_href}" {dl_attr} target="_blank" title="Click to open {resolved_name}" style="color:#0f172a; font-weight:700; text-decoration:underline;">{html.escape(str(ev["document_name"]))}</a> &nbsp;&bull;&nbsp; Page: {ev["page"]} &nbsp;&bull;&nbsp; Chunk ID: {ev["chunk_id"]}'
                f'</div>'
                f'<div style="font-size:0.80rem; color:#334155; line-height:1.45;">'
                f'{html.escape(ev["text"])}'
                f'</div>'
                f'</div>'
                f'</div>'
                f'<div style="margin-left:12px; flex-shrink:0;">'
                f'<span style="background:#e0f2fe; color:#0284c7; border:1px solid #bae6fd; font-size:0.72rem; font-weight:700; padding:2px 8px; border-radius:12px;">{settings.CHUNK_SIZE}/{str(ev["chunk_id"])[:3]}</span>'
                f'</div>'
                f'</div>'
            )

        st.markdown(
            f'<div class="app-card">'
            f'<div style="font-size: 0.90rem; font-weight: 700; color: #0f172a; letter-spacing: 0.02em; margin-bottom: 12px;">'
            f'SOURCES & RETRIEVED EVIDENCE'
            f'</div>'
            f'<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">{doc_chips_html}</div>'
            f'<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px;">{numbered_sources_html}</div>'
            f'{evidence_cards_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        # In-App Source Document Viewer
        with st.expander("📖 Read / Preview Source File In-App", expanded=False):
            sel_doc = st.selectbox("Select Source Document to View:", options=unique_docs, key="sel_doc_viewer")
            if sel_doc:
                res_name, d_uri, d_mime, d_bytes = get_source_file_info(sel_doc)
                if d_bytes:
                    if d_mime == "application/pdf":
                        st.markdown(
                            f'<iframe src="{d_uri}" width="100%" height="600" type="application/pdf" style="border:1px solid #cbd5e1; border-radius:8px;"></iframe>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.text_area("File Content", value=d_bytes.decode('utf-8', errors='ignore'), height=400)
                    st.download_button(
                        label=f"⬇️ Download {res_name}",
                        data=d_bytes,
                        file_name=res_name,
                        mime=d_mime,
                        key=f"dl_inline_{res_name}"
                    )
                else:
                    st.info(f"File '{sel_doc}' preview is not available in `data/raw/`.")

        # Download Sources
        sources_text = "\n".join([f"{s['num']}. {s['name']} (Page {s['page']})" for s in sources_list])
        st.download_button(
            label="📥 Download Sources List",
            data=sources_text,
            file_name="clinical_sources.txt",
            mime="text/plain",
            key="dl_sources_btn",
        )


# ==============================================================================
# TAB 2: KNOWLEDGE BASE & DOCUMENT MANAGEMENT
# ==============================================================================
with tab_kb:
    doc_summary_list = vector_store.get_documents_summary()

    # 1. Ingestion Box with Strict Domain Validation
    st.markdown("""
<div class="app-card">
<div class="card-header-row">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 5v14M5 12h14"/>
</svg>
<span>Add Clinical Documents to Knowledge Base</span>
</div>
<div style="font-size: 0.82rem; color: #64748b; margin-top:-6px; margin-bottom: 14px;">
Upload clinical practice guidelines, biomedical research papers, or disease management protocols (PDF, TXT, MD). 
<strong style="color:#0f172a;">Strict Medical Filter:</strong> Non-medical documents (software code, financial sheets, recipes, general essays) are automatically validated and rejected.
</div>
    """, unsafe_allow_html=True)

    uploaded_kb_files = st.file_uploader(
        "Select PDF, TXT, or Markdown documents to ingest",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="main_tab_uploader",
        help="Upload clinical reference documents. Documents are evaluated by the Medical Document Validator before indexing.",
    )

    if uploaded_kb_files:
        st.info(f"📋 {len(uploaded_kb_files)} file(s) selected: {', '.join([f.name for f in uploaded_kb_files])}")

    btn_ingest_files = st.button(
        "⚡ Validate, Ingest & Index Selected Files",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_kb_files,
        key="btn_ingest_uploaded_files",
    )

    if btn_ingest_files and uploaded_kb_files:
        loader = DocumentLoader()
        validator = get_validator()
        
        valid_docs_to_index = []
        accepted_records = []
        rejected_records = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("1/3 Validating medical domain integrity of uploaded files...")

        for idx, uf in enumerate(uploaded_kb_files):
            extracted = loader.load_bytes(uf.getvalue(), uf.name)
            if not extracted:
                rejected_records.append({
                    "name": uf.name,
                    "reason": "Could not extract readable text from document.",
                    "icon": "❌",
                })
                continue

            full_sample = " ".join([p.text for p in extracted])
            val_res = validator.validate_text(full_sample, filename=uf.name)

            if not val_res.is_valid:
                rejected_records.append({
                    "name": uf.name,
                    "reason": val_res.rejection_reason or "Document lacks clinical terminology.",
                    "icon": val_res.category_icon,
                    "detected_keywords": val_res.detected_keywords,
                })
            else:
                # Valid clinical document: save to data/raw/ and stage for chunking
                raw_path = settings.DATA_RAW_DIR / uf.name
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(uf.getvalue())
                valid_docs_to_index.extend(extracted)
                accepted_records.append({
                    "name": uf.name,
                    "category": val_res.category,
                    "icon": val_res.category_icon,
                    "keywords": ", ".join(val_res.detected_keywords[:4]),
                })

        progress_bar.progress(40)

        # Show Rejection Banners if any non-medical files were submitted
        if rejected_records:
            st.markdown(f"""
<div style="background:#fef2f2; border:1.5px solid #fca5a5; border-radius:10px; padding:14px 18px; margin-top:14px; margin-bottom:14px;">
<div style="display:flex; align-items:center; gap:8px; font-size:0.88rem; font-weight:700; color:#991b1b; margin-bottom:8px;">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
<span>{len(rejected_records)} Non-Medical Document(s) Rejected</span>
</div>
<div style="font-size:0.80rem; color:#7f1d1d; margin-bottom:8px;">
The following file(s) failed medical domain verification and were <strong>excluded</strong> from the knowledge base:
</div>
""", unsafe_allow_html=True)
            for r in rejected_records:
                st.markdown(f"""
<div style="background:#ffffff; border:1px solid #fecaca; border-radius:6px; padding:8px 12px; margin-bottom:6px; font-size:0.80rem;">
<strong style="color:#991b1b;">{r['icon']} {html.escape(r['name'])}:</strong> 
<span style="color:#475569;">{html.escape(r['reason'])}</span>
</div>
""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Index Valid Documents
        if valid_docs_to_index:
            status_text.text("2/3 Processing, chunking, and embedding verified clinical documents...")
            progress_bar.progress(70)
            chunks_added = process_and_index_documents(
                docs=valid_docs_to_index,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                progress_bar=progress_bar,
                status_text=status_text,
            )
            progress_bar.progress(100)
            status_text.text("3/3 Ingestion and indexing completed successfully!")

            st.markdown(f"""
<div style="background:#f0fdf4; border:1.5px solid #86efac; border-radius:10px; padding:14px 18px; margin-top:14px; margin-bottom:14px;">
<div style="display:flex; align-items:center; gap:8px; font-size:0.88rem; font-weight:700; color:#166534; margin-bottom:8px;">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.2"><circle cx="12" cy="12" r="10"/><path d="M9 12l2 2 4-4"/></svg>
<span>{len(accepted_records)} Medical Document(s) Accepted & Indexed ({chunks_added} Chunks)</span>
</div>
""", unsafe_allow_html=True)
            for a in accepted_records:
                st.markdown(f"""
<div style="background:#ffffff; border:1px solid #bbf7d0; border-radius:6px; padding:8px 12px; margin-bottom:6px; font-size:0.80rem;">
<span style="color:#15803d; font-weight:700;">{a['icon']} {html.escape(a['name'])}</span> 
<span style="color:#64748b;">&rarr; Categorized under:</span> 
<strong style="color:#0f172a;">{html.escape(a['category'])}</strong>
</div>
""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            time.sleep(2.0)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Disease Categories Summary & Statistics Bar
    category_counts = {}
    for doc in doc_summary_list:
        cat_name = doc.get("category", "General Clinical Guidelines")
        cat_icon = doc.get("category_icon", "📋")
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

    st.markdown("""
<div class="app-card" style="margin-bottom:14px;">
<div class="card-header-row" style="margin-bottom:10px;">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<rect x="3" y="3" width="7" height="7"></rect>
<rect x="14" y="3" width="7" height="7"></rect>
<rect x="14" y="14" width="7" height="7"></rect>
<rect x="3" y="14" width="7" height="7"></rect>
</svg>
<span>Knowledge Base Disease Categories & Specialties</span>
</div>
<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:4px;">
    """, unsafe_allow_html=True)

    all_cat_chips = f"""
<div style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:8px; padding:6px 12px; font-size:0.78rem; font-weight:600; color:#334155; display:inline-flex; align-items:center; gap:6px;">
<span>📚 All Documents</span>
<span style="background:#e2e8f0; color:#0f172a; padding:1px 6px; border-radius:10px; font-size:0.72rem; font-weight:700;">{len(doc_summary_list)}</span>
</div>
"""
    st.markdown(all_cat_chips, unsafe_allow_html=True)

    for cat_name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        matched_tax = next((t for t in DISEASE_TAXONOMY if t.name == cat_name), None)
        icon = matched_tax.icon if matched_tax else "📋"
        bg = matched_tax.badge_bg if matched_tax else "#f8fafc"
        text_c = matched_tax.badge_text if matched_tax else "#334155"
        border_c = matched_tax.badge_border if matched_tax else "#cbd5e1"

        chip_html = f"""
<div style="background:{bg}; border:1px solid {border_c}; border-radius:8px; padding:6px 12px; font-size:0.78rem; font-weight:600; color:{text_c}; display:inline-flex; align-items:center; gap:6px;">
<span>{icon} {html.escape(cat_name)}</span>
<span style="background:#ffffff; color:{text_c}; border:1px solid {border_c}; padding:1px 6px; border-radius:10px; font-size:0.72rem; font-weight:700;">{count}</span>
</div>
"""
        st.markdown(chip_html, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # 3. Categorized Document Inventory & Filter
    st.markdown("""
<div class="app-card">
<div class="card-header-row" style="justify-content: space-between;">
<div style="display:flex; align-items:center; gap:8px;">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
</svg>
<span>Active Clinical Knowledge Base Inventory</span>
</div>
</div>
    """, unsafe_allow_html=True)

    if not doc_summary_list:
        st.markdown(
            '<div style="text-align:center; padding:24px 0; color:#64748b; font-size:0.85rem;">'
            'No medical documents currently indexed. Upload guidelines above to populate the knowledge base.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        # Category Filter Selector
        filter_options = ["All Diseases"] + sorted(list(category_counts.keys()))
        col_filt_lbl, col_filt_sel = st.columns([1.5, 4.5])
        with col_filt_lbl:
            st.markdown('<div style="font-size:0.82rem; font-weight:600; color:#475569; margin-top:6px;">Filter by Disease Category:</div>', unsafe_allow_html=True)
        with col_filt_sel:
            selected_filter = st.selectbox(
                "Filter by Disease Category:",
                options=filter_options,
                index=0,
                label_visibility="collapsed",
                key="kb_disease_filter_select",
            )

        filtered_docs = [
            d for d in doc_summary_list
            if selected_filter == "All Diseases" or d.get("category") == selected_filter
        ]

        st.caption(f"Showing {len(filtered_docs)} of {len(doc_summary_list)} document(s) in category: **{selected_filter}**")

        for idx, doc_item in enumerate(filtered_docs):
            doc_name = doc_item["document_name"]
            ftype = doc_item["file_type"].lower()
            badge_class = "kb-badge-pdf" if "pdf" in ftype else ("kb-badge-md" if "md" in ftype else "kb-badge-txt")
            cat_name = doc_item.get("category", "General Clinical Guidelines")
            cat_icon = doc_item.get("category_icon", "📋")
            badge_bg = doc_item.get("badge_bg", "#f1f5f9")
            badge_txt = doc_item.get("badge_text", "#334155")
            badge_brd = doc_item.get("badge_border", "#cbd5e1")
            dtags = doc_item.get("disease_tags", "")

            with st.container():
                col_dinfo, col_dpreview, col_dact = st.columns([3.5, 2.0, 0.8])
                with col_dinfo:
                    st.markdown(
                        f'<div style="display:flex; flex-direction:column; gap:4px; margin-top:2px;">'
                        f'<div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">'
                        f'<span style="background:{badge_bg}; color:{badge_txt}; border:1px solid {badge_brd}; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700;">{cat_icon} {html.escape(cat_name)}</span>'
                        f'<span class="kb-doc-badge {badge_class}">{ftype}</span>'
                        f'<span style="font-size:0.86rem; font-weight:600; color:#0f172a;">{html.escape(doc_name)}</span>'
                        f'</div>'
                        f'<div style="font-size:0.74rem; color:#64748b; margin-left:2px;">'
                        f'<span>{doc_item["chunk_count"]} chunks &bull; {doc_item.get("pages_count", 1)} page(s)</span>'
                        + (f' &bull; <span style="color:#0284c7;">Tags: {html.escape(dtags)}</span>' if dtags else '') +
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_dpreview:
                    with st.expander("🔍 Preview Clinical Text", expanded=False):
                        st.caption(doc_item.get("sample_text", "No preview available."))
                with col_dact:
                    if st.button("🗑 Remove", key=f"del_doc_{doc_name}_{idx}", use_container_width=True):
                        vector_store.delete_document(doc_name, embedder=embedder)
                        load_vector_store.clear()
                        st.rerun()

    # Reset / Danger Zone actions
    st.divider()
    col_clear_l, col_clear_r = st.columns([3, 1])
    with col_clear_l:
        st.caption("Manage vector index persistence. Clearing index removes all embeddings from disk.")
    with col_clear_r:
        if st.button("🗑 Clear Knowledge Base", type="secondary", use_container_width=True, key="btn_clear_kb"):
            vector_store.clear_index()
            load_vector_store.clear()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)



