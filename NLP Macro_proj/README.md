# Retrieval-Augmented Clinical Question Answering Using Medical Knowledge Bases

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20IndexFlatIP-orange.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Undergraduate NLP Capstone / Course Project**  
> An evidence-grounded, source-traceable Clinical Question-Answering system powered by Semantic Vector Retrieval (FAISS + Sentence Transformers) and Large Language Model (LLM) orchestration with strict hallucination safeguards.

---

## Table of Contents

1. [Project Title & Overview](#1-project-title--overview)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [Key Objectives](#4-key-objectives)
5. [System Architecture](#5-system-architecture)
6. [Technology Stack & Rationale](#6-technology-stack--rationale)
7. [Repository Structure](#7-repository-structure)
8. [Installation & Setup](#8-installation--setup)
9. [Configuration & Environment Variables](#9-configuration--environment-variables)
10. [Adding Medical Documents](#10-adding-medical-documents)
11. [Building the Vector Index](#11-building-the-vector-index)
12. [Running the Application](#12-running-the-application)
13. [Academic Evaluation & Benchmark Results](#13-academic-evaluation--benchmark-results)
14. [Hallucination Mitigation & Safeguards](#14-hallucination-mitigation--safeguards)
15. [Academic Viva Voce & Theoretical Concepts](#15-academic-viva-voce--theoretical-concepts)
16. [Limitations & Future Scope](#16-limitations--future-scope)
17. [Medical Safety Disclaimer](#17-medical-safety-disclaimer)

---

## 1. Project Title & Overview

**Title:** *Retrieval-Augmented Clinical Question Answering Using Medical Knowledge Bases*

This project implements an end-to-end NLP system that ingests authoritative medical practice guidelines (e.g., Hypertension, Asthma, Type 2 Diabetes, Cardiovascular Disease Prevention, Community-Acquired Pneumonia), converts them into dense semantic vectors using Sentence Transformers, indexes them in a persistent FAISS vector store, semantically retrieves top-K evidence chunks for user queries, applies a relevance similarity threshold, and synthesizes source-attributed, evidence-grounded answers via an LLM abstraction layer.

---

## 2. Problem Statement

Standard Large Language Models (LLMs) deployed directly for biomedical question answering present severe limitations in high-stakes domains:
1. **Clinical Hallucinations:** Generative models can generate plausible-sounding yet factually inaccurate medical claims, incorrect drug dosages, or contraindicated combinations.
2. **Lack of Provenance:** Standard generative models cannot provide verifiable page- or passage-level citations for their statements.
3. **Knowledge Cutoffs & Domain Specificity:** Generic foundational models lack awareness of localized institutional guidelines or recently updated consensus definitions.
4. **Uncalibrated Confidence on Out-of-Domain Inputs:** Standard LLMs will attempt to answer non-medical or unsupported queries confidently instead of acknowledging ignorance.

---

## 3. Proposed Solution

This project adopts **Retrieval-Augmented Generation (RAG)**:
- By pairing a **dense semantic retriever (FAISS + SentenceTransformers)** with an **LLM generator**, generation is constrained to retrieved evidence passages.
- A **relevance thresholding mechanism** ensures that if the knowledge base contains no relevant text (e.g., for out-of-domain or unsupported queries), the system politely refrains from answering.
- Every generated answer includes **traceable source citations** (document name, page number, similarity score, chunk ID) with an interactive **Evidence Inspector**.

---

## 4. Key Objectives

- **Document Ingestion:** Multi-format loading (PDF, Markdown, TXT) with clinical-safe text normalization.
- **Semantic Vector Indexing:** Embedding generation via `sentence-transformers/all-MiniLM-L6-v2` and indexing via `faiss.IndexFlatIP`.
- **Cosine Similarity Lookup:** Exact dot-product search over unit-normalized embeddings.
- **Evidence-Grounded Prompting:** Structured prompt engineering enforcing strict adherence to context.
- **Hallucination Control:** Multi-tier safeguard pipeline (relevance threshold, grounded prompts, source citations, safety guardrails).
- **Quantitative Evaluation:** IR metrics (Precision@K, Recall@K, Hit Rate@K, MRR) and generation metrics (Faithfulness, Reference F1, Baseline vs RAG).
- **Interactive UI:** Professional multi-tab Streamlit dashboard.

---

## 5. System Architecture

```
                                 +--------------------------------+
                                 |  Medical Knowledge Documents   |
                                 | (PDF, Markdown, TXT in raw/)   |
                                 +---------------+----------------+
                                                 |
                                                 v
                                 +--------------------------------+
                                 |   Document Loader & Cleaner    |
                                 | (PyPDF / PyMuPDF / NFKC Clean) |
                                 +---------------+----------------+
                                                 |
                                                 v
                                 +--------------------------------+
                                 |    Sentence-Aware Chunker      |
                                 |  (600 chars, 100 char overlap) |
                                 +---------------+----------------+
                                                 |
                                                 v
                                 +--------------------------------+
                                 |   SentenceTransformer Model    |
                                 |    (all-MiniLM-L6-v2, 384-D)   |
                                 +---------------+----------------+
                                                 | (L2 Normalization)
                                                 v
                                 +--------------------------------+
                                 |       FAISS Vector Store       |
                                 |   (IndexFlatIP + metadata.json)|
                                 +---------------+----------------+
                                                 |
                                                 |
[User Query] ------------------------------------+
     |
     v
[Query Embedder (384-D)]
     |
     v
[FAISS Cosine Similarity Search (Top-K = 4)]
     |
     v
[Relevance Threshold Check (Score >= 0.38)]
     |
     +-----------------------------+
     |                             |
(Score >= 0.38)             (Score < 0.38)
     |                             |
     v                             v
[Assemble Context Blocks]    [Return "Insufficient Evidence" Refusal]
     |
     v
[Evidence-Grounded Prompt]
     |
     v
[LLM Provider Backend]
(Groq / OpenAI / Gemini / Mock)
     |
     v
[Source-Attributed Clinical Answer + Evidence Inspector]
```

---

## 6. Technology Stack & Rationale

| Layer | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.11+ / 3.13 | Industry and academic standard for NLP and Machine Learning. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | High quality 384-dimensional dense semantic representations, fast CPU inference, 80MB footprint. |
| **Vector Database** | `faiss-cpu` (Facebook AI Similarity Search) | High-performance C++ vector index with Python bindings; `IndexFlatIP` allows exact cosine similarity search. |
| **Document Parsers** | `pypdf`, `pymupdf` (fitz) | Reliable PDF text and metadata extraction with multi-page provenance. |
| **LLM Provider** | Groq / OpenAI / Gemini / Mock | Abstracted provider layer supporting high-speed inference and deterministic offline mock testing. |
| **Web Interface** | Streamlit | Lightweight, reactive Python UI for academic demonstration. |
| **Testing** | `pytest` | Comprehensive unit tests for chunking, embeddings, retrieval, pipeline, and evaluation. |

---

## 7. Repository Structure

```
medical_rag/
├── app/
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration & environment loader
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py              # PDF/TXT/MD document loaders
│   │   ├── cleaner.py              # Medical-safe text normalization
│   │   └── chunker.py              # Sentence-aware text chunker
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py             # SentenceTransformers embedding engine
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── faiss_store.py          # FAISS IndexFlatIP & metadata manager
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py            # Semantic retrieval & threshold filter
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm.py                  # Multi-provider LLM abstraction
│   │   └── prompts.py              # Clinical system prompts & safety directives
│   ├── rag/
│   │   ├── __init__.py
│   │   └── pipeline.py             # End-to-end RAG orchestrator
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py    # Precision@K, Recall@K, Hit Rate, MRR
│   │   └── answer_metrics.py       # Faithfulness, F1, Baseline comparison
│   └── utils/
│       ├── __init__.py
│       └── logging_config.py       # Centralized structured logger
│
├── data/
│   ├── raw/                        # Clinical documents (PDF & Markdown)
│   │   ├── asthma_management_guidelines.pdf & .md
│   │   ├── cardiovascular_disease_prevention.pdf & .md
│   │   ├── community_acquired_pneumonia_guidelines.pdf & .md
│   │   ├── hypertension_clinical_guidelines.pdf & .md
│   │   └── type2_diabetes_guidelines.pdf & .md
│   ├── processed/                  # Intermediate chunk caches
│   └── evaluation/
│       └── questions.json          # 26-item gold-standard benchmark dataset
│
├── vector_store/
│   ├── medical.index               # Serialized FAISS IndexFlatIP binary
│   └── metadata.json               # Aligned chunk metadata catalog
│
├── scripts/
│   ├── ingest_documents.py         # Document extraction CLI
│   ├── build_index.py              # Vector index construction CLI
│   └── evaluate.py                 # Academic benchmark evaluation CLI
│
├── tests/
│   ├── test_loaders.py             # Test file parsers and text cleaner
│   ├── test_chunking.py            # Test sliding window & chunk boundaries
│   ├── test_embeddings.py          # Test vector dimensionality & normalization
│   ├── test_retrieval.py           # Test FAISS lookup & ranking
│   ├── test_pipeline.py            # Test full RAG workflow & refusal
│   └── test_evaluation.py          # Test metric formulas
│
├── streamlit_app.py                # Interactive Streamlit Web Application
├── requirements.txt                # Python package dependencies
├── .env.example                    # Template environment variables
├── .gitignore                      # Git exclusion rules
└── README.md                       # Comprehensive project documentation
```

---

## 8. Installation & Setup

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd "NLP Macro_proj"
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. Configuration & Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your preferred LLM provider and API key in `.env`:
```env
# LLM Provider: "groq", "openai", "gemini", or "mock"
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-120b
LLM_TEMPERATURE=0.1

# Embedding & Retrieval Settings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
TOP_K=4
SIMILARITY_THRESHOLD=0.38
```

> **Note:** If no API key is provided, the system automatically runs using `MockLLMProvider`, allowing complete offline demonstrations and automated unit tests at zero cost.

---

## 10. Adding Medical Documents

Place your clinical reference documents (PDF, Markdown, or TXT) into `data/raw/`:
```bash
data/raw/
  ├── clinical_guideline_1.pdf
  ├── reference_doc_2.md
  └── medical_standard_3.txt
```

The system automatically parses all supported files in `data/raw/` preserving document names and page numbers.

---

## 11. Building the Vector Index

Run the automated indexing script:
```bash
python scripts/build_index.py
```

**Execution Pipeline:**
1. Loads all documents from `data/raw/`.
2. Cleans text while preserving medical acronyms, drug dosages, and units.
3. Splits documents into overlapping semantic chunks.
4. Generates L2-normalized dense embeddings via `all-MiniLM-L6-v2`.
5. Serializes the FAISS index to `vector_store/medical.index` and metadata to `vector_store/metadata.json`.

---

## 12. Running the Application

Launch the Streamlit web application:
```bash
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`.

---

## 13. Academic Evaluation & Benchmark Results

Run the full evaluation benchmark CLI:
```bash
python scripts/evaluate.py
```

### 13.1 Information Retrieval (IR) Performance

Evaluated on the 26-question gold-standard benchmark dataset across $K \in \{1, 3, 5\}$:

| Metric | K = 1 | K = 3 | K = 5 |
|---|---|---|---|
| **Precision@K** | 0.8846 | 0.8718 | 0.8308 |
| **Recall@K** | 0.4423 | 0.8846 | 0.8846 |
| **Hit Rate@K** | 0.8846 | 0.8846 | 0.8846 |
| **Mean Reciprocal Rank (MRR)** | **0.8846** | **0.8846** | **0.8846** |

### 13.2 Baseline (No-RAG) vs Proposed RAG Generation

| Dimension | Baseline (No-RAG) | Proposed Clinical RAG | Improvement |
|---|---|---|---|
| **Evidence Groundedness / Faithfulness** | `0.0541` (5.4%) | `0.7391` (73.9%) | **+1266% relative increase** |
| **Source Citation Attribution Rate** | `0%` (Unverifiable) | `100%` (Traceable) | **Complete provenance** |
| **Out-of-Domain Hallucination Control** | Answers confidently | Refuses with Insufficient Evidence | **Eliminates false claims** |

---

## 14. Hallucination Mitigation & Safeguards

The architecture incorporates a 5-tier defense system:
1. **Relevance Thresholding:** Queries yielding a top cosine similarity $< 0.38$ trigger an automatic refusal: *"I could not find sufficient evidence in the provided medical knowledge base to answer this question reliably."*
2. **Context-Grounded System Prompt:** The LLM is instructed under strict system constraints to never invent clinical facts or extrapolate beyond provided context.
3. **Source Attribution:** Every output is linked to verified document names and page numbers.
4. **Safety Filter:** Diagnostic phrases (e.g., *"diagnose me"*, *"what dosage should I take"*) trigger a clinical safety refusal and redirect to licensed practitioners.
5. **Faithfulness Scoring:** Automated token-overlap evaluation continuously verifies that answers adhere to retrieved passages.

---

## 15. Academic Viva Voce & Theoretical Concepts

### Concept 1: Dense Semantic Embeddings vs Sparse Keyword Search (BM25)
* **Keyword Matching (BM25 / TF-IDF):** Relies on exact token overlap. Fails when clinical terminology varies (e.g., *"high BP"* vs. *"arterial hypertension"*).
* **Dense Embeddings (Sentence Transformers):** Maps textual sentences into a continuous $D$-dimensional vector space ($\mathbb{R}^{384}$) using bi-encoder transformer architectures (MiniLM). Semantic distance directly mirrors clinical meaning.

### Concept 2: Vector Normalization and Cosine Similarity in FAISS
The cosine similarity between query vector $\mathbf{q}$ and document vector $\mathbf{d}$ is:
$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

By applying unit L2 normalization during embedding computation ($\|\mathbf{q}\|_2 = 1, \|\mathbf{d}\|_2 = 1$), the Inner Product (IP) computed by `faiss.IndexFlatIP` is mathematically equivalent to Cosine Similarity:
$$\mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{384} q_i d_i = \cos(\theta)$$

### Concept 3: Mathematical Metric Definitions

#### Precision@K
Measures the proportion of retrieved documents in the top-$K$ list that are clinically relevant:
$$\text{Precision@K} = \frac{|\text{Retrieved Relevant Documents in Top-}K|}{K}$$

#### Recall@K
Measures the proportion of all relevant documents in the knowledge base that were successfully retrieved:
$$\text{Recall@K} = \frac{|\text{Retrieved Relevant Documents in Top-}K|}{|\text{Total Relevant Documents for Query}|}$$

#### Mean Reciprocal Rank (MRR)
Evaluates ranking quality based on the position of the first relevant document:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
*(where $\text{rank}_i$ is the rank position of the first relevant document for query $i$, or $\infty$ if no relevant item is retrieved).*

---

## 16. Limitations & Future Scope

### Current Limitations
1. **Dense-Only Retrieval:** Highly specialized alphanumeric codes or rare drug trade names can occasionally benefit from hybrid lexical search.
2. **Context Window Limits:** Top-K chunks are concatenated linearly without cross-encoder reranking.

### Future Improvements
1. **Hybrid Retrieval (BM25 + Dense FAISS):** Reciprocal Rank Fusion (RRF) combining sparse and dense retrieval.
2. **Cross-Encoder Reranking:** Adding a lightweight cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score top-20 retrieved candidates into top-4.
3. **Structured Medical Knowledge Graph Integration:** Combining tabular EHR/Ontology graphs (UMLS/SNOMED-CT) with textual RAG.

---

## 17. Medical Safety Disclaimer

> **IMPORTANT DISCLAIMER**  
> This software is developed solely for academic research, education, and algorithmic evaluation. It does NOT constitute medical advice, clinical diagnosis, patient management recommendations, or prescription guidance. Medical decisions should always be made by licensed healthcare professionals.
