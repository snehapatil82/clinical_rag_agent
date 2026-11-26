"""
Comprehensive Evaluation CLI Script for Medical RAG.
Runs retrieval evaluation, answer groundedness evaluation, and baseline vs RAG comparison
on the gold-standard benchmark dataset.
"""

import argparse
import json
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.embeddings.embedder import MedicalEmbedder
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.retriever import ClinicalRetriever
from app.rag.pipeline import ClinicalRAGPipeline
from app.generation.llm import get_llm_provider
from app.evaluation.retrieval_metrics import evaluate_retrieval_dataset
from app.evaluation.answer_metrics import compare_baseline_vs_rag
from app.utils.logging_config import setup_logger

logger = setup_logger("scripts.evaluate")


def run_evaluation(
    eval_file: Path,
    vector_dir: Path,
    k_values: list = [1, 3, 5],
    comparison_samples: int = 5,
):
    """Executes full retrieval and answer quality benchmark."""
    print("\n" + "=" * 70)
    print("      MEDICAL RAG ACADEMIC BENCHMARK EVALUATION")
    print("=" * 70)

    if not eval_file.exists():
        logger.error(f"Evaluation benchmark file not found: {eval_file}")
        sys.exit(1)

    with open(eval_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    print(f"Loaded {len(benchmark_data)} evaluation test cases from {eval_file.name}\n")

    # Step 1: Initialize Vector Store and Retriever
    vector_store = FAISSVectorStore(store_dir=vector_dir)
    if not vector_store.load():
        print("[ERROR] FAISS index could not be loaded. Please run 'python scripts/build_index.py' first.")
        sys.exit(1)

    embedder = MedicalEmbedder()
    retriever = ClinicalRetriever(vector_store=vector_store, embedder=embedder)

    # Step 2: Information Retrieval Evaluation across K values
    print("-" * 70)
    print("1. RETRIEVAL PERFORMANCE BENCHMARK")
    print("-" * 70)
    print(f"{'Metric':<20} | {'K=1':<12} | {'K=3':<12} | {'K=5':<12}")
    print("-" * 70)

    retrieval_summaries = {}
    for k in k_values:
        summary = evaluate_retrieval_dataset(retriever, benchmark_data, k=k)
        retrieval_summaries[k] = summary

    p_row = f"{'Precision@K':<20} | {retrieval_summaries[1]['Precision@1']:<12.4f} | {retrieval_summaries[3]['Precision@3']:<12.4f} | {retrieval_summaries[5]['Precision@5']:<12.4f}"
    r_row = f"{'Recall@K':<20} | {retrieval_summaries[1]['Recall@1']:<12.4f} | {retrieval_summaries[3]['Recall@3']:<12.4f} | {retrieval_summaries[5]['Recall@5']:<12.4f}"
    h_row = f"{'Hit Rate@K':<20} | {retrieval_summaries[1]['Hit_Rate@1']:<12.4f} | {retrieval_summaries[3]['Hit_Rate@3']:<12.4f} | {retrieval_summaries[5]['Hit_Rate@5']:<12.4f}"
    mrr_val = retrieval_summaries[3]["MRR"]

    print(p_row)
    print(r_row)
    print(h_row)
    print(f"{'Mean Reciprocal Rank (MRR)':<20} | {mrr_val:<12.4f}")
    print("-" * 70)

    # Step 3: Baseline vs RAG Generation Comparison
    print("\n" + "-" * 70)
    print("2. BASELINE (NO-RAG) VS PROPOSED RAG COMPARATIVE BENCHMARK")
    print("-" * 70)

    llm = get_llm_provider()
    rag_pipeline = ClinicalRAGPipeline(vector_store=vector_store, embedder=embedder, llm_provider=llm)

    comparison_results = compare_baseline_vs_rag(
        rag_pipeline=rag_pipeline,
        benchmark_dataset=benchmark_data,
        sample_size=comparison_samples,
    )

    print(f"{'Metric':<35} | {'Baseline (No-RAG)':<18} | {'Proposed RAG':<18}")
    print("-" * 70)
    print(f"{'Mean Evidence Groundedness':<35} | {comparison_results['mean_baseline_faithfulness']:<18.4f} | {comparison_results['mean_rag_faithfulness']:<18.4f}")
    print(f"{'Mean Reference Answer F1':<35} | {comparison_results['mean_baseline_reference_f1']:<18.4f} | {comparison_results['mean_rag_reference_f1']:<18.4f}")
    print(f"{'Source Citation Attribution Rate':<35} | {comparison_results['baseline_source_attribution_rate']:<18.0%} | {comparison_results['rag_source_attribution_rate']:<18.0%}")
    print("-" * 70)

    # Step 4: Sample Qualitative Comparison
    print("\n" + "-" * 70)
    print("3. QUALITATIVE GENERATION SAMPLES")
    print("-" * 70)
    for idx, sample in enumerate(comparison_results["comparisons"][:2], start=1):
        print(f"\n[Case #{idx}] Question: {sample['question']}")
        print(f"  --> RAG Answer: {sample['rag']['answer'][:160]}...")
        print(f"      Sources Cited: {sample['rag']['sources']}")
        print(f"      Confidence: {sample['rag']['confidence']:.4f} | Faithfulness: {sample['rag']['faithfulness']:.4f}")
        print(f"  --> Baseline Answer: {sample['baseline']['answer'][:160]}...")
        print(f"      Sources Cited: {sample['baseline']['sources']} (None available)")

    print("\n" + "=" * 70)
    print("            EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Medical RAG performance.")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=settings.EVALUATION_DATA_PATH,
        help="Path to evaluation questions JSON file",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=settings.VECTOR_STORE_DIR,
        help="Path to vector store directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples for generation comparison",
    )
    args = parser.parse_args()

    run_evaluation(
        eval_file=args.eval_file,
        vector_dir=args.vector_dir,
        comparison_samples=args.samples,
    )
