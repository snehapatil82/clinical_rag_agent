"""
Information Retrieval (IR) Evaluation Metrics.
Implements standard academic metrics: Precision@K, Recall@K, Hit Rate@K, and Mean Reciprocal Rank (MRR).
"""

from typing import Any, Dict, List, Set, Union
import numpy as np

from app.retrieval.retriever import ClinicalRetriever, RetrievalResult
from app.utils.logging_config import setup_logger

logger = setup_logger("evaluation.retrieval_metrics")


def compute_retrieval_metrics(
    retrieved_docs: List[str],
    ground_truth_docs: Union[List[str], Set[str]],
    k: int = 3,
) -> Dict[str, float]:
    """
    Computes IR evaluation metrics for a single query.

    Args:
        retrieved_docs: List of retrieved document names in order of ranking.
        ground_truth_docs: List or set of relevant document names for this query.
        k: Cutoff rank K.

    Returns:
        Dictionary with precision@k, recall@k, hit_rate@k, and reciprocal_rank.
    """
    gt_set = {d.strip().lower() for d in ground_truth_docs}
    top_k_retrieved = [d.strip().lower() for d in retrieved_docs[:k]]

    if not gt_set:
        # Out-of-domain / negative sample query
        is_empty_retrieval = len(top_k_retrieved) == 0
        return {
            f"precision@{k}": 1.0 if is_empty_retrieval else 0.0,
            f"recall@{k}": 1.0 if is_empty_retrieval else 0.0,
            f"hit_rate@{k}": 1.0 if is_empty_retrieval else 0.0,
            "reciprocal_rank": 1.0 if is_empty_retrieval else 0.0,
        }

    # Count relevant items retrieved in top-k
    relevant_in_top_k = sum(1 for doc in top_k_retrieved if any(gt in doc or doc in gt for gt in gt_set))

    # Precision@K = |Retrieved Relevant| / K
    precision = relevant_in_top_k / k if k > 0 else 0.0

    # Recall@K = |Retrieved Relevant| / |Total Relevant|
    recall = relevant_in_top_k / len(gt_set) if len(gt_set) > 0 else 0.0
    recall = min(recall, 1.0)

    # Hit Rate@K = 1 if at least one relevant doc in top-K, else 0
    hit_rate = 1.0 if relevant_in_top_k > 0 else 0.0

    # Reciprocal Rank = 1 / rank of first relevant item
    reciprocal_rank = 0.0
    for rank_idx, doc in enumerate(retrieved_docs, start=1):
        if any(gt in doc.strip().lower() or doc.strip().lower() in gt for gt in gt_set):
            reciprocal_rank = 1.0 / rank_idx
            break

    return {
        f"precision@{k}": round(precision, 4),
        f"recall@{k}": round(recall, 4),
        f"hit_rate@{k}": round(hit_rate, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
    }


def evaluate_retrieval_dataset(
    retriever: ClinicalRetriever,
    benchmark_dataset: List[Dict[str, Any]],
    k: int = 3,
) -> Dict[str, Any]:
    """
    Evaluates retrieval performance across a full benchmark dataset.

    Args:
        retriever: Configured ClinicalRetriever instance.
        benchmark_dataset: List of dicts, each with 'question' and 'relevant_documents'.
        k: Evaluation cutoff rank.

    Returns:
        Dictionary with aggregate mean metrics and per-query breakdown.
    """
    precisions = []
    recalls = []
    hit_rates = []
    rrs = []
    per_query_results = []

    for item in benchmark_dataset:
        question = item["question"]
        ground_truth = item.get("relevant_documents", [])

        # Perform retrieval
        retrieved_results = retriever.retrieve(question, top_k=k)
        retrieved_doc_names = [r.document_name for r in retrieved_results]

        metrics = compute_retrieval_metrics(retrieved_doc_names, ground_truth, k=k)

        precisions.append(metrics[f"precision@{k}"])
        recalls.append(metrics[f"recall@{k}"])
        hit_rates.append(metrics[f"hit_rate@{k}"])
        rrs.append(metrics["reciprocal_rank"])

        per_query_results.append({
            "question": question,
            "category": item.get("category", "General"),
            "ground_truth": ground_truth,
            "retrieved": retrieved_doc_names,
            "metrics": metrics,
            "top_similarity": round(retrieved_results[0].score, 4) if retrieved_results else 0.0,
        })

    summary = {
        f"Precision@{k}": round(float(np.mean(precisions)), 4),
        f"Recall@{k}": round(float(np.mean(recalls)), 4),
        f"Hit_Rate@{k}": round(float(np.mean(hit_rates)), 4),
        "MRR": round(float(np.mean(rrs)), 4),
        "total_evaluated_queries": len(benchmark_dataset),
        "k": k,
        "per_query_details": per_query_results,
    }

    logger.info(
        f"Retrieval Evaluation (K={k}): "
        f"Precision@{k}={summary[f'Precision@{k}']}, "
        f"Recall@{k}={summary[f'Recall@{k}']}, "
        f"HitRate@{k}={summary[f'Hit_Rate@{k}']}, "
        f"MRR={summary['MRR']}"
    )

    return summary
