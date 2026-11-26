"""
Academic Evaluation Suite for Retrieval and Answer Quality.
"""

from app.evaluation.retrieval_metrics import evaluate_retrieval_dataset, compute_retrieval_metrics
from app.evaluation.answer_metrics import evaluate_answer_groundedness, compare_baseline_vs_rag

__all__ = [
    "evaluate_retrieval_dataset",
    "compute_retrieval_metrics",
    "evaluate_answer_groundedness",
    "compare_baseline_vs_rag",
]
