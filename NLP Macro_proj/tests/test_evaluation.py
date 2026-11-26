"""
Unit tests for Information Retrieval and Answer Groundedness Metrics.
"""

from app.evaluation.retrieval_metrics import compute_retrieval_metrics
from app.evaluation.answer_metrics import evaluate_answer_groundedness, compute_f1_similarity


def test_precision_recall_hit_rate_calculation():
    retrieved = ["doc_a.pdf", "doc_b.pdf", "doc_c.pdf"]
    ground_truth = ["doc_a.pdf", "doc_x.pdf"]

    # K = 3
    # Retrieved in top 3: doc_a (relevant), doc_b (not), doc_c (not) => 1 relevant
    # Precision@3 = 1 / 3 = 0.3333
    # Recall@3 = 1 / 2 = 0.5000
    # Hit Rate@3 = 1.0
    # Reciprocal Rank = 1 / 1 = 1.0
    metrics = compute_retrieval_metrics(retrieved, ground_truth, k=3)
    assert metrics["precision@3"] == 0.3333
    assert metrics["recall@3"] == 0.5
    assert metrics["hit_rate@3"] == 1.0
    assert metrics["reciprocal_rank"] == 1.0


def test_mrr_calculation_second_rank():
    retrieved = ["irrelevant.pdf", "target_doc.pdf", "other.pdf"]
    ground_truth = ["target_doc.pdf"]

    metrics = compute_retrieval_metrics(retrieved, ground_truth, k=3)
    assert metrics["reciprocal_rank"] == 0.5  # 1/2


def test_answer_faithfulness_scoring():
    context = ["Hypertension first line treatment includes ACE inhibitors and thiazide diuretics."]
    grounded_answer = "First line treatment for hypertension includes ACE inhibitors and thiazide diuretics."
    hallucinated_answer = "Patients must immediately undergo laser angioplasty and ketogenic fasting."

    score_grounded = evaluate_answer_groundedness(grounded_answer, context)
    score_hallucinated = evaluate_answer_groundedness(hallucinated_answer, context)

    assert score_grounded["faithfulness_score"] > 0.80
    assert score_hallucinated["faithfulness_score"] < 0.20


def test_f1_similarity_perfect_match():
    text = "Asthma symptoms include wheezing dyspnea and cough"
    f1 = compute_f1_similarity(text, text)
    assert f1 == 1.0
