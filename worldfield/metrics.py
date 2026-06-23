"""Small metric helpers shared by tests and future experiment runners."""

from __future__ import annotations


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision, recall, and F1 from edge-count style metrics."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def retrieval_recall_at_1(predicted_labels, true_labels) -> float:
    """Return the fraction of top-1 retrieved labels that match truth."""
    predicted = list(predicted_labels)
    truth = list(true_labels)
    if len(predicted) != len(truth):
        raise ValueError("predicted_labels and true_labels must have the same length")
    if not truth:
        return 0.0
    return sum(p == t for p, t in zip(predicted, truth)) / len(truth)
