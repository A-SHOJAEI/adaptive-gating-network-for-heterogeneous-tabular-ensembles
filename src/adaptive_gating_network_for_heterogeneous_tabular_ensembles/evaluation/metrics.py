"""Evaluation metrics for adaptive gating ensemble."""

import logging
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import (
    compute_feature_complexity,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metric name to value
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Add AUC if probabilities provided
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class
                metrics["auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")

    return metrics


def compute_gating_accuracy(
    y_true: np.ndarray,
    base_predictions: List[np.ndarray],
    gating_assignments: np.ndarray,
) -> float:
    """
    Compute gating network accuracy.

    Measures how often the gating network selects a model that predicts correctly.

    Args:
        y_true: True labels
        base_predictions: List of predictions from each base model
        gating_assignments: Model assignments from gating network

    Returns:
        Gating accuracy (0 to 1)
    """
    correct_routings = 0
    total = len(y_true)

    for i in range(total):
        selected_model = int(gating_assignments[i])
        if selected_model < len(base_predictions):
            if base_predictions[selected_model][i] == y_true[i]:
                correct_routings += 1

    gating_acc = correct_routings / total
    logger.info(f"Gating accuracy: {gating_acc:.4f}")
    return gating_acc


def compute_routing_efficiency(
    y_true: np.ndarray,
    base_predictions: List[np.ndarray],
    gating_assignments: np.ndarray,
) -> float:
    """
    Compute routing efficiency.

    Measures how well the gating network routes samples compared to oracle routing.

    Args:
        y_true: True labels
        base_predictions: List of predictions from each base model
        gating_assignments: Model assignments from gating network

    Returns:
        Routing efficiency (0 to 1)
    """
    n_samples = len(y_true)
    n_models = len(base_predictions)

    # Compute oracle performance (best possible routing)
    oracle_correct = 0
    for i in range(n_samples):
        # Check if any model got this sample correct
        any_correct = any(base_predictions[j][i] == y_true[i] for j in range(n_models))
        if any_correct:
            oracle_correct += 1

    # Compute actual gating performance
    gating_correct = 0
    for i in range(n_samples):
        selected_model = int(gating_assignments[i])
        if selected_model < n_models and base_predictions[selected_model][i] == y_true[i]:
            gating_correct += 1

    # Efficiency = actual / oracle
    efficiency = gating_correct / max(oracle_correct, 1)
    logger.info(
        f"Routing efficiency: {efficiency:.4f} "
        f"(gating: {gating_correct}/{n_samples}, oracle: {oracle_correct}/{n_samples})"
    )
    return efficiency


def compute_complexity_aware_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_complexity: np.ndarray,
    complexity_bins: int = 3,
) -> float:
    """
    Compute complexity-aware AUC.

    Computes AUC separately for different complexity levels and returns weighted average.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        sample_complexity: Complexity score for each sample
        complexity_bins: Number of complexity bins

    Returns:
        Weighted average AUC across complexity levels
    """
    # Bin samples by complexity
    complexity_percentiles = np.percentile(
        sample_complexity, np.linspace(0, 100, complexity_bins + 1)
    )

    auc_scores = []
    weights = []

    for i in range(complexity_bins):
        lower = complexity_percentiles[i]
        upper = complexity_percentiles[i + 1]

        if i == complexity_bins - 1:
            mask = (sample_complexity >= lower) & (sample_complexity <= upper)
        else:
            mask = (sample_complexity >= lower) & (sample_complexity < upper)

        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            try:
                if len(np.unique(y_true)) == 2:
                    auc = roc_auc_score(y_true[mask], y_proba[mask, 1])
                else:
                    auc = roc_auc_score(
                        y_true[mask], y_proba[mask], multi_class="ovr", average="weighted"
                    )
                auc_scores.append(auc)
                weights.append(mask.sum())
            except Exception as e:
                logger.warning(f"Could not compute AUC for bin {i}: {e}")

    if not auc_scores:
        logger.warning("Could not compute complexity-aware AUC")
        return 0.0

    # Weighted average
    complexity_aware_auc = np.average(auc_scores, weights=weights)
    logger.info(f"Complexity-aware AUC: {complexity_aware_auc:.4f}")
    return complexity_aware_auc


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate and return classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Classification report string
    """
    report = classification_report(y_true, y_pred, zero_division=0)
    return report


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)
