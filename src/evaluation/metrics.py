"""Evaluation metrics for risk models."""

from typing import Dict, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def top_k_alert_mask(
    probabilities: Union[np.ndarray, list],
    alert_rate: float = 0.005,
) -> np.ndarray:
    """Select top-K transactions based on alert rate."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array.")
    if not 0 < alert_rate <= 1:
        raise ValueError("alert_rate must be in (0, 1].")

    alert_count = max(1, int(np.ceil(len(probabilities) * alert_rate)))
    order = np.argsort(-probabilities, kind="stable")
    selected = order[:alert_count]
    mask = np.zeros(len(probabilities), dtype=bool)
    mask[selected] = True
    return mask


def precision_recall_at_alert_rate(
    y_true: Union[np.ndarray, list],
    probabilities: Union[np.ndarray, list],
    alert_rate: float = 0.005,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    predicted = top_k_alert_mask(probabilities, alert_rate)
    threshold = float(probabilities[predicted].min())

    tp = np.sum(
        (predicted == 1) &
        (y_true == 1)
    )

    fp = np.sum(
        (predicted == 1) &
        (y_true == 0)
    )

    fn = np.sum(
        (predicted == 0) &
        (y_true == 1)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    base_rate = y_true.mean()

    lift = (
        precision / base_rate
        if base_rate > 0
        else 0
    )

    return {
        "alert_rate": alert_rate,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "lift": lift,
        "alerts": int(predicted.sum()),
    }


def expected_decision_cost(
    y_true: Union[np.ndarray, list],
    probabilities: Union[np.ndarray, list],
    threshold: float,
    false_negative_cost: float = 100,
    false_positive_cost: float = 1,
) -> float:
    """Return the operational cost induced by a binary alert threshold."""

    y_true = np.asarray(y_true)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)

    false_negatives = np.sum((predictions == 0) & (y_true == 1))
    false_positives = np.sum((predictions == 1) & (y_true == 0))

    return float(
        false_negative_cost * false_negatives
        + false_positive_cost * false_positives
    )

def calculate_risk_weighted_exposure(
    amounts: Union[np.ndarray, list],
    probabilities: Union[np.ndarray, list],
) -> np.ndarray:
    """
    Probability-weighted transaction amount.

    This is NOT called expected financial loss because
    the dataset does not contain realized loss severity.
    """

    amounts = np.asarray(amounts)
    probabilities = np.asarray(probabilities)

    return amounts * probabilities


def evaluate_model(
    y_true: Union[np.ndarray, list],
    probabilities: Union[np.ndarray, list],
) -> Dict[str, float]:

    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        raise ValueError("Evaluation requires both target classes.")

    probabilities = np.clip(
        probabilities,
        1e-8,
        1 - 1e-8,
    )

    results = {
        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probabilities,
            ),

        "log_loss":
            log_loss(
                y_true,
                probabilities,
            ),
    }

    for rate in [
        0.001,
        0.005,
        0.01,
    ]:

        performance = (
            precision_recall_at_alert_rate(
                y_true,
                probabilities,
                alert_rate=rate,
            )
        )

        prefix = f"alert_{rate:.3%}"

        results[
            f"{prefix}_precision"
        ] = performance["precision"]

        results[
            f"{prefix}_recall"
        ] = performance["recall"]

        results[
            f"{prefix}_lift"
        ] = performance["lift"]

    return results
