"""Async-aware inference layer for FastAPI."""

import asyncio
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


async def model_input_from_features(
    transaction_features: Mapping,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Validate and extract model features from transaction features.

    Rejects forbidden keys (account identifiers, target).
    Raises ValueError if required features are missing.
    """
    forbidden_keys = {"Sender_account", "Receiver_account", "Is_laundering", "Laundering_type"}
    supplied_keys = set(transaction_features.keys())
    overlap = supplied_keys & forbidden_keys

    if overlap:
        raise ValueError(
            f"Identifier or target fields are not model inputs: {overlap}"
        )

    feature_set = set(feature_names)
    missing = feature_set - supplied_keys

    if missing:
        raise ValueError(
            f"Missing model features. Required: {missing}"
        )

    record = {feature: transaction_features[feature] for feature in feature_names}
    return pd.DataFrame([record])


async def predict_calibrated_probability(
    artifact: dict[str, Any],
    features: pd.DataFrame,
) -> float:
    """
    Run inference and return calibrated probability.

    CPU-bound operations (preprocessing, model.predict_proba, calibration)
    are dispatched to the event loop's default thread pool.
    """

    def _preprocess():
        return artifact["preprocessor"].transform(features)

    def _predict(X_processed):
        return artifact["model"].predict_proba(X_processed)[:, 1]

    def _calibrate(raw_probs):
        return artifact["calibrator"].predict(raw_probs)[0]

    loop = asyncio.get_event_loop()

    X_processed = await loop.run_in_executor(None, _preprocess)
    raw_prob = await loop.run_in_executor(None, _predict, X_processed)
    calibrated = await loop.run_in_executor(None, _calibrate, raw_prob)

    return float(calibrated)


def probability_percentile(
    artifact: dict[str, Any],
    probability: float,
) -> Optional[float]:
    """
    Calculate percentile rank of a probability against validation quantiles.

    Lightweight operation (no async needed); pure Python array indexing.
    """
    quantiles = artifact.get("validation_probability_quantiles")

    if not quantiles:
        return None

    percentile = np.searchsorted(quantiles, probability) * 100 / len(quantiles)
    return float(percentile)
