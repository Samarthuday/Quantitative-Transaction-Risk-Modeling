"""Artifact-backed scoring helpers shared by offline and API inference."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def model_input_from_features(
    transaction_features: Mapping[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    """Validate and order an already-computed feature payload.

    Account identifiers are deliberately absent: a production feature store
    must turn them into historical behavioural variables before this boundary.
    """

    forbidden = {"Sender_account", "Receiver_account", "Is_laundering", "Laundering_type"}
    supplied = set(transaction_features)
    forbidden_supplied = supplied & forbidden
    if forbidden_supplied:
        raise ValueError(
            "Identifier or target fields are not model inputs: "
            f"{sorted(forbidden_supplied)}"
        )

    missing = set(feature_names) - supplied
    if missing:
        raise ValueError(
            "Missing model features. Historical behavioural features must be "
            f"provided by the feature store: {sorted(missing)}"
        )

    return pd.DataFrame([{name: transaction_features[name] for name in feature_names}])


def predict_calibrated_probability(artifact: Mapping[str, Any], features: pd.DataFrame) -> float:
    """Score one feature row with the saved preprocessing, model, and calibrator."""

    processed = artifact["preprocessor"].transform(features)
    raw_probability = artifact["model"].predict_proba(processed)[:, 1]
    return float(artifact["calibrator"].predict(raw_probability)[0])


def probability_percentile(artifact: Mapping[str, Any], probability: float) -> float | None:
    """Rank a probability against validation probabilities, when available."""

    reference = artifact.get("validation_probability_quantiles")
    if not reference:
        return None

    values = np.asarray(reference, dtype=float)
    return float(100 * np.searchsorted(values, probability, side="right") / len(values))
