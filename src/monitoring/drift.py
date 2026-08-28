"""Model drift and data distribution monitoring."""

from typing import Dict, List

import numpy as np
import pandas as pd


class DriftDetector:
    """Detect prediction distribution shifts and feature drift."""

    def __init__(self, baseline_quantiles: List[float]):
        """
        Initialize drift detector.

        Args:
            baseline_quantiles: Baseline prediction probability quantiles
        """
        self.baseline_quantiles = np.array(baseline_quantiles)
        self.baseline_median = np.percentile(baseline_quantiles, 50)
        self.baseline_std = np.std(baseline_quantiles)

    def detect_prediction_drift(
        self,
        predictions: np.ndarray,
        threshold_std: float = 2.0,
    ) -> Dict[str, float]:
        """
        Detect shift in prediction distribution.

        Args:
            predictions: Array of predicted probabilities
            threshold_std: Standard deviation threshold for drift detection

        Returns:
            Dictionary with drift metrics
        """
        predictions = np.asarray(predictions)

        current_median = np.median(predictions)
        current_std = np.std(predictions)

        median_shift = abs(current_median - self.baseline_median)
        std_ratio = current_std / max(self.baseline_std, 1e-8)

        median_drift = median_shift > (threshold_std * self.baseline_std)
        variance_drift = std_ratio > 2.0 or std_ratio < 0.5

        return {
            "baseline_median": float(self.baseline_median),
            "current_median": float(current_median),
            "median_shift": float(median_shift),
            "baseline_std": float(self.baseline_std),
            "current_std": float(current_std),
            "std_ratio": float(std_ratio),
            "median_drift_detected": bool(median_drift),
            "variance_drift_detected": bool(variance_drift),
            "drift_detected": bool(median_drift or variance_drift),
        }

    def detect_feature_drift(
        self,
        feature_df: pd.DataFrame,
        baseline_stats: Dict[str, Dict[str, float]],
        threshold: float = 0.1,
    ) -> Dict[str, Dict[str, float]]:
        """
        Detect shifts in feature distributions.

        Args:
            feature_df: DataFrame with features
            baseline_stats: Dictionary of baseline statistics per feature
            threshold: Percentage change threshold for drift

        Returns:
            Dictionary with feature drift metrics
        """
        drift_results = {}

        for col in feature_df.columns:
            if col not in baseline_stats:
                continue

            current_mean = feature_df[col].mean()
            current_std = feature_df[col].std()
            baseline_mean = baseline_stats[col].get("mean", 0)
            baseline_std = baseline_stats[col].get("std", 1)

            mean_change = abs(current_mean - baseline_mean) / max(abs(baseline_mean), 1e-8)
            std_change = abs(current_std - baseline_std) / max(baseline_std, 1e-8)

            drift_results[col] = {
                "baseline_mean": float(baseline_mean),
                "current_mean": float(current_mean),
                "mean_change_pct": float(mean_change * 100),
                "baseline_std": float(baseline_std),
                "current_std": float(current_std),
                "std_change_pct": float(std_change * 100),
                "drift_detected": bool(mean_change > threshold or std_change > threshold),
            }

        return drift_results


def compute_baseline_stats(
    df: pd.DataFrame,
    features: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute baseline statistics for feature drift detection.

    Args:
        df: Training/baseline dataset
        features: List of features to compute stats for

    Returns:
        Dictionary with mean and std for each feature
    """
    stats = {}

    for col in features:
        if col not in df.columns:
            continue

        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
        }

    return stats
