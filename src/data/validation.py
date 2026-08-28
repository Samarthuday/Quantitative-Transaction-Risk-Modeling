"""Data quality validation and checks."""

import logging

import numpy as np
import pandas as pd

from src.models.train import MODEL_FEATURES, TARGET

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validate feature dataset quality and schema."""

    def __init__(self, nan_threshold: float = 0.05):
        """
        Initialize validator.

        Args:
            nan_threshold: Acceptable NaN percentage (0.05 = 5%)
        """
        self.nan_threshold = nan_threshold
        self.issues = []

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Perform all validation checks.

        Args:
            df: Feature dataset to validate

        Returns:
            True if all checks pass, False otherwise
        """
        self.issues = []

        self._check_schema(df)
        self._check_nans(df)
        self._check_chronological_order(df)
        self._check_target_distribution(df)
        self._check_numeric_ranges(df)

        if self.issues:
            logger.warning(f"Data quality issues found: {len(self.issues)}")
            for issue in self.issues:
                logger.warning(f"  - {issue}")
            return False

        logger.info("Data quality validation passed")
        return True

    def _check_schema(self, df: pd.DataFrame):
        """Check that all required features are present."""
        expected_features = set(MODEL_FEATURES)
        actual_features = set(df.columns)

        missing = expected_features - actual_features
        if missing:
            self.issues.append(f"Missing features: {missing}")

        if TARGET not in df.columns:
            self.issues.append(f"Target column '{TARGET}' not found")

        if "timestamp" not in df.columns:
            self.issues.append("timestamp column not found")

    def _check_nans(self, df: pd.DataFrame):
        """Check NaN values in each column."""
        for col in df.columns:
            nan_count = df[col].isna().sum()
            nan_pct = nan_count / len(df)

            if nan_pct > self.nan_threshold:
                self.issues.append(
                    f"{col}: {nan_pct:.2%} NaN values (threshold: {self.nan_threshold:.2%})"
                )

    def _check_chronological_order(self, df: pd.DataFrame):
        """Verify data is chronologically ordered."""
        if "timestamp" not in df.columns:
            return

        if not df["timestamp"].is_monotonic_increasing:
            self.issues.append("Data is not chronologically sorted")

    def _check_target_distribution(self, df: pd.DataFrame):
        """Check target class distribution."""
        if TARGET not in df.columns:
            return

        if df[TARGET].nunique() < 2:
            self.issues.append(f"Target '{TARGET}' has fewer than 2 classes")

        prevalence = df[TARGET].mean()
        if prevalence < 0.0001 or prevalence > 0.9999:
            logger.warning(
                f"Target prevalence is extreme: {prevalence:.6f} "
                "(may indicate data leakage or sampling bias)"
            )

    def _check_numeric_ranges(self, df: pd.DataFrame):
        """Check numeric features for unreasonable values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Skip target and special columns
            if col in [TARGET] or col.startswith("sender_") or col.startswith("receiver_"):
                continue

            # Check for infinity values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                self.issues.append(f"{col}: {inf_count} infinite values")

            # Check for extreme outliers
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr > 0:
                lower_bound = q1 - 10 * iqr
                upper_bound = q3 + 10 * iqr

                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > len(df) * 0.05:  # More than 5% outliers
                    logger.warning(
                        f"{col}: {outliers} extreme outliers ({outliers / len(df):.2%})"
                    )


def validate_features(
    df: pd.DataFrame,
    nan_threshold: float = 0.05,
) -> bool:
    """
    Validate a feature dataset.

    Args:
        df: Feature dataset
        nan_threshold: Acceptable NaN percentage

    Returns:
        True if valid, False otherwise

    Raises:
        ValueError: If critical validation checks fail
    """
    validator = DataQualityValidator(nan_threshold=nan_threshold)

    if not validator.validate(df):
        if validator.issues:
            first_issue = validator.issues[0]
            if "Missing" in first_issue or "not found" in first_issue:
                raise ValueError(f"Critical validation failure: {first_issue}")

    return True
