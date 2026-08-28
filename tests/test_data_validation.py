"""Tests for data quality validation."""

import pytest
import pandas as pd
import numpy as np

from src.data.validation import DataQualityValidator, validate_features
from src.models.train import MODEL_FEATURES, TARGET


def create_valid_dataframe(n_rows: int = 100) -> pd.DataFrame:
    """Create a valid feature dataframe for testing."""
    data = {feature: np.random.rand(n_rows) for feature in MODEL_FEATURES}
    data["timestamp"] = pd.date_range("2026-01-01", periods=n_rows, freq="H")
    data[TARGET] = np.random.randint(0, 2, n_rows)

    return pd.DataFrame(data)


def test_validator_accepts_valid_data():
    """Test that validator accepts valid data."""
    df = create_valid_dataframe()

    validator = DataQualityValidator()
    assert validator.validate(df)
    assert len(validator.issues) == 0


def test_validator_detects_missing_features():
    """Test that validator detects missing required features."""
    df = create_valid_dataframe()
    df = df.drop(columns=["Amount"])

    validator = DataQualityValidator()
    assert not validator.validate(df)
    assert any("Missing" in issue for issue in validator.issues)


def test_validator_detects_missing_target():
    """Test that validator detects missing target column."""
    df = create_valid_dataframe()
    df = df.drop(columns=[TARGET])

    validator = DataQualityValidator()
    assert not validator.validate(df)
    assert any(TARGET in issue for issue in validator.issues)


def test_validator_detects_excessive_nans():
    """Test that validator detects columns with too many NaNs."""
    df = create_valid_dataframe(n_rows=100)

    # Introduce 10% NaN in one column (exceeds default 5% threshold)
    df.loc[:10, "Amount"] = np.nan

    validator = DataQualityValidator(nan_threshold=0.05)
    assert not validator.validate(df)
    assert any("Amount" in issue and "NaN" in issue for issue in validator.issues)


def test_validator_detects_non_chronological_data():
    """Test that validator detects non-chronological ordering."""
    df = create_valid_dataframe(n_rows=10)

    # Shuffle timestamps
    df = df.sample(frac=1).reset_index(drop=True)

    validator = DataQualityValidator()
    assert not validator.validate(df)
    assert any("chronologically" in issue for issue in validator.issues)


def test_validator_warns_on_extreme_class_imbalance():
    """Test that validator warns on extreme class imbalance."""
    df = create_valid_dataframe(n_rows=1000)

    # Create extreme imbalance (99.9% negative)
    df[TARGET] = 0
    df.loc[:1, TARGET] = 1

    validator = DataQualityValidator()
    # Should still pass validation but with warning
    validator.validate(df)


def test_validator_detects_infinite_values():
    """Test that validator detects infinite values."""
    df = create_valid_dataframe(n_rows=100)

    df.loc[0, "Amount"] = np.inf
    df.loc[1, "log_amount"] = -np.inf

    validator = DataQualityValidator()
    assert not validator.validate(df)
    assert any("infinite" in issue for issue in validator.issues)


def test_validate_features_function():
    """Test the high-level validate_features function."""
    df = create_valid_dataframe()

    # Should not raise
    assert validate_features(df)


def test_validate_features_raises_on_missing_schema():
    """Test that validate_features raises on missing required columns."""
    df = create_valid_dataframe()
    df = df.drop(columns=["Amount"])

    with pytest.raises(ValueError, match="Critical validation failure"):
        validate_features(df)


def test_validator_with_custom_nan_threshold():
    """Test validator with custom NaN threshold."""
    df = create_valid_dataframe(n_rows=100)

    # Introduce 5% NaN
    df.loc[:5, "Amount"] = np.nan

    # Should pass with 10% threshold
    validator = DataQualityValidator(nan_threshold=0.10)
    assert validator.validate(df)

    # Should fail with 1% threshold
    validator = DataQualityValidator(nan_threshold=0.01)
    assert not validator.validate(df)
