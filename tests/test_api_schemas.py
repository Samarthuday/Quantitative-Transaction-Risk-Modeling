"""Tests for Pydantic request validation models."""

import pytest
from pydantic import ValidationError

from src.api.models import FeatureVector


def create_valid_payload() -> dict:
    """Create a valid feature vector payload."""
    return {
        "Amount": 1000.0,
        "log_amount": 6.91,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "dow_sin": 0.0,
        "dow_cos": 1.0,
        "month_sin": 0.259,
        "month_cos": 0.966,
        "is_weekend": 0,
        "is_night": 0,
        "currency_mismatch": 0,
        "cross_border": 0,
        "is_round_amount": 1,
        "sender_txn_count_24h": 5,
        "sender_amount_sum_24h": 5000.0,
        "sender_amount_mean_30d": 1000.0,
        "sender_amount_std_30d": 500.0,
        "sender_amount_zscore": 0.0,
        "receiver_txn_count_24h": 3,
        "receiver_amount_sum_24h": 3000.0,
        "seconds_since_sender_txn": 3600,
        "sender_txn_count_lifetime": 100,
        "receiver_txn_count_lifetime": 50,
        "sender_out_degree": 25,
        "receiver_in_degree": 20,
        "pair_transaction_count": 5,
        "sender_counterparty_hhi": 0.2,
        "Payment_type": "Transfer",
        "Payment_currency": "USD",
        "Received_currency": "USD",
        "Sender_bank_location": "US",
        "Receiver_bank_location": "US",
    }


def test_valid_feature_vector():
    """Test that valid feature vectors pass validation."""
    payload = create_valid_payload()

    result = FeatureVector.model_validate(payload)

    assert result.Amount == 1000.0
    assert result.Payment_currency == "USD"


def test_missing_required_field():
    """Test that missing required fields raise ValidationError."""
    payload = create_valid_payload()
    del payload["Amount"]

    with pytest.raises(ValidationError) as exc_info:
        FeatureVector.model_validate(payload)

    errors = exc_info.value.errors()
    assert any("Amount" in str(err["loc"]) for err in errors)


def test_invalid_numeric_range():
    """Test that out-of-range numeric values are rejected."""
    payload = create_valid_payload()
    payload["Amount"] = -100.0

    with pytest.raises(ValidationError):
        FeatureVector.model_validate(payload)


def test_invalid_cyclical_encoding():
    """Test that cyclical encodings outside [-1, 1] are rejected."""
    payload = create_valid_payload()
    payload["hour_sin"] = 1.5

    with pytest.raises(ValidationError):
        FeatureVector.model_validate(payload)


def test_invalid_binary_field():
    """Test that binary fields reject non-0/1 values."""
    payload = create_valid_payload()
    payload["is_weekend"] = 2

    with pytest.raises(ValidationError):
        FeatureVector.model_validate(payload)


def test_unknown_fields_rejected():
    """Test that unknown fields are rejected (extra='forbid')."""
    payload = create_valid_payload()
    payload["unknown_field"] = "should_fail"

    with pytest.raises(ValidationError) as exc_info:
        FeatureVector.model_validate(payload)

    errors = exc_info.value.errors()
    assert any(err["type"] == "extra_forbidden" for err in errors)


def test_string_for_numeric_field():
    """Test that string values for numeric fields are rejected."""
    payload = create_valid_payload()
    payload["Amount"] = "not_a_number"

    with pytest.raises(ValidationError):
        FeatureVector.model_validate(payload)


def test_nullable_fields():
    """Test that some fields allow None values."""
    payload = create_valid_payload()
    payload["sender_amount_mean_30d"] = None
    payload["sender_amount_std_30d"] = None

    result = FeatureVector.model_validate(payload)

    assert result.sender_amount_mean_30d is None
    assert result.sender_amount_std_30d is None


def test_all_required_features_present():
    """Test that all required fields are present and validated."""
    payload = create_valid_payload()

    result = FeatureVector.model_validate(payload)

    assert result.Amount == payload["Amount"]
    assert result.Payment_type == payload["Payment_type"]
    assert result.sender_counterparty_hhi == payload["sender_counterparty_hhi"]


def test_hhi_range():
    """Test that HHI is constrained to [0, 1]."""
    payload = create_valid_payload()
    payload["sender_counterparty_hhi"] = 1.5

    with pytest.raises(ValidationError):
        FeatureVector.model_validate(payload)


def test_model_dump():
    """Test model_dump() serialization."""
    payload = create_valid_payload()
    model = FeatureVector.model_validate(payload)

    dumped = model.model_dump()

    assert dumped["Amount"] == 1000.0
    assert dumped["Payment_currency"] == "USD"
    assert "unknown_field" not in dumped
