import numpy as np
import pandas as pd

from src.features.behavioral_features import add_behavioral_features
from src.features.transaction_features import add_transaction_features


def test_transaction_features_do_not_encode_account_ids():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 12:00:00"]),
            "Sender_account": ["sender-001"],
            "Receiver_account": ["receiver-001"],
            "Amount": [1000.0],
            "Payment_currency": ["UK pounds"],
            "Received_currency": ["UK pounds"],
            "Sender_bank_location": ["UK"],
            "Receiver_bank_location": ["UK"],
        }
    )

    result = add_transaction_features(frame)

    assert "log_amount" in result
    assert "currency_mismatch" in result
    assert result["Sender_account"].dtype == object
    assert result["Receiver_account"].dtype == object


def test_transaction_features_handles_zero_amounts():
    """Test that zero and near-zero amounts are handled correctly."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "Sender_account": ["s1", "s2"],
            "Receiver_account": ["r1", "r2"],
            "Amount": [0.0, 0.01],
            "Payment_currency": ["USD", "USD"],
            "Received_currency": ["USD", "USD"],
            "Sender_bank_location": ["US", "US"],
            "Receiver_bank_location": ["US", "US"],
        }
    )

    result = add_transaction_features(frame)

    assert result["log_amount"].notna().all()
    assert not np.isinf(result["log_amount"]).any()


def test_transaction_features_cyclical_encoding():
    """Test that cyclical time encoding produces values in [-1, 1]."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([
                "2026-01-01 00:00:00",
                "2026-01-01 12:00:00",
                "2026-01-01 23:59:59",
                "2026-12-31 23:59:59",
            ]),
            "Sender_account": ["s1", "s2", "s3", "s4"],
            "Receiver_account": ["r1", "r2", "r3", "r4"],
            "Amount": [100.0] * 4,
            "Payment_currency": ["USD"] * 4,
            "Received_currency": ["USD"] * 4,
            "Sender_bank_location": ["US"] * 4,
            "Receiver_bank_location": ["US"] * 4,
        }
    )

    result = add_transaction_features(frame)

    cyclical_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    for col in cyclical_cols:
        assert (result[col].abs() <= 1.0).all(), f"{col} outside [-1, 1]"


def test_transaction_features_preserves_row_count():
    """Test that feature engineering doesn't add or remove rows."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([f"2026-01-{i:02d}" for i in range(1, 11)]),
            "Sender_account": [f"s{i}" for i in range(10)],
            "Receiver_account": [f"r{i}" for i in range(10)],
            "Amount": np.random.rand(10) * 1000,
            "Payment_currency": ["USD"] * 10,
            "Received_currency": ["USD"] * 10,
            "Sender_bank_location": ["US"] * 10,
            "Receiver_bank_location": ["US"] * 10,
        }
    )

    result = add_transaction_features(frame)

    assert len(result) == len(frame)


def test_behavioral_features_handles_missing_history():
    """Test that behavioral features handle transactions with no prior history."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 12:00:00"]),
            "Sender_account": ["sender-001"],
            "Receiver_account": ["receiver-001"],
            "Amount": [1000.0],
            "Payment_currency": ["USD"],
            "Received_currency": ["USD"],
            "Sender_bank_location": ["US"],
            "Receiver_bank_location": ["US"],
            "Payment_type": ["Transfer"],
            "Is_laundering": [0],
            "Laundering_type": ["None"],
        }
    )

    # Add transaction features first
    frame = add_transaction_features(frame)

    # Then behavioral features
    result = add_behavioral_features(frame)

    # Check that all numeric behavioral features are populated
    behavioral_cols = [
        "sender_txn_count_24h",
        "sender_amount_sum_24h",
        "seconds_since_sender_txn",
        "sender_amount_zscore",
    ]

    for col in behavioral_cols:
        if col in result.columns:
            assert result[col].notna().all(), f"{col} has NaN values"
