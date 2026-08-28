import pandas as pd

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
