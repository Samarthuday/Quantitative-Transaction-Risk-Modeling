from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config import get_config
from src.data.loader import load_saml_data
from src.features.behavioral_features import add_behavioral_features
from src.features.transaction_features import add_transaction_features
from src.models.calibration import ProbabilityCalibrator
from src.models.train import MODEL_FEATURES, TARGET, fit_model, temporal_split


def synthetic_features(rows=48):
    import numpy as np

    records = []
    for index in range(rows):
        # Create valid synthetic features
        record = {}

        # Numeric features with valid ranges
        record["Amount"] = float(100 * ((index % 7) + 1))
        record["log_amount"] = np.log1p(record["Amount"])

        # Cyclical encodings must be in [-1, 1]
        record["hour_sin"] = np.sin(2 * np.pi * (index % 24) / 24)
        record["hour_cos"] = np.cos(2 * np.pi * (index % 24) / 24)
        record["dow_sin"] = np.sin(2 * np.pi * (index % 7) / 7)
        record["dow_cos"] = np.cos(2 * np.pi * (index % 7) / 7)
        record["month_sin"] = np.sin(2 * np.pi * (index % 12) / 12)
        record["month_cos"] = np.cos(2 * np.pi * (index % 12) / 12)

        # Binary features
        record["is_weekend"] = int(index % 7 >= 5)
        record["is_night"] = int((index % 24) in [22, 23, 0, 1, 2, 3, 4, 5, 6])
        record["currency_mismatch"] = int(index % 3 == 0)
        record["cross_border"] = int(index % 2 == 0)
        record["is_round_amount"] = int(index % 5 == 0)

        # Behavioral features
        record["sender_txn_count_24h"] = max(0, index % 10)
        record["sender_amount_sum_24h"] = float(index * 100 % 5000)
        record["sender_amount_mean_30d"] = float(index * 50 % 1000)
        record["sender_amount_std_30d"] = float(index * 30 % 500)
        record["sender_amount_zscore"] = float((index - 5) / 5)

        record["receiver_txn_count_24h"] = max(0, index % 8)
        record["receiver_amount_sum_24h"] = float(index * 80 % 4000)
        record["seconds_since_sender_txn"] = max(-1, (index - 1) * 3600)

        # Network features
        record["sender_txn_count_lifetime"] = max(0, index * 2)
        record["receiver_txn_count_lifetime"] = max(0, index * 2 - 5)
        record["sender_out_degree"] = max(0, index % 30)
        record["receiver_in_degree"] = max(0, index % 25)
        record["pair_transaction_count"] = max(0, index % 10)
        record["sender_counterparty_hhi"] = float((index % 100) / 100)

        # Categorical features
        record["Payment_type"] = "Transfer"
        record["Payment_currency"] = "USD"
        record["Received_currency"] = "USD"
        record["Sender_bank_location"] = "US"
        record["Receiver_bank_location"] = "US"

        # Metadata
        record["timestamp"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(
            index,
            unit="D",
        )
        record[TARGET] = index % 2

        records.append(record)

    return pd.DataFrame(records)


def test_training_artifact_and_api_prediction(tmp_path: Path):
    raw_frame = pd.DataFrame(
        {
            "Time": ["12:00:00", "12:01:00"],
            "Date": ["2026-01-01", "2026-01-02"],
            "Sender_account": ["sender-1", "sender-1"],
            "Receiver_account": ["receiver-1", "receiver-2"],
            "Amount": [100.0, 200.0],
            "Payment_currency": ["USD", "USD"],
            "Received_currency": ["USD", "USD"],
            "Sender_bank_location": ["US", "US"],
            "Receiver_bank_location": ["US", "US"],
            "Payment_type": ["Transfer", "Transfer"],
            "Is_laundering": [0, 1],
            "Laundering_type": ["None", "Structuring"],
        }
    )
    raw_path = tmp_path / "mini_saml.csv"
    raw_frame.to_csv(raw_path, index=False)
    loaded = load_saml_data(raw_path)
    engineered = add_behavioral_features(add_transaction_features(loaded))
    assert "sender_counterparty_hhi" in engineered

    frame = synthetic_features()
    train, calibration, _, test = temporal_split(
        frame,
        train_fraction=0.5,
        calibration_fraction=0.2,
        validation_fraction=0.1,
    )
    config = get_config(fast=True)
    preprocessor, model, calibration_x, calibration_y = fit_model(
        train,
        calibration,
        config=config,
    )
    calibration_probabilities = model.predict_proba(calibration_x)[:, 1]
    calibrator = ProbabilityCalibrator().fit(
        calibration_probabilities,
        calibration_y,
    )
    artifact_path = tmp_path / "risk_model.joblib"
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "calibrator": calibrator,
            "features": MODEL_FEATURES,
            "decision_threshold": 0.5,
            "alert_rate": 0.005,
            "test_metrics": {},
            "model_version": "test",
        },
        artifact_path,
    )

    client = TestClient(create_app(artifact_path))
    response = client.post(
        "/api/predict",
        json=test.iloc[0][MODEL_FEATURES].to_dict(),
    )

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["risk_probability"] <= 1
    assert isinstance(body["requires_review"], bool)
