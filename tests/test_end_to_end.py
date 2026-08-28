from pathlib import Path

import joblib
import pandas as pd

from src.api.app import create_app
from src.data.loader import load_saml_data
from src.features.behavioral_features import add_behavioral_features
from src.features.transaction_features import add_transaction_features
from src.models.calibration import ProbabilityCalibrator
from src.models.train import MODEL_FEATURES, TARGET, fit_model, temporal_split


def synthetic_features(rows=48):
    records = []
    for index in range(rows):
        record = {feature: float((index % 7) + 1) for feature in MODEL_FEATURES}
        for feature in MODEL_FEATURES:
            if feature in {
                "Payment_type",
                "Payment_currency",
                "Received_currency",
                "Sender_bank_location",
                "Receiver_bank_location",
            }:
                record[feature] = "USD"
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
    preprocessor, model, calibration_x, calibration_y = fit_model(
        train,
        calibration,
        fast=True,
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

    client = create_app(artifact_path).test_client()
    response = client.post(
        "/api/predict",
        json=test.iloc[0][MODEL_FEATURES].to_dict(),
    )

    assert response.status_code == 200
    body = response.get_json()
    assert 0 <= body["risk_probability"] <= 1
    assert isinstance(body["requires_review"], bool)
