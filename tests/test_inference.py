from pathlib import Path

import pytest

from src.api.app import create_app
from src.models.inference import model_input_from_features


def test_inference_rejects_identifiers_and_targets():
    with pytest.raises(ValueError, match="Identifier or target fields"):
        model_input_from_features(
            {"Amount": 100.0, "Sender_account": "not-a-feature"},
            ["Amount"],
        )


def test_inference_requires_feature_store_output():
    with pytest.raises(ValueError, match="Missing model features"):
        model_input_from_features({"Amount": 100.0}, ["Amount", "sender_txn_count_24h"])


def test_api_reports_missing_model_artifact(tmp_path: Path):
    app = create_app(tmp_path / "absent.joblib")
    response = app.test_client().get("/api/health")

    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is False
