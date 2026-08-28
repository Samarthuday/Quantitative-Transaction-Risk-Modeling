from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.inference import model_input_from_features
from src.api.main import create_app


@pytest.mark.asyncio
async def test_inference_rejects_identifiers_and_targets():
    """Test that async inference rejects account identifiers."""
    with pytest.raises(ValueError, match="Identifier or target fields"):
        await model_input_from_features(
            {"Amount": 100.0, "Sender_account": "not-a-feature"},
            ["Amount"],
        )


@pytest.mark.asyncio
async def test_inference_requires_feature_store_output():
    """Test that async inference requires all model features."""
    with pytest.raises(ValueError, match="Missing model features"):
        await model_input_from_features(
            {"Amount": 100.0},
            ["Amount", "sender_txn_count_24h"],
        )


def test_api_reports_missing_model_artifact(tmp_path: Path):
    """Test that API reports missing model artifact correctly."""
    app = create_app(tmp_path / "absent.joblib")
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
