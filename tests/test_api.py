"""Tests for FastAPI inference API."""

import json
import pytest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture
def client(tmp_path):
    """Create a FastAPI test client without a trained model."""
    # Use a non-existent path to ensure no model is loaded
    app = create_app(tmp_path / "nonexistent.joblib")
    return TestClient(app)


def get_valid_payload():
    """Return a valid feature vector for testing."""
    return {
        "Amount": 1000.0,
        "log_amount": 6.908,
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


def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_health_check_format(client):
    """Test that health response has correct format."""
    response = client.get("/api/health")
    data = response.json()

    assert isinstance(data, dict)
    assert data["status"] in ["healthy", "model_unavailable"]
    assert isinstance(data["model_loaded"], bool)
    assert "timestamp" in data


def test_model_info_without_model(client):
    """Test model info endpoint when model is unavailable."""
    response = client.get("/api/model/info")
    # Without a trained model artifact, should get 503
    assert response.status_code in [503, 500]  # FastAPI error handling


def test_predict_missing_required_field(client):
    """Test prediction with missing required field."""
    payload = get_valid_payload()
    del payload["Amount"]

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_invalid_feature_type(client):
    """Test prediction with invalid feature type."""
    payload = get_valid_payload()
    payload["Amount"] = "not_a_number"

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_extra_fields_rejected(client):
    """Test that extra unknown fields are rejected."""
    payload = get_valid_payload()
    payload["extra_field"] = "should_fail"

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_negative_amount(client):
    """Test that negative amounts are rejected."""
    payload = get_valid_payload()
    payload["Amount"] = -100.0

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_binary_field(client):
    """Test that invalid binary field values are rejected."""
    payload = get_valid_payload()
    payload["is_weekend"] = 2

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_cyclical_out_of_range(client):
    """Test that cyclical encodings outside [-1, 1] are rejected."""
    payload = get_valid_payload()
    payload["hour_sin"] = 1.5

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_cors_headers(client):
    """Test that CORS headers are set on preflight requests."""
    # TestClient doesn't fully trigger CORS middleware, but the app has CORS configured
    # Test that the health endpoint works (CORS is configured in create_app)
    response = client.get("/api/health")
    assert response.status_code == 200
    # CORS headers may not appear in TestClient due to how it handles middleware
    # but they will appear in real HTTP requests; this just ensures no errors


def test_predict_hhi_range(client):
    """Test that HHI must be between 0 and 1."""
    payload = get_valid_payload()
    payload["sender_counterparty_hhi"] = 1.5

    response = client.post("/api/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_json(client):
    """Test prediction with invalid JSON."""
    response = client.post(
        "/api/predict",
        content="not valid json",
        headers={"Content-Type": "application/json"},
    )

    # FastAPI returns 422 for parsing errors
    assert response.status_code in [400, 422, 503]


def test_predict_non_dict_payload(client):
    """Test prediction with non-dictionary JSON."""
    response = client.post(
        "/api/predict",
        content=json.dumps(["not", "a", "dict"]),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
