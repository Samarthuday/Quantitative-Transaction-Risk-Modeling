"""Artifact-backed API for feature-store supplied transaction features.

This API intentionally does not calculate behavioural history from a single
raw transaction. That responsibility belongs to an online feature store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.models.inference import (
    model_input_from_features,
    predict_calibrated_probability,
    probability_percentile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts/risk_model.joblib"


def create_app(model_path: Path = MODEL_PATH) -> Flask:
    app = Flask(__name__)
    CORS(app)

    artifact: dict[str, Any] | None = None
    load_error: str | None = None
    try:
        artifact = joblib.load(model_path)
    except FileNotFoundError:
        load_error = f"Model artifact not found at {model_path}. Train the offline model first."
    except Exception as error:  # pragma: no cover - defensive startup path
        load_error = f"Unable to load model artifact: {error}"

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "healthy" if artifact else "model_unavailable",
                "model_loaded": artifact is not None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detail": load_error,
            }
        )

    @app.get("/api/model/info")
    def model_info():
        if artifact is None:
            return jsonify({"error": load_error}), 503
        return jsonify(
            {
                "model": "XGBoost",
                "model_version": artifact["model_version"],
                "alert_rate": artifact["alert_rate"],
                "decision_threshold": artifact["decision_threshold"],
                "test_metrics": artifact["test_metrics"],
            }
        )

    @app.post("/api/predict")
    def predict():
        if artifact is None:
            return jsonify({"error": load_error}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        try:
            features = model_input_from_features(payload, artifact["features"])
            probability = predict_calibrated_probability(artifact, features)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        result: dict[str, Any] = {
            "risk_probability": probability,
            "requires_review": probability >= artifact["decision_threshold"],
        }
        percentile = probability_percentile(artifact, probability)
        if percentile is not None:
            result["risk_percentile"] = percentile
        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
