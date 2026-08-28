"""FastAPI application for transaction risk prediction inference."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.inference import (
    model_input_from_features,
    predict_calibrated_probability,
    probability_percentile,
)
from src.api.models import (
    FeatureVector,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.api.results_loader import (
    get_ablation_results,
    get_typology_results,
    get_all_figures,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts/risk_model.joblib"
PUBLIC_DIR = PROJECT_ROOT / "public"


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    """Create and configure FastAPI application with optional model artifact."""

    app = FastAPI(
        title="Transaction Risk Prediction API",
        description="XGBoost-based inference for transaction risk scoring",
        version="2.1.0",
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load model artifact at startup
    artifact: Optional[dict[str, Any]] = None
    load_error: Optional[str] = None

    try:
        artifact = joblib.load(model_path)
    except FileNotFoundError:
        load_error = f"Model artifact not found at {model_path}. Train the offline model first."
    except Exception as error:
        load_error = f"Unable to load model artifact: {error}"

    # Routes

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Service health check."""
        return HealthResponse(
            status="healthy" if artifact else "model_unavailable",
            model_loaded=artifact is not None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            detail=load_error,
        )

    @app.get("/api/model/info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        """Model metadata and evaluation metrics."""
        if artifact is None:
            raise HTTPException(status_code=503, detail=load_error)

        return ModelInfoResponse(
            model="XGBoost",
            model_version=artifact["model_version"],
            alert_rate=artifact["alert_rate"],
            decision_threshold=artifact["decision_threshold"],
            test_metrics=artifact["test_metrics"],
        )

    @app.get("/api/results")
    async def get_results() -> dict[str, Any]:
        """Analysis results: ablation study, behavioral typology, and all report figures."""
        return {
            "ablation": get_ablation_results(),
            "typology": get_typology_results(),
            "figures": get_all_figures(),
        }

    @app.post("/api/predict", response_model=PredictionResponse)
    async def predict(features: FeatureVector) -> PredictionResponse:
        """Predict transaction risk probability."""
        if artifact is None:
            raise HTTPException(status_code=503, detail=load_error)

        try:
            # Validate features and construct input DataFrame
            feature_dict = features.model_dump(mode="json")
            model_features = artifact["features"]
            X = await model_input_from_features(feature_dict, model_features)

            # Run inference asynchronously
            probability = await predict_calibrated_probability(artifact, X)

            # Calculate percentile rank
            percentile = probability_percentile(artifact, probability)

            return PredictionResponse(
                risk_probability=probability,
                requires_review=probability >= artifact["decision_threshold"],
                risk_percentile=percentile,
            )

        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

    # Serve static SPA from public/ directory
    if PUBLIC_DIR.exists():
        app.mount("", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")

    return app


# Create module-level app instance for `uvicorn src.api.main:app`
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
