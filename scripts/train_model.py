import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    evaluate_model,
    threshold_for_alert_rate,
)
from src.models.calibration import (
    ProbabilityCalibrator,
)
from src.models.train import (
    MODEL_FEATURES,
    TARGET,
    chronological_split,
    fit_model,
)
from src.models.baseline import build_logistic_baseline

FEATURE_PATH = PROJECT_ROOT / "data/processed/transactions_features.parquet"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts/risk_model.joblib"


def print_split_stats(name, frame):

    positive = frame[TARGET].sum()
    prevalence = frame[TARGET].mean()

    print(
        f"{name}: "
        f"{len(frame):,} rows | "
        f"{positive:,} suspicious | "
        f"{prevalence:.4%}"
    )


def main():

    ARTIFACT_PATH.parent.mkdir(exist_ok=True)

    print("Loading feature dataset...")

    df = pd.read_parquet(
        FEATURE_PATH
    )

    train, validation, test = (
        chronological_split(df)
    )

    print_split_stats(
        "TRAIN",
        train,
    )

    print_split_stats(
        "VALIDATION",
        validation,
    )

    print_split_stats(
        "TEST",
        test,
    )

    (
        preprocessor,
        model,
        X_validation_processed,
        y_validation,
    ) = fit_model(
        train,
        validation,
    )

    # ---------------------------
    # CALIBRATION
    # ---------------------------

    raw_validation_prob = (
        model.predict_proba(
            X_validation_processed
        )[:, 1]
    )

    calibrator = ProbabilityCalibrator()

    calibrator.fit(
        raw_validation_prob,
        y_validation,
    )

    calibrated_validation_prob = (
        calibrator.predict(
            raw_validation_prob
        )
    )

    # Use a realistic operational alert capacity
    alert_rate = 0.005

    threshold = (
        threshold_for_alert_rate(
            calibrated_validation_prob,
            alert_rate,
        )
    )

    print(
        f"\nThreshold for "
        f"{alert_rate:.2%} alert rate: "
        f"{threshold:.6f}"
    )

    # ---------------------------
    # TEST
    # ---------------------------

    X_test = test[MODEL_FEATURES]
    y_test = test[TARGET]

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    raw_test_prob = (
        model.predict_proba(
            X_test_processed
        )[:, 1]
    )

    calibrated_test_prob = (
        calibrator.predict(
            raw_test_prob
        )
    )

    metrics = evaluate_model(
        y_test,
        calibrated_test_prob,
    )

    print("\nOUT-OF-TIME TEST RESULTS")
    print("=" * 50)

    for key, value in metrics.items():

        if isinstance(value, float):
            print(
                f"{key:30s}: "
                f"{value:.6f}"
            )

        else:
            print(
                f"{key:30s}: "
                f"{value}"
            )

    # Benchmark the nonlinear model against a scalable logistic classifier on
    # the exact same chronological partitions and preprocessing contract.
    baseline = build_logistic_baseline()
    baseline.fit(preprocessor.transform(train[MODEL_FEATURES]), train[TARGET])
    baseline_validation_prob = baseline.predict_proba(
        X_validation_processed
    )[:, 1]
    baseline_calibrator = ProbabilityCalibrator().fit(
        baseline_validation_prob,
        y_validation,
    )
    baseline_test_prob = baseline_calibrator.predict(
        baseline.predict_proba(X_test_processed)[:, 1]
    )
    baseline_metrics = evaluate_model(y_test, baseline_test_prob)

    print("\nLOGISTIC BASELINE TEST RESULTS")
    print("=" * 50)
    for key in ("pr_auc", "alert_0.500%_recall", "alert_0.500%_lift"):
        print(f"{key:30s}: {baseline_metrics[key]:.6f}")

    # ---------------------------
    # SAVE
    # ---------------------------

    artifact = {
        "preprocessor": preprocessor,
        "model": model,
        "calibrator": calibrator,

        "features": MODEL_FEATURES,

        "decision_threshold": threshold,
        "alert_rate": alert_rate,

        "test_metrics": metrics,
        "baseline_test_metrics": baseline_metrics,
        "validation_probability_quantiles": [
            float(value)
            for value in sorted(calibrated_validation_prob)
        ],

        "model_version": "2.0.0",
    }

    joblib.dump(
        artifact,
        ARTIFACT_PATH,
    )

    print(
        "\nSaved model to "
        f"{ARTIFACT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
