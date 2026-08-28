import argparse
import csv
import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.logging_config import setup_logging

logger = setup_logging(__name__)

from src.evaluation.metrics import (
    evaluate_model,
    top_k_alert_mask,
)
from src.models.baseline import build_logistic_baseline
from src.models.calibration import (
    ProbabilityCalibrator,
)
from src.models.train import (
    MODEL_FEATURES,
    TARGET,
    fit_model,
    temporal_split,
)

FEATURE_PATH = PROJECT_ROOT / "data/processed/transactions_features.parquet"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts/risk_model.joblib"


def package_versions():
    packages = ["numpy", "pandas", "scikit-learn", "xgboost", "duckdb", "joblib"]
    return {
        package: version(package)
        for package in packages
    }


def print_split_stats(name, frame):

    positive = frame[TARGET].sum()
    prevalence = frame[TARGET].mean()

    logger.info(
        f"{name}: "
        f"{len(frame):,} rows | "
        f"{positive:,} suspicious | "
        f"{prevalence:.4%}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURE_PATH)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    args.artifact.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading feature dataset from {args.features}...")

    df = pd.read_parquet(
        args.features
    )
    logger.info(f"Loaded {len(df):,} transactions with {df.shape[1]} features")

    logger.info("Performing temporal split (65% train / 10% calibration / 10% validation / 15% test)...")
    train, calibration, validation, test = temporal_split(df)

    config = get_config(fast=args.fast)
    if args.fast:
        logger.info("Using FAST mode configuration")
    logger.info(f"XGBoost config: n_estimators={config.xgboost.n_estimators}, max_depth={config.xgboost.max_depth}")

    print_split_stats(
        "TRAIN",
        train,
    )

    print_split_stats(
        "CALIBRATION",
        calibration,
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
        calibration,
        config=config,
    )

    # ---------------------------
    # CALIBRATION
    # ---------------------------

    raw_calibration_prob = (
        model.predict_proba(
            X_validation_processed
        )[:, 1]
    )

    calibrator = ProbabilityCalibrator()

    calibrator.fit(
        raw_calibration_prob,
        y_validation,
    )

    calibrated_validation_prob = (
        calibrator.predict(raw_calibration_prob)
    )

    calibration_report = {
        "raw_brier_score": float(
            evaluate_model(y_validation, raw_calibration_prob)["brier_score"]
        ),
        "calibrated_brier_score": float(
            evaluate_model(y_validation, calibrated_validation_prob)["brier_score"]
        ),
        "raw_log_loss": float(
            evaluate_model(y_validation, raw_calibration_prob)["log_loss"]
        ),
        "calibrated_log_loss": float(
            evaluate_model(y_validation, calibrated_validation_prob)["log_loss"]
        ),
    }
    figure_dir = PROJECT_ROOT / "reports/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 6))
    for probabilities, label in (
        (raw_calibration_prob, "Raw XGBoost"),
        (calibrated_validation_prob, "Calibrated XGBoost"),
    ):
        observed, predicted = calibration_curve(
            y_validation,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title("Probability calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "calibration_curve.png", dpi=150)
    plt.close(figure)

    # Use a realistic operational alert capacity
    alert_rate = 0.005

    calibration_alerts = top_k_alert_mask(
        calibrated_validation_prob,
        alert_rate,
    )
    threshold = float(calibrated_validation_prob[calibration_alerts].min())

    logger.info(
        f"Threshold for {alert_rate:.2%} alert rate: {threshold:.6f}"
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

    typology_results = []
    if "Laundering_type" in test:
        alert_mask = top_k_alert_mask(calibrated_test_prob, alert_rate)
        for typology, group in test.groupby("Laundering_type", dropna=False):
            group_mask = group.index.isin(test.index[alert_mask])
            positives = group[TARGET].sum()
            typology_results.append(
                {
                    "typology": str(typology),
                    "transactions": int(len(group)),
                    "positives": int(positives),
                    "recall_at_0.5%": float(
                        group_mask[group[TARGET].to_numpy() == 1].mean()
                        if positives
                        else 0
                    ),
                }
            )
        typology_path = PROJECT_ROOT / "reports/typology_results.csv"
        with typology_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=typology_results[0].keys())
            writer.writeheader()
            writer.writerows(typology_results)

    logger.info("OUT-OF-TIME TEST RESULTS:")

    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key:30s}: {value:.6f}")
        else:
            logger.info(f"  {key:30s}: {value}")

    logger.info("Training logistic baseline for comparison...")
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

    logger.info("LOGISTIC BASELINE TEST RESULTS:")
    for key in ("pr_auc", "alert_0.500%_recall", "alert_0.500%_lift"):
        logger.info(f"  {key:30s}: {baseline_metrics[key]:.6f}")

    # ---------------------------
    # SAVE
    # ---------------------------

    artifact = {
        "preprocessor": preprocessor,
        "model": model,
        "calibrator": calibrator,
        "calibration_metrics": calibration_report,

        "features": MODEL_FEATURES,

        "decision_threshold": threshold,
        "alert_rate": alert_rate,

        "test_metrics": metrics,
        "baseline_test_metrics": baseline_metrics,
        "validation_probability_quantiles": [
            float(value)
            for value in np.quantile(
                calibrated_validation_prob,
                np.linspace(0, 1, 1001),
            )
        ],

        "training_start": train["timestamp"].min().isoformat(),
        "training_end": train["timestamp"].max().isoformat(),
        "test_start": test["timestamp"].min().isoformat(),
        "test_end": test["timestamp"].max().isoformat(),
        "training_prevalence": float(train[TARGET].mean()),
        "validation_prevalence": float(validation[TARGET].mean()),
        "test_prevalence": float(test[TARGET].mean()),
        "model_parameters": model.get_params(),
        "package_versions": package_versions(),

        "model_version": "2.1.0",
    }

    logger.info(f"Saving model artifact to {args.artifact}...")
    joblib.dump(artifact, args.artifact)
    logger.info(f"Model saved successfully")

    logger.info(f"Writing metrics report...")
    report_path = PROJECT_ROOT / "reports/model_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_version": artifact["model_version"],
        "test_metrics": metrics,
        "baseline_test_metrics": baseline_metrics,
        "training_prevalence": artifact["training_prevalence"],
        "test_prevalence": artifact["test_prevalence"],
        "calibration": calibration_report,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info(f"Metrics report written to {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
