import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_config
from src.evaluation.metrics import (
    evaluate_model,
    precision_recall_at_alert_rate,
)
from src.logging_config import setup_logging
from src.models.calibration import ProbabilityCalibrator
from src.models.train import (
    ABLATION_FEATURE_SETS,
    TARGET,
    fit_model,
    temporal_split,
)

logger = setup_logging(__name__)

FEATURE_PATH = PROJECT_ROOT / "data/processed/transactions_features.parquet"
ALERT_RATE = 0.005


def evaluate_feature_set(name, features, train, calibration, validation, test, config):
    logger.info(f"Evaluating feature set: '{name}' ({len(features)} features)")
    (
        preprocessor,
        model,
        X_validation_processed,
        y_validation,
    ) = fit_model(train, calibration, features, config=config)
    logger.debug(f"Model trained for '{name}'")

    validation_probabilities = model.predict_proba(
        X_validation_processed
    )[:, 1]

    calibrator = ProbabilityCalibrator().fit(
        validation_probabilities,
        y_validation,
    )

    test_probabilities = calibrator.predict(
        model.predict_proba(
            preprocessor.transform(test[features])
        )[:, 1]
    )

    metrics = evaluate_model(
        test[TARGET],
        test_probabilities,
    )
    alert_metrics = precision_recall_at_alert_rate(
        test[TARGET],
        test_probabilities,
        ALERT_RATE,
    )

    return {
        "model": name,
        "pr_auc": metrics["pr_auc"],
        "recall_at_0.5%": alert_metrics["recall"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURE_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/ablation_results.csv")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    logger.info(f"Loading feature dataset from {args.features}...")
    df = pd.read_parquet(args.features)
    logger.info(f"Loaded {len(df):,} transactions")

    logger.info("Performing temporal split...")
    train, calibration, validation, test = temporal_split(df)

    config = get_config(fast=args.fast)
    if args.fast:
        logger.info("Using FAST mode configuration")

    logger.info(f"Evaluating {len(ABLATION_FEATURE_SETS)} feature sets...")
    results = [
        evaluate_feature_set(
            name,
            features,
            train,
            calibration,
            validation,
            test,
            config,
        )
        for name, features in ABLATION_FEATURE_SETS.items()
    ]

    logger.info("FEATURE ABLATION RESULTS:")
    for result in results:
        logger.info(
            f"  {result['model']:<22} PR-AUC: {result['pr_auc']:.6f} | "
            f"Recall@0.5%: {result['recall_at_0.5%']:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Ablation results written to {args.output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
