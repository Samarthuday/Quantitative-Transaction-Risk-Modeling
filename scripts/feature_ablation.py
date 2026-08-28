import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.metrics import (
    evaluate_model,
    precision_recall_at_alert_rate,
)
from src.models.calibration import ProbabilityCalibrator
from src.models.train import (
    ABLATION_FEATURE_SETS,
    TARGET,
    fit_model,
    temporal_split,
)

FEATURE_PATH = PROJECT_ROOT / "data/processed/transactions_features.parquet"
ALERT_RATE = 0.005


def evaluate_feature_set(name, features, train, calibration, validation, test, fast=False):
    (
        preprocessor,
        model,
        X_validation_processed,
        y_validation,
    ) = fit_model(train, calibration, features, fast=fast)

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

    df = pd.read_parquet(args.features)
    train, calibration, validation, test = temporal_split(df)

    results = [
        evaluate_feature_set(
            name,
            features,
            train,
            calibration,
            validation,
            test,
            args.fast,
        )
        for name, features in ABLATION_FEATURE_SETS.items()
    ]

    print("\nFEATURE ABLATION RESULTS")
    print("=" * 55)
    print(f"{'Model':<22} {'PR-AUC':>12} {'Recall@0.5%':>16}")
    print("-" * 55)
    for result in results:
        print(
            f"{result['model']:<22} "
            f"{result['pr_auc']:>12.6f} "
            f"{result['recall_at_0.5%']:>16.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
