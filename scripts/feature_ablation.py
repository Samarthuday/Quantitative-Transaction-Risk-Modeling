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
    chronological_split,
    fit_model,
)

FEATURE_PATH = PROJECT_ROOT / "data/processed/transactions_features.parquet"
ALERT_RATE = 0.005


def evaluate_feature_set(name, features, train, validation, test):
    (
        preprocessor,
        model,
        X_validation_processed,
        y_validation,
    ) = fit_model(train, validation, features)

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
    df = pd.read_parquet(FEATURE_PATH)
    train, validation, test = chronological_split(df)

    results = [
        evaluate_feature_set(
            name,
            features,
            train,
            validation,
            test,
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


if __name__ == "__main__":
    main()
