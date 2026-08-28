import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_model, precision_recall_at_alert_rate
from src.models.calibration import ProbabilityCalibrator
from src.models.train import MODEL_FEATURES, TARGET, fit_model


def run_backtest(frame, fast=False):
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    timestamps = ordered["timestamp"].drop_duplicates().sort_values().tolist()
    block_count = 14
    if len(timestamps) < block_count:
        raise ValueError("walk-forward backtesting requires at least 8 timestamps")

    results = []
    for window in range(4):
        train_end = 4 + window * 2
        calibration_end = train_end + 2
        test_end = calibration_end + 2
        if test_end > len(timestamps):
            continue
        train = ordered[ordered["timestamp"] < timestamps[train_end]]
        calibration = ordered[
            (ordered["timestamp"] >= timestamps[train_end])
            & (ordered["timestamp"] < timestamps[calibration_end])
        ]
        test = ordered[
            (ordered["timestamp"] >= timestamps[calibration_end])
            & (ordered["timestamp"] < timestamps[test_end])
        ]
        if calibration[TARGET].nunique() < 2:
            continue
        if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
            continue

        preprocessor, model, calibration_x, calibration_y = fit_model(
            train,
            calibration,
            MODEL_FEATURES,
            fast=fast,
        )
        calibration_raw = model.predict_proba(calibration_x)[:, 1]
        calibrator = ProbabilityCalibrator().fit(calibration_raw, calibration_y)
        test_x = preprocessor.transform(test[MODEL_FEATURES])
        probabilities = calibrator.predict(model.predict_proba(test_x)[:, 1])
        metrics = evaluate_model(test[TARGET], probabilities)
        budget = precision_recall_at_alert_rate(
            test[TARGET], probabilities, alert_rate=0.005
        )
        results.append(
            {
                "window": f"W{window + 1}",
                "pr_auc": metrics["pr_auc"],
                "recall_at_0.5%": budget["recall"],
                "precision_at_0.5%": budget["precision"],
                "lift_at_0.5%": budget["lift"],
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data/processed/transactions_features.parquet")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/walk_forward_results.csv")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    results = run_backtest(pd.read_parquet(args.features), fast=args.fast)
    if not results:
        raise RuntimeError("no valid walk-forward windows were available")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
