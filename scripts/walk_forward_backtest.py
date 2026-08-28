import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.evaluation.metrics import evaluate_model, precision_recall_at_alert_rate
from src.logging_config import setup_logging
from src.models.calibration import ProbabilityCalibrator
from src.models.train import MODEL_FEATURES, TARGET, fit_model

logger = setup_logging(__name__)


def run_backtest(frame, config):
    logger.info(f"Starting walk-forward backtest on {len(frame):,} transactions")
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    timestamps = ordered["timestamp"].drop_duplicates().sort_values().tolist()
    block_count = 14
    if len(timestamps) < block_count:
        raise ValueError("walk-forward backtesting requires at least 14 unique timestamps")

    logger.info(f"Found {len(timestamps)} unique timestamps, creating rolling windows...")
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
            logger.debug(f"Window {window + 1}: Skipping (insufficient target variance in calibration)")
            continue
        if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
            logger.debug(f"Window {window + 1}: Skipping (insufficient target variance in train/test)")
            continue

        logger.info(f"Processing window {window + 1}: train={len(train):,}, calib={len(calibration):,}, test={len(test):,}")
        preprocessor, model, calibration_x, calibration_y = fit_model(
            train,
            calibration,
            MODEL_FEATURES,
            config=config,
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

    logger.info(f"Loading feature dataset from {args.features}...")
    df = pd.read_parquet(args.features)
    logger.info(f"Loaded {len(df):,} transactions")

    config = get_config(fast=args.fast)
    if args.fast:
        logger.info("Using FAST mode configuration")

    results = run_backtest(df, config=config)

    if not results:
        raise RuntimeError("no valid walk-forward windows were available")

    logger.info(f"Completed {len(results)} walk-forward windows")
    logger.info("WALK-FORWARD BACKTEST RESULTS:")
    for result in results:
        logger.info(
            f"  {result['window']}: PR-AUC={result['pr_auc']:.6f} | "
            f"Recall@0.5%={result['recall_at_0.5%']:.6f} | "
            f"Lift@0.5%={result['lift_at_0.5%']:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results written to {args.output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
