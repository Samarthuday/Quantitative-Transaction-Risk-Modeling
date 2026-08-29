"""Unseen-entity generalization: does performance hold when sender or
receiver was never observed during training, or is it driven by
account-history features (lifetime counts, degree, HHI) that only work
for accounts the model has already seen?

No retraining: reuses the artifact trained on the standard temporal split
and evaluates it on three partitions of the same held-out test set.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    evaluate_model,
    metrics_at_threshold,
    precision_recall_at_alert_rate,
)
from src.logging_config import setup_logging
from src.models.train import MODEL_FEATURES, TARGET, temporal_split

logger = setup_logging(__name__)

ARTIFACT_PATH = PROJECT_ROOT / "artifacts/risk_model.joblib"
FEATURE_PATH = PROJECT_ROOT / "data/processed/transactions_features.parquet"
OUTPUT_PATH = PROJECT_ROOT / "reports/unseen_entity_results.json"
FIGURE_PATH = PROJECT_ROOT / "reports/figures/unseen_entity_generalization.png"
ALERT_RATE = 0.005


def summarize_fixed_threshold(
    mask: np.ndarray, y_test: np.ndarray, calibrated_prob: np.ndarray, threshold: float, label: str
) -> dict:
    """What the artifact's actual production threshold does to this subgroup,
    as opposed to summarize()'s per-subgroup top-K (best ranking within the
    subgroup alone)."""
    subset_y = y_test[mask]
    subset_prob = calibrated_prob[mask]
    if len(subset_y) == 0:
        return {"label": label, "metrics": None}
    return {"label": label, "metrics": metrics_at_threshold(subset_y, subset_prob, threshold)}


def summarize(mask: np.ndarray, y_test: np.ndarray, calibrated_prob: np.ndarray, label: str) -> dict:
    subset_y = y_test[mask]
    subset_prob = calibrated_prob[mask]

    positives = int(subset_y.sum())
    result = {
        "label": label,
        "transactions": int(mask.sum()),
        "positives": positives,
        "prevalence": float(subset_y.mean()) if len(subset_y) else None,
        "metrics": None,
    }

    if len(subset_y) == 0 or positives == 0 or positives == len(subset_y):
        logger.warning(
            f"{label}: insufficient class variance ({positives} positives / "
            f"{len(subset_y)} rows) -- metrics skipped"
        )
        return result

    metrics = evaluate_model(subset_y, subset_prob)
    budget = precision_recall_at_alert_rate(subset_y, subset_prob, alert_rate=ALERT_RATE)
    result["metrics"] = {
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "precision_at_alert_rate": budget["precision"],
        "recall_at_alert_rate": budget["recall"],
        "lift_at_alert_rate": budget["lift"],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURE_PATH)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args()

    logger.info(f"Loading artifact from {args.artifact}...")
    artifact = joblib.load(args.artifact)
    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    calibrator = artifact["calibrator"]

    logger.info(f"Loading features from {args.features}...")
    df = pd.read_parquet(args.features)

    train, _, _, test = temporal_split(df)

    seen_accounts = set(train["Sender_account"]) | set(train["Receiver_account"])
    logger.info(f"Accounts observed during training: {len(seen_accounts):,}")

    sender_seen = test["Sender_account"].isin(seen_accounts).to_numpy()
    receiver_seen = test["Receiver_account"].isin(seen_accounts).to_numpy()
    both_seen = sender_seen & receiver_seen
    unseen_entity = ~both_seen

    logger.info(
        f"Test set: {len(test):,} transactions | "
        f"{both_seen.sum():,} both-parties-seen ({both_seen.mean():.2%}) | "
        f"{unseen_entity.sum():,} involve an unseen account ({unseen_entity.mean():.2%})"
    )

    X_test_processed = preprocessor.transform(test[MODEL_FEATURES])
    raw_prob = model.predict_proba(X_test_processed)[:, 1]
    calibrated_prob = calibrator.predict(raw_prob)
    y_test = test[TARGET].to_numpy()

    standard = summarize(np.ones(len(y_test), dtype=bool), y_test, calibrated_prob, "Standard out-of-time (full test set)")
    seen_result = summarize(both_seen, y_test, calibrated_prob, "Both sender and receiver seen during training")
    unseen_result = summarize(unseen_entity, y_test, calibrated_prob, "At least one party unseen during training")

    decision_threshold = artifact["decision_threshold"]
    fixed_threshold_seen = summarize_fixed_threshold(
        both_seen, y_test, calibrated_prob, decision_threshold, "Both parties seen"
    )
    fixed_threshold_unseen = summarize_fixed_threshold(
        unseen_entity, y_test, calibrated_prob, decision_threshold, "Unseen entity"
    )

    results = {
        "alert_rate": ALERT_RATE,
        "accounts_seen_in_training": len(seen_accounts),
        "standard_out_of_time": standard,
        "both_parties_seen": seen_result,
        "unseen_entity": unseen_result,
        "fixed_production_threshold": {
            "decision_threshold": decision_threshold,
            "both_parties_seen": fixed_threshold_seen,
            "unseen_entity": fixed_threshold_unseen,
        },
    }

    logger.info("=" * 70)
    logger.info("UNSEEN-ENTITY GENERALIZATION RESULTS (per-subgroup top-K)")
    logger.info("=" * 70)
    for key in ("standard_out_of_time", "both_parties_seen", "unseen_entity"):
        r = results[key]
        logger.info(f"\n{r['label']}:")
        prevalence = f"{r['prevalence']:.4%}" if r["prevalence"] is not None else "n/a"
        logger.info(f"  Transactions: {r['transactions']:,} | Positives: {r['positives']:,} | Prevalence: {prevalence}")
        if r["metrics"]:
            for metric_key, value in r["metrics"].items():
                logger.info(f"  {metric_key:30s}: {value:.6f}")

    logger.info("=" * 70)
    logger.info(f"SAME FIXED PRODUCTION THRESHOLD ({decision_threshold:.6f}) APPLIED TO BOTH SUBGROUPS")
    logger.info("=" * 70)
    for r in (fixed_threshold_seen, fixed_threshold_unseen):
        logger.info(f"\n{r['label']}:")
        if r["metrics"]:
            for metric_key, value in r["metrics"].items():
                logger.info(f"  {metric_key:30s}: {value:.6f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    logger.info(f"\nResults written to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    chart_labels = {
        "standard_out_of_time": "Standard\nOut-of-Time",
        "both_parties_seen": "Both Parties\nSeen",
        "unseen_entity": "Unseen\nEntity",
    }
    labels = []
    pr_aucs = []
    for key in ("standard_out_of_time", "both_parties_seen", "unseen_entity"):
        r = results[key]
        if r["metrics"] is None:
            continue
        labels.append(chart_labels[key])
        pr_aucs.append(r["metrics"]["pr_auc"])

    if pr_aucs:
        FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#06b6d4", "#10b981", "#f59e0b"]
        bars = ax.bar(labels, pr_aucs, color=colors[: len(labels)], edgecolor="black", linewidth=1.5)
        ax.set_ylabel("PR-AUC Score", fontsize=12, fontweight="bold")
        ax.set_title("Generalization to Unseen Entities", fontsize=14, fontweight="bold", pad=20)
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.4f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved comparison chart to {FIGURE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
