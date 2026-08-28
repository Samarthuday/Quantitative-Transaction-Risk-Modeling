"""Generate report figures (ROC, PR, feature importance, ablation, typology) from existing artifacts."""

import csv
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_config import setup_logging
from src.models.calibration import ProbabilityCalibrator
from src.models.train import MODEL_FEATURES, TARGET, temporal_split

logger = setup_logging(__name__)

ARTIFACT_PATH = PROJECT_ROOT / "artifacts/risk_model.joblib"
FEATURE_PATH = PROJECT_ROOT / "data/processed/transactions_features.parquet"
FIGURES_DIR = PROJECT_ROOT / "reports/figures"
ABLATION_PATH = PROJECT_ROOT / "reports/ablation_results.csv"
TYPOLOGY_PATH = PROJECT_ROOT / "reports/typology_results.csv"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading artifact from {ARTIFACT_PATH}...")
    artifact = joblib.load(ARTIFACT_PATH)
    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    calibrator = artifact["calibrator"]

    logger.info(f"Loading features from {FEATURE_PATH}...")
    df = pd.read_parquet(FEATURE_PATH)

    logger.info("Performing temporal split to recover test set...")
    _, _, _, test = temporal_split(df)
    X_test = test[MODEL_FEATURES]
    y_test = test[TARGET]

    logger.info("Running inference on test set...")
    X_test_processed = preprocessor.transform(X_test)
    raw_test_prob = model.predict_proba(X_test_processed)[:, 1]
    calibrated_test_prob = calibrator.predict(raw_test_prob)

    # ===== ROC Curve =====
    logger.info("Generating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, calibrated_test_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve (Test Set)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC curve to {FIGURES_DIR / 'roc_curve.png'}")

    # ===== Precision-Recall Curve =====
    logger.info("Generating Precision-Recall curve...")
    precision, recall, _ = precision_recall_curve(y_test, calibrated_test_prob)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, color="#06b6d4", lw=2.5, label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.fill_between(recall, precision, alpha=0.2, color="#06b6d4")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall Curve (Test Set)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved PR curve to {FIGURES_DIR / 'precision_recall_curve.png'}")

    # ===== Feature Importance (Top 15) =====
    logger.info("Generating feature importance chart...")
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(feature_importance_df)), feature_importance_df["importance"], color="#10b981")
    ax.set_yticks(range(len(feature_importance_df)))
    ax.set_yticklabels(feature_importance_df["feature"], fontsize=10)
    ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
    ax.set_title("Top 15 Most Important Features", fontsize=14, fontweight="bold", pad=20)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f" {width:.3f}",
                ha="left", va="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature importance to {FIGURES_DIR / 'feature_importance.png'}")

    # ===== Ablation Comparison =====
    logger.info("Generating ablation comparison chart...")
    ablation_data = []
    with ABLATION_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ablation_data.append({
                "model": row["model"],
                "pr_auc": float(row["pr_auc"])
            })

    if ablation_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = [d["model"] for d in ablation_data]
        pr_aucs = [d["pr_auc"] for d in ablation_data]
        colors = ["#94a3b8", "#a78bfa", "#06b6d4", "#10b981"]
        bars = ax.bar(models, pr_aucs, color=colors[:len(models)], edgecolor="black", linewidth=1.5)
        ax.set_ylabel("PR-AUC Score", fontsize=12, fontweight="bold")
        ax.set_title("Feature Ablation Study - PR-AUC Improvement", fontsize=14, fontweight="bold", pad=20)
        ax.set_ylim([0, 1.0])
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.4f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "ablation_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved ablation comparison to {FIGURES_DIR / 'ablation_comparison.png'}")

    # ===== Typology Detection =====
    logger.info("Generating typology detection chart...")
    typology_data = []
    with TYPOLOGY_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["positives"]) > 0:
                typology_data.append({
                    "typology": row["typology"],
                    "recall": float(row["recall_at_0.5%"])
                })

    typology_data = sorted(typology_data, key=lambda x: x["recall"], reverse=True)

    if typology_data:
        fig, ax = plt.subplots(figsize=(10, 10))
        typologies = [d["typology"] for d in typology_data]
        recalls = [d["recall"] for d in typology_data]
        colors = ["#10b981" if r >= 0.9 else "#f59e0b" if r >= 0.5 else "#ef4444" for r in recalls]
        bars = ax.barh(range(len(typologies)), recalls, color=colors, edgecolor="black", linewidth=1)
        ax.set_yticks(range(len(typologies)))
        ax.set_yticklabels(typologies, fontsize=9)
        ax.set_xlabel("Detection Rate (Recall @ 0.5% Alert Budget)", fontsize=11, fontweight="bold")
        ax.set_title("AML Behavioral Pattern Detection Rates", fontsize=14, fontweight="bold", pad=20)
        ax.set_xlim([0, 1.05])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2, f" {width:.1%}",
                    ha="left", va="center", fontsize=8, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "typology_detection.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved typology detection to {FIGURES_DIR / 'typology_detection.png'}")

    logger.info("✓ All report figures generated successfully")


if __name__ == "__main__":
    main()
