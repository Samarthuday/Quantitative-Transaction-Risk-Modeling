"""Load and cache analysis results (ablation, typology, figures)."""

import base64
import csv
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_ablation_results() -> list[dict[str, Any]]:
    """Load feature ablation study results from CSV."""
    ablation_path = PROJECT_ROOT / "reports/ablation_results.csv"
    if not ablation_path.exists():
        return []

    results = []
    with ablation_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "model": row["model"],
                "pr_auc": float(row["pr_auc"]),
                "recall_at_0.5%": float(row["recall_at_0.5%"]),
            })
    return results


def load_typology_results() -> list[dict[str, Any]]:
    """Load behavioral typology analysis results from CSV."""
    typology_path = PROJECT_ROOT / "reports/typology_results.csv"
    if not typology_path.exists():
        return []

    results = []
    with typology_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "typology": row["typology"],
                "transactions": int(row["transactions"]),
                "positives": int(row["positives"]),
                "recall_at_0.5%": float(row["recall_at_0.5%"]),
            })
    return results


def encode_all_figures() -> dict[str, str]:
    """Load all report figures from reports/figures/*.png and encode to base64."""
    figures_dir = PROJECT_ROOT / "reports/figures"
    if not figures_dir.exists():
        return {}

    figures = {}
    figure_mapping = {
        "calibration_curve": "Calibration Curve",
        "roc_curve": "ROC Curve",
        "precision_recall_curve": "Precision-Recall Curve",
        "feature_importance": "Feature Importance",
        "ablation_comparison": "Ablation Comparison",
        "typology_detection": "Typology Detection",
    }

    for png_path in sorted(figures_dir.glob("*.png")):
        stem = png_path.stem
        with png_path.open("rb") as f:
            image_bytes = f.read()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        label = figure_mapping.get(stem, stem.replace("_", " ").title())
        figures[stem] = {
            "base64": base64_str,
            "label": label,
        }

    return figures


# Cache results at module level
_ablation_cache: Optional[list] = None
_typology_cache: Optional[list] = None
_figures_cache: Optional[dict] = None


def get_ablation_results() -> list[dict[str, Any]]:
    """Get cached ablation results."""
    global _ablation_cache
    if _ablation_cache is None:
        _ablation_cache = load_ablation_results()
    return _ablation_cache


def get_typology_results() -> list[dict[str, Any]]:
    """Get cached typology results."""
    global _typology_cache
    if _typology_cache is None:
        _typology_cache = load_typology_results()
    return _typology_cache


def get_all_figures() -> dict[str, dict[str, str]]:
    """Get cached all report figures as base64 with labels."""
    global _figures_cache
    if _figures_cache is None:
        _figures_cache = encode_all_figures()
    return _figures_cache
