"""Load and cache analysis results (ablation, typology, figures)."""

import base64
import csv
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "docs/assets"


def _resolve_path(relative_name: str) -> Optional[Path]:
    """Prefer a freshly generated report, fall back to the committed published copy."""
    local_path = REPORTS_DIR / relative_name
    if local_path.exists():
        return local_path

    fallback_path = ASSETS_DIR / relative_name
    if fallback_path.exists():
        return fallback_path

    return None


def load_ablation_results() -> list[dict[str, Any]]:
    """Load feature ablation study results from CSV."""
    ablation_path = _resolve_path("ablation_results.csv")
    if ablation_path is None:
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
    typology_path = _resolve_path("typology_results.csv")
    if typology_path is None:
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
    """Load report figures, preferring freshly generated ones over the committed fallback."""
    figure_mapping = {
        "calibration_curve": "Calibration Curve",
        "roc_curve": "ROC Curve",
        "precision_recall_curve": "Precision-Recall Curve",
        "feature_importance": "Feature Importance",
        "ablation_comparison": "Ablation Comparison",
        "typology_detection": "Typology Detection",
    }

    # Fallback pass first so freshly generated figures (below) take precedence.
    png_paths: dict[str, Path] = {}
    for directory in (ASSETS_DIR, REPORTS_DIR / "figures"):
        if not directory.exists():
            continue
        for png_path in directory.glob("*.png"):
            png_paths[png_path.stem] = png_path

    figures = {}
    for stem, png_path in sorted(png_paths.items()):
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
