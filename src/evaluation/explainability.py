"""Model explainability using SHAP values."""

from typing import Union

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def create_shap_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """
    Create a SHAP TreeExplainer for an XGBoost model.

    Args:
        model: Fitted XGBoost classifier

    Returns:
        SHAP TreeExplainer instance
    """
    return shap.TreeExplainer(model)


def calculate_shap_values(
    model: xgb.XGBClassifier,
    X: Union[np.ndarray, pd.DataFrame],
) -> np.ndarray:
    """
    Calculate SHAP values for model predictions.

    Args:
        model: Fitted XGBoost classifier
        X: Input features

    Returns:
        SHAP values array
    """
    explainer = create_shap_explainer(model)
    return explainer.shap_values(X)


def feature_importance_summary(
    model: xgb.XGBClassifier,
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: list = None,
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP values (global feature importance).

    Args:
        model: Fitted XGBoost classifier
        X: Input features for explanation
        feature_names: List of feature names

    Returns:
        DataFrame with features and importance scores, sorted descending
    """
    shap_values = calculate_shap_values(model, X)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]

    return pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap,
    }).sort_values("importance", ascending=False)


def explain_prediction_instance(
    model: xgb.XGBClassifier,
    X: Union[np.ndarray, pd.DataFrame],
    instance_idx: int,
    feature_names: list = None,
) -> pd.DataFrame:
    """
    Explain a single prediction using SHAP values.

    Args:
        model: Fitted XGBoost classifier
        X: Input features
        instance_idx: Index of instance to explain
        feature_names: List of feature names

    Returns:
        DataFrame with feature contributions to prediction
    """
    explainer = create_shap_explainer(model)
    shap_values = explainer.shap_values(X)

    instance_shap = shap_values[instance_idx]

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(instance_shap))]

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": instance_shap,
        "abs_shap_value": np.abs(instance_shap),
    }).sort_values("abs_shap_value", ascending=False)

    return contributions