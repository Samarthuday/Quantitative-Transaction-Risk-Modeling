import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
):
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)

    train_end = int(n * train_fraction)
    validation_end = int(
        n * (train_fraction + validation_fraction)
    )

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    return train, validation, test

TARGET = "Is_laundering"


NUMERIC_FEATURES = [
    "Amount",
    "log_amount",

    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",

    "is_weekend",
    "is_night",

    "currency_mismatch",
    "cross_border",
    "is_round_amount",

    "sender_txn_count_24h",
    "sender_amount_sum_24h",
    "sender_amount_mean_30d",
    "sender_amount_std_30d",
    "sender_amount_zscore",

    "receiver_txn_count_24h",
    "receiver_amount_sum_24h",

    "seconds_since_sender_txn",

    "sender_txn_count_lifetime",
    "receiver_txn_count_lifetime",
    "sender_out_degree",
    "receiver_in_degree",
    "pair_transaction_count",
    "sender_counterparty_hhi",
]


CATEGORICAL_FEATURES = [
    "Payment_type",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
]


MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

BASE_FEATURES = [
    "Amount",
    "log_amount",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_night",
    "currency_mismatch",
    "cross_border",
    "is_round_amount",
]

BEHAVIORAL_FEATURES = [
    "sender_txn_count_24h",
    "sender_amount_sum_24h",
    "sender_amount_mean_30d",
    "sender_amount_std_30d",
    "sender_amount_zscore",
    "receiver_txn_count_24h",
    "receiver_amount_sum_24h",
    "seconds_since_sender_txn",
]

NETWORK_FEATURES = [
    "sender_txn_count_lifetime",
    "receiver_txn_count_lifetime",
    "sender_out_degree",
    "receiver_in_degree",
    "pair_transaction_count",
    "sender_counterparty_hhi",
]

ABLATION_FEATURE_SETS = {
    "Base": BASE_FEATURES + CATEGORICAL_FEATURES,
    "+ Behavioral": BASE_FEATURES + BEHAVIORAL_FEATURES + CATEGORICAL_FEATURES,
    "+ Network": BASE_FEATURES + NETWORK_FEATURES + CATEGORICAL_FEATURES,
    "All": MODEL_FEATURES,
}


def build_preprocessor(feature_names=MODEL_FEATURES):
    numeric_features = [
        feature for feature in feature_names
        if feature in NUMERIC_FEATURES
    ]
    categorical_features = [
        feature for feature in feature_names
        if feature in CATEGORICAL_FEATURES
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_xgboost_model(y_train):
    positives = int(y_train.sum())
    negatives = len(y_train) - positives

    scale_pos_weight = negatives / max(positives, 1)

    print(
        "scale_pos_weight:",
        round(scale_pos_weight, 2),
    )

    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def fit_model(train, validation, feature_names=MODEL_FEATURES):
    X_train = train[feature_names]
    y_train = train[TARGET]

    X_validation = validation[feature_names]
    y_validation = validation[TARGET]

    preprocessor = build_preprocessor(feature_names)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)

    model = build_xgboost_model(y_train)
    model.fit(
        X_train_processed,
        y_train,
    )

    return (
        preprocessor,
        model,
        X_validation_processed,
        y_validation,
    )