import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import TrainingConfig, get_config


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


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.65,
    calibration_fraction: float = 0.10,
    validation_fraction: float = 0.10,
):
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    if train_fraction <= 0 or calibration_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive.")
    if train_fraction + calibration_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave observations for test data.")

    timestamps = ordered["timestamp"].drop_duplicates().sort_values().to_numpy()
    timestamp_count = len(timestamps)
    train_end = max(1, int(timestamp_count * train_fraction))
    calibration_end = max(train_end + 1, int(timestamp_count * (train_fraction + calibration_fraction)))
    validation_end = max(calibration_end + 1, int(timestamp_count * (train_fraction + calibration_fraction + validation_fraction)))
    if validation_end >= timestamp_count:
        raise ValueError("not enough distinct timestamps for four temporal periods.")

    boundaries = timestamps[[train_end, calibration_end, validation_end]]
    train = ordered[ordered["timestamp"] < boundaries[0]].copy()
    calibration = ordered[
        (ordered["timestamp"] >= boundaries[0])
        & (ordered["timestamp"] < boundaries[1])
    ].copy()
    validation = ordered[
        (ordered["timestamp"] >= boundaries[1])
        & (ordered["timestamp"] < boundaries[2])
    ].copy()
    test = ordered[ordered["timestamp"] >= boundaries[2]].copy()

    return train, calibration, validation, test

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


def build_preprocessor(feature_names=MODEL_FEATURES, config: TrainingConfig = None):
    if config is None:
        config = get_config()

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
                SimpleImputer(strategy=config.preprocessing.numeric_impute_strategy),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=config.preprocessing.categorical_impute_strategy),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown=config.preprocessing.categorical_handle_unknown,
                    min_frequency=config.preprocessing.categorical_min_frequency,
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


def build_xgboost_model(y_train, config: TrainingConfig = None):
    if config is None:
        config = get_config()

    positives = int(y_train.sum())
    negatives = len(y_train) - positives

    scale_pos_weight = negatives / max(positives, 1)

    model_params = config.xgboost.to_dict()
    model_params.update({
        "objective": "binary:logistic",
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
    })

    return xgb.XGBClassifier(**model_params)


def fit_model(train, validation, feature_names=MODEL_FEATURES, config: TrainingConfig = None):
    if config is None:
        config = get_config()

    X_train = train[feature_names]
    y_train = train[TARGET]

    X_validation = validation[feature_names]
    y_validation = validation[TARGET]

    preprocessor = build_preprocessor(feature_names, config=config)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)

    model = build_xgboost_model(y_train, config=config)
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