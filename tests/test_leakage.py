import pandas as pd

from src.evaluation.metrics import top_k_alert_mask
from src.features.behavioral_features import add_behavioral_features
from src.models.train import MODEL_FEATURES, chronological_split, temporal_split

FORBIDDEN_FEATURES = {
    "Is_laundering",
    "Laundering_type",
    "Sender_account",
    "Receiver_account",
}


def test_no_identifier_or_target_leakage_features():
    assert not (set(MODEL_FEATURES) & FORBIDDEN_FEATURES)


def test_behavioral_features_exclude_current_timestamp():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 12:00:00",
                    "2026-01-01 12:00:00",
                    "2026-01-01 12:01:00",
                ]
            ),
            "Sender_account": ["sender", "sender", "sender"],
            "Receiver_account": ["receiver-a", "receiver-b", "receiver-a"],
            "Amount": [100.0, 200.0, 300.0],
            "Payment_currency": ["USD", "USD", "USD"],
            "Received_currency": ["USD", "USD", "USD"],
        }
    )

    result = add_behavioral_features(frame)

    assert result.loc[0, "sender_txn_count_24h"] == 0
    assert result.loc[1, "sender_txn_count_24h"] == 0
    assert result.loc[2, "sender_txn_count_24h"] == 2
    assert result.loc[2, "pair_transaction_count"] == 1
    assert result.loc[2, "sender_out_degree"] == 2


def test_chronological_split_boundaries():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10),
            "value": range(10),
        }
    )

    train, validation, test = chronological_split(
        frame,
        train_fraction=0.6,
        validation_fraction=0.2,
    )

    assert train["value"].tolist() == list(range(6))
    assert validation["value"].tolist() == [6, 7]
    assert test["value"].tolist() == [8, 9]
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()


def test_amount_history_is_currency_aware():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 12:00:00", "2026-01-01 12:01:00"]
            ),
            "Sender_account": ["sender", "sender"],
            "Receiver_account": ["receiver-a", "receiver-b"],
            "Amount": [100.0, 200.0],
            "Payment_currency": ["USD", "EUR"],
            "Received_currency": ["USD", "EUR"],
        }
    )

    result = add_behavioral_features(frame)

    assert result.loc[1, "sender_amount_sum_24h"] == 0
    assert result.loc[1, "sender_amount_mean_30d"] == 0
    assert result.loc[1, "sender_counterparty_hhi"] == 0


def test_alert_budget_selects_exact_top_k_on_ties():
    mask = top_k_alert_mask([0.9, 0.5, 0.5, 0.1], alert_rate=0.5)

    assert mask.tolist() == [True, True, False, False]


def test_temporal_split_keeps_timestamp_groups_together():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                ]
            ),
            "value": range(10),
        }
    )

    train, calibration, validation, test = temporal_split(frame)

    periods = [train, calibration, validation, test]
    assert all(not period.empty for period in periods)
    assert all(
        left["timestamp"].max() < right["timestamp"].min()
        for left, right in zip(periods, periods[1:])
    )
