import duckdb
import pandas as pd


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate historical behavioral variables.

    IMPORTANT:
    All rolling windows end immediately BEFORE the current transaction.
    This prevents future information from leaking into the prediction.
    """

    con = duckdb.connect()

    con.register("transactions", df)

    query = """
    WITH history AS (
        SELECT
            *,

            COUNT(*) OVER (
                PARTITION BY Sender_account, Payment_currency
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ) AS sender_txn_count_24h,

            COALESCE(
                SUM(Amount) OVER (
                    PARTITION BY Sender_account, Payment_currency
                    ORDER BY timestamp
                    RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                    AND INTERVAL '1 microsecond' PRECEDING
                ),
                0
            ) AS sender_amount_sum_24h,

            AVG(Amount) OVER (
                PARTITION BY Sender_account, Payment_currency
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ) AS sender_amount_mean_30d,

            STDDEV_SAMP(Amount) OVER (
                PARTITION BY Sender_account, Payment_currency
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '30 days' PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ) AS sender_amount_std_30d,

            COUNT(*) OVER (
                PARTITION BY Receiver_account, Received_currency
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ) AS receiver_txn_count_24h,

            COALESCE(
                SUM(Amount) OVER (
                    PARTITION BY Receiver_account, Received_currency
                    ORDER BY timestamp
                    RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                    AND INTERVAL '1 microsecond' PRECEDING
                ),
                0
            ) AS receiver_amount_sum_24h,

            LAG(timestamp) OVER (
                PARTITION BY Sender_account
                ORDER BY timestamp
            ) AS previous_sender_timestamp,

            COUNT(*) OVER (
                PARTITION BY Sender_account
                ORDER BY timestamp
                RANGE BETWEEN UNBOUNDED PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ) AS sender_txn_count_lifetime,

            COUNT(*) OVER (
                PARTITION BY Receiver_account
                ORDER BY timestamp
                RANGE BETWEEN UNBOUNDED PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ) AS receiver_txn_count_lifetime,

            COUNT(DISTINCT Receiver_account) OVER (
                PARTITION BY Sender_account
                ORDER BY timestamp
                RANGE BETWEEN UNBOUNDED PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ) AS sender_out_degree,

            COUNT(DISTINCT Sender_account) OVER (
                PARTITION BY Receiver_account
                ORDER BY timestamp
                RANGE BETWEEN UNBOUNDED PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ) AS receiver_in_degree,

            COUNT(*) OVER (
                PARTITION BY Sender_account, Receiver_account
                ORDER BY timestamp
                RANGE BETWEEN UNBOUNDED PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ) AS pair_transaction_count

        FROM transactions
    )

    SELECT
        *,

        CASE
            WHEN sender_amount_std_30d IS NULL
                 OR sender_amount_std_30d = 0
            THEN 0
            ELSE
                (Amount - sender_amount_mean_30d)
                / sender_amount_std_30d
        END AS sender_amount_zscore,

        COALESCE(
            DATE_DIFF(
                'second',
                previous_sender_timestamp,
                timestamp
            ),
            -1
        ) AS seconds_since_sender_txn

    FROM history
    ORDER BY timestamp
    """

    result = con.execute(query).fetchdf()

    con.close()

    numerical_history_columns = [
        "sender_txn_count_24h",
        "sender_amount_sum_24h",
        "sender_amount_mean_30d",
        "sender_amount_std_30d",
        "receiver_txn_count_24h",
        "receiver_amount_sum_24h",
        "sender_txn_count_lifetime",
        "receiver_txn_count_lifetime",
        "sender_out_degree",
        "receiver_in_degree",
        "pair_transaction_count",
        "sender_amount_zscore",
        "seconds_since_sender_txn",
    ]

    result[numerical_history_columns] = (
        result[numerical_history_columns]
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )

    sender_totals = {}
    sender_squared_totals = {}
    sender_receiver_totals = {}
    concentration = []

    for _, timestamp_group in result.groupby("timestamp", sort=False):
        for row in timestamp_group.itertuples(index=False):
            sender = row.Sender_account
            receiver = row.Receiver_account
            currency = row.Payment_currency
            sender_key = (sender, currency)
            total = sender_totals.get(sender_key, 0.0)
            concentration.append(
                sender_squared_totals.get(sender_key, 0.0) / total**2
                if total > 0
                else 0.0
            )

        for row in timestamp_group.itertuples(index=False):
            sender = row.Sender_account
            receiver = row.Receiver_account
            currency = row.Payment_currency
            amount = float(row.Amount)
            sender_key = (sender, currency)
            pair_key = (sender, receiver, currency)
            pair_total = sender_receiver_totals.get(pair_key, 0.0)
            sender_receiver_totals[pair_key] = pair_total + amount
            sender_totals[sender_key] = sender_totals.get(sender_key, 0.0) + amount
            sender_squared_totals[sender_key] = (
                sender_squared_totals.get(sender_key, 0.0)
                + 2 * pair_total * amount
                + amount**2
            )

    result["sender_counterparty_hhi"] = concentration

    return result