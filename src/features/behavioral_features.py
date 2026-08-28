import duckdb
import pandas as pd


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate point-in-time behavioral and network features."""
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
            COALESCE(SUM(Amount) OVER (
                PARTITION BY Sender_account, Payment_currency
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ), 0) AS sender_amount_sum_24h,
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
            COALESCE(SUM(Amount) OVER (
                PARTITION BY Receiver_account, Received_currency
                ORDER BY timestamp
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ), 0) AS receiver_amount_sum_24h,
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
    ),
    pair_history AS (
        SELECT
            Sender_account,
            Payment_currency,
            Receiver_account,
            timestamp,
            SUM(Amount) AS pair_amount
        FROM transactions
        GROUP BY Sender_account, Payment_currency, Receiver_account, timestamp
    ),
    pair_cumulative AS (
        SELECT
            *,
            COALESCE(SUM(pair_amount) OVER (
                PARTITION BY Sender_account, Payment_currency, Receiver_account
                ORDER BY timestamp
                RANGE BETWEEN UNBOUNDED PRECEDING
                AND INTERVAL '1 microsecond' PRECEDING
            ), 0) AS historical_pair_amount
        FROM pair_history
    ),
    sender_hhi AS (
        SELECT
            Sender_account,
            Payment_currency,
            timestamp,
            CASE
                WHEN SUM(historical_pair_amount) = 0 THEN 0
                ELSE SUM(POWER(historical_pair_amount, 2))
                    / POWER(SUM(historical_pair_amount), 2)
            END AS sender_counterparty_hhi
        FROM pair_cumulative
        GROUP BY Sender_account, Payment_currency, timestamp
    )
    SELECT
        history.*,
        CASE
            WHEN sender_amount_std_30d IS NULL
                 OR sender_amount_std_30d = 0
            THEN 0
            ELSE (Amount - sender_amount_mean_30d) / sender_amount_std_30d
        END AS sender_amount_zscore,
        COALESCE(
            DATE_DIFF('second', previous_sender_timestamp, history.timestamp),
            -1
        ) AS seconds_since_sender_txn,
        COALESCE(sender_hhi.sender_counterparty_hhi, 0)
            AS sender_counterparty_hhi
    FROM history
    LEFT JOIN sender_hhi
        ON history.Sender_account = sender_hhi.Sender_account
        AND history.Payment_currency = sender_hhi.Payment_currency
        AND history.timestamp = sender_hhi.timestamp
    ORDER BY history.timestamp
    """

    result = con.execute(query).fetchdf()
    con.close()

    numeric_columns = [
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
        "sender_counterparty_hhi",
    ]
    result[numeric_columns] = (
        result[numeric_columns]
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )
    return result
