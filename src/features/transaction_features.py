import numpy as np
import pandas as pd


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived only from the current transaction.
    No historical information is used here.
    """

    df = df.copy()

    ts = df["timestamp"]

    hour = ts.dt.hour
    day_of_week = ts.dt.dayofweek
    month = ts.dt.month

    # Heavy-tailed transaction amounts
    df["log_amount"] = np.log1p(df["Amount"])

    # Cyclical temporal encoding
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Time risk indicators
    df["is_weekend"] = (day_of_week >= 5).astype("int8")
    df["is_night"] = ((hour >= 22) | (hour <= 6)).astype("int8")

    # Transaction structure
    df["currency_mismatch"] = (
        df["Payment_currency"].astype(str)
        != df["Received_currency"].astype(str)
    ).astype("int8")

    df["cross_border"] = (
        df["Sender_bank_location"].astype(str)
        != df["Receiver_bank_location"].astype(str)
    ).astype("int8")

    # Round-value behavior
    df["is_round_amount"] = np.isclose(
        df["Amount"] % 1000,
        0,
    ).astype("int8")

    return df