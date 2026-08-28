from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "Time",
    "Date",
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Is_laundering",
    "Laundering_type",
]


def load_saml_data(
    path: str | Path = "data/raw/SAML-D.csv",
) -> pd.DataFrame:
    """
    Load the SAML-D transaction dataset and construct a chronological timestamp.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"SAML-D dataset not found at {path}. "
            "Place the CSV inside data/raw/."
        )

    # A Git LFS pointer is text metadata, not the dataset itself. Detecting it
    # here gives contributors an actionable error instead of a misleading
    # missing-column failure from pandas.
    if path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise FileNotFoundError(
            f"{path} is a Git LFS pointer, not the SAML-D CSV. "
            "Download the licensed dataset and place the actual CSV in data/raw/."
        )

    df = pd.read_csv(
        path,
        dtype={
            "Sender_account": "string",
            "Receiver_account": "string",
            "Amount": "float64",
            "Payment_currency": "category",
            "Received_currency": "category",
            "Sender_bank_location": "category",
            "Receiver_bank_location": "category",
            "Payment_type": "category",
            "Is_laundering": "int8",
            "Laundering_type": "category",
        },
    )

    missing = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp", "Amount"])

    # Chronology is critical for financial backtesting.
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df
