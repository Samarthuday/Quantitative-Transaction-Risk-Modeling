import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_saml_data
from src.features.behavioral_features import add_behavioral_features
from src.features.transaction_features import add_transaction_features

RAW_PATH = PROJECT_ROOT / "data/raw/SAML-D.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/transactions_features.parquet"


def main():
    print("Loading SAML-D...")
    df = load_saml_data(RAW_PATH)

    print(f"Loaded {len(df):,} transactions")

    print("Building transaction features...")
    df = add_transaction_features(df)

    print("Building behavioral features...")
    df = add_behavioral_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="snappy",
    )

    print(f"Feature dataset written to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()