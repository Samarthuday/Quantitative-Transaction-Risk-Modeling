import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    print("Loading SAML-D...")
    df = load_saml_data(args.input)

    print(f"Loaded {len(df):,} transactions")

    print("Building transaction features...")
    df = add_transaction_features(df)

    print("Building behavioral features...")
    df = add_behavioral_features(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        args.output,
        index=False,
        compression="snappy",
    )

    print(f"Feature dataset written to {args.output}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()