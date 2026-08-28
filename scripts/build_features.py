import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_saml_data
from src.features.behavioral_features import add_behavioral_features
from src.features.transaction_features import add_transaction_features
from src.logging_config import setup_logging

logger = setup_logging(__name__)

RAW_PATH = PROJECT_ROOT / "data/raw/SAML-D.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/transactions_features.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    logger.info(f"Loading SAML-D from {args.input}...")
    df = load_saml_data(args.input)
    logger.info(f"Loaded {len(df):,} transactions")

    logger.info("Building transaction features...")
    df = add_transaction_features(df)
    logger.debug(f"Transaction features: {[c for c in df.columns if c.startswith(('hour_', 'dow_', 'month_', 'is_', 'currency_', 'cross_'))]}")

    logger.info("Building behavioral and network features...")
    df = add_behavioral_features(df)
    logger.debug(f"Final dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing feature dataset to {args.output}...")
    df.to_parquet(
        args.output,
        index=False,
        compression="snappy",
    )

    logger.info(f"Feature engineering completed. Output: {args.output}")
    logger.info(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()