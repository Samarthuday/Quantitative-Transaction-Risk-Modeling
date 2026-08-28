import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw/SAML-D.csv")
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data/processed/transactions_features.parquet")
    parser.add_argument("--artifact", type=Path, default=PROJECT_ROOT / "artifacts/risk_model.joblib")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    python = sys.executable

    run([python, "scripts/build_features.py", "--input", str(args.input), "--output", str(args.features)])
    train = [python, "scripts/train_model.py", "--features", str(args.features), "--artifact", str(args.artifact)]
    ablation = [python, "scripts/feature_ablation.py", "--features", str(args.features)]
    walk_forward = [python, "scripts/walk_forward_backtest.py", "--features", str(args.features)]
    if args.fast:
        train.append("--fast")
        ablation.append("--fast")
        walk_forward.append("--fast")
    run(train)
    run(ablation)
    run(walk_forward)


if __name__ == "__main__":
    main()
