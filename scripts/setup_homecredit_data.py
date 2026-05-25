#!/usr/bin/env python3
"""Download, validate, and prepare the Home Credit Default Risk dataset.

Dataset
-------
Home Credit Default Risk (Kaggle competition, public)
URL: https://www.kaggle.com/c/home-credit-default-risk

Main file used: application_train.csv  (~307 k rows, 122 columns)
Target column : TARGET  (1 = defaulted within contract period, 0 = repaid)

Prerequisites
-------------
1. Install the Kaggle CLI:
       pip install kaggle

2. Create a Kaggle API token:
       https://www.kaggle.com/settings  →  "Create New Token"
       This downloads  ~/.kaggle/kaggle.json  (keep it private).

3. Accept the competition rules at:
       https://www.kaggle.com/c/home-credit-default-risk/rules

Usage
-----
# Download and validate (will prompt to accept rules if needed):
  python setup_homecredit_data.py

# Skip download if the file already exists:
  python setup_homecredit_data.py --skip-download

# Change default output directory:
  python setup_homecredit_data.py --data-dir data/homecredit

# Limit rows when testing downstream scripts:
  python setup_homecredit_data.py --sample-rows 10000 --sample-output outputs/homecredit_sample.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

COMPETITION = "home-credit-default-risk"
MAIN_FILE = "application_train.csv"

EXPECTED_COLUMNS = {
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
}

MIN_ROWS = 100_000


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _kaggle_available() -> bool:
    try:
        import kaggle  # noqa: F401
        return True
    except ImportError:
        return False


def download_competition_files(data_dir: Path) -> None:
    """Download the competition zip via the Kaggle API and unzip into data_dir."""
    if not _kaggle_available():
        print(
            "ERROR: The 'kaggle' package is not installed.\n"
            "       Run:  pip install kaggle"
        )
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Locate the kaggle CLI binary next to the current Python executable.
    kaggle_bin = Path(sys.executable).parent / "kaggle"
    if not kaggle_bin.exists():
        kaggle_bin = Path("kaggle")  # fall back to PATH

    print(f"Downloading competition '{COMPETITION}' to {data_dir} …")
    result = subprocess.run(
        [
            str(kaggle_bin),
            "competitions", "download",
            "-c", COMPETITION,
            "-p", str(data_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("DOWNLOAD FAILED\n" + result.stderr)
        sys.exit(1)

    zip_candidates = list(data_dir.glob("*.zip"))
    if not zip_candidates:
        print("No zip file found after download. Check the output above.")
        sys.exit(1)

    zip_path = zip_candidates[0]
    print(f"Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print("Extraction complete.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_main_file(csv_path: Path) -> pd.DataFrame:
    """Load and validate application_train.csv; return the DataFrame."""
    if not csv_path.exists():
        print(f"ERROR: Expected file not found: {csv_path}")
        sys.exit(1)

    print(f"Loading {csv_path.name} …")
    df = pd.read_csv(csv_path, low_memory=False)

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        print(f"WARNING: Expected columns missing: {sorted(missing_cols)}")

    if len(df) < MIN_ROWS:
        print(
            f"WARNING: Only {len(df):,} rows found (expected ≥ {MIN_ROWS:,}). "
            "The file may be incomplete."
        )

    target_counts = df["TARGET"].value_counts().sort_index()
    default_rate = float(df["TARGET"].mean())

    print(f"\nFile summary ({csv_path.name}):")
    print(f"  Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage     : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  Target counts    : {target_counts.to_dict()}")
    print(f"  Default rate     : {default_rate:.2%}")
    print(f"  Missing values   : {df.isnull().sum().sum():,} cells "
          f"({df.isnull().mean().mean():.1%} overall)")
    print(f"  Numeric columns  : {df.select_dtypes('number').shape[1]}")
    print(f"  String columns   : {df.select_dtypes('object').shape[1]}")

    return df


# ---------------------------------------------------------------------------
# Optional sample export
# ---------------------------------------------------------------------------

def export_sample(df: pd.DataFrame, output_path: Path, n_rows: int, seed: int) -> None:
    """Write a stratified random sample to CSV for fast downstream testing."""
    from sklearn.model_selection import train_test_split

    if n_rows >= len(df):
        sample = df.copy()
    else:
        sample, _ = train_test_split(
            df, train_size=n_rows, stratify=df["TARGET"], random_state=seed
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.reset_index(drop=True).to_csv(output_path, index=False)
    print(
        f"\nSample export: {len(sample):,} rows → {output_path}  "
        f"(default rate: {sample['TARGET'].mean():.2%})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and validate the Home Credit Default Risk dataset."
    )
    parser.add_argument(
        "--data-dir",
        default="data/homecredit",
        help="Directory for raw competition files. Default: data/homecredit",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Kaggle download; only validate existing files.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="If > 0, export a stratified sample of this size to --sample-output.",
    )
    parser.add_argument(
        "--sample-output",
        default="outputs/homecredit_sample.csv",
        help="Path for the sample CSV. Default: outputs/homecredit_sample.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data_dir = Path(args.data_dir)
    csv_path = data_dir / MAIN_FILE

    if not args.skip_download:
        if csv_path.exists():
            print(f"{MAIN_FILE} already present at {csv_path}. Use --skip-download to skip validation.")
        else:
            download_competition_files(data_dir)
    else:
        print(f"--skip-download: skipping Kaggle download, validating {csv_path}")

    df = validate_main_file(csv_path)

    if args.sample_rows > 0:
        export_sample(df, Path(args.sample_output), args.sample_rows, args.seed)

    print("\nSetup complete.")


if __name__ == "__main__":
    main()
