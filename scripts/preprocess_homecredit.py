#!/usr/bin/env python3
"""Preprocess application_train.csv from the Home Credit Default Risk dataset.

Preprocessing steps (mirrors the synthetic-data pipeline):
  1. Drop identifier and leakage-prone columns.
  2. Derive human-readable age/employment features from DAYS_* columns.
  3. Impute numeric columns with median; categorical with mode.
  4. One-hot encode categorical columns (drop='first').
  5. Apply a time-ordered surrogate split using SK_ID_CURR as a proxy
     for chronological order (ascending ID ≈ older applications).
  6. Save the clean feature matrix + target to outputs/homecredit_preprocessed.csv.

The output CSV is then consumed by run_4model_baseline.py (adapted) or directly
by model scripts that accept the same feature/target format.

Usage
-----
  python preprocess_homecredit.py --input data/homecredit/application_train.csv

  # Use a pre-drawn sample (fast iteration):
  python preprocess_homecredit.py --input outputs/homecredit_sample.csv

  # Full run with custom output path:
  python preprocess_homecredit.py \\
      --input data/homecredit/application_train.csv \\
      --output outputs/homecredit_preprocessed.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# Columns dropped because they are identifiers, post-outcome, or extreme leakage
DROP_COLUMNS: List[str] = [
    "SK_ID_CURR",
]

# DAYS_* columns encode time relative to application date (negative = days before)
# We convert them to positive magnitudes with meaningful names.
DAYS_COLUMNS_MAP = {
    "DAYS_BIRTH": "age_years",
    "DAYS_EMPLOYED": "employment_years",
    "DAYS_REGISTRATION": "registration_years",
    "DAYS_ID_PUBLISH": "id_publish_years",
    "DAYS_LAST_PHONE_CHANGE": "last_phone_change_years",
}

# Binary flag columns that use 'Y'/'N' encoding → convert to 1/0
BINARY_YN_COLUMNS: List[str] = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]

# Threshold above which DAYS_EMPLOYED values are treated as missing
# (365243 is a sentinel used by Home Credit for unemployed/retired applicants)
DAYS_EMPLOYED_SENTINEL = 365243


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _convert_days_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DAYS_* to positive-magnitude year features."""
    df = df.copy()
    for col, new_name in DAYS_COLUMNS_MAP.items():
        if col not in df.columns:
            continue
        values = df[col].copy().astype(float)
        if col == "DAYS_EMPLOYED":
            values = values.replace(DAYS_EMPLOYED_SENTINEL, np.nan)
        df[new_name] = (-values / 365.25).clip(lower=0)
        df.drop(columns=[col], inplace=True)
    return df


def _encode_binary_yn(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'Y'/'N' binary columns to 1/0 integers."""
    df = df.copy()
    for col in BINARY_YN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"Y": 1, "N": 0})
    return df


def _drop_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)


# ---------------------------------------------------------------------------
# Core preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (feature_frame, target_series) ready for modelling."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    y = df[TARGET_COLUMN].astype(int).copy()
    df = _drop_columns(df, DROP_COLUMNS)
    df = df.drop(columns=[TARGET_COLUMN])

    df = _convert_days_columns(df)
    df = _encode_binary_yn(df)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric imputation
    if num_cols:
        imputer = SimpleImputer(strategy="median")
        df[num_cols] = imputer.fit_transform(df[num_cols])

    # Categorical imputation + OHE
    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        ohe_arr = ohe.fit_transform(df[cat_cols])
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_names, index=df.index)

        df = df.drop(columns=cat_cols)
        df = pd.concat([df, ohe_df], axis=1)

    return df.reset_index(drop=True), y.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time-based split proxy
# ---------------------------------------------------------------------------

def add_surrogate_time_column(df: pd.DataFrame, id_col: pd.Series) -> pd.DataFrame:
    """Add a surrogate 'sort_key' so time-split logic can sort by SK_ID_CURR."""
    df = df.copy()
    df["sort_key"] = id_col.values
    return df


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(input_path: Path, output_path: Path, verbose: bool = True) -> None:
    if verbose:
        print(f"Loading {input_path} …")
    raw = pd.read_csv(input_path, low_memory=False)

    if verbose:
        print(f"  Raw shape: {raw.shape[0]:,} rows × {raw.shape[1]} columns")
        print(f"  Default rate: {raw[TARGET_COLUMN].mean():.2%}")

    id_series = raw[ID_COLUMN].copy() if ID_COLUMN in raw.columns else None

    X, y = preprocess(raw)

    if id_series is not None:
        X = add_surrogate_time_column(X, id_series)

    out_df = X.copy()
    out_df[TARGET_COLUMN] = y.values

    if verbose:
        print(f"\nPreprocessed shape : {out_df.shape[0]:,} rows × {out_df.shape[1]} columns")
        print(f"  Numeric features : {X.select_dtypes(include=[np.number]).shape[1] - (1 if 'sort_key' in X.columns else 0)}")
        print(f"  OHE features     : {len([c for c in X.columns if '_' in c and any(c.startswith(cat) for cat in ['NAME_', 'CODE_', 'FLAG_', 'WEEKDAY_', 'ORGANIZATION_', 'FONDKAPREMONT_', 'HOUSETYPE_', 'WALLSMATERIAL_', 'EMERGENCYSTATE_'])])}")
        print(f"  Missing cells    : {out_df.isnull().sum().sum()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved preprocessed data → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess Home Credit application_train.csv for the 4-model pipeline."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to application_train.csv (or a sample CSV).",
    )
    parser.add_argument(
        "--output",
        default="outputs/homecredit_preprocessed.csv",
        help="Output path. Default: outputs/homecredit_preprocessed.csv",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(Path(args.input), Path(args.output), verbose=not args.quiet)


if __name__ == "__main__":
    main()
