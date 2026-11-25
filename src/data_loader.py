"""
src/data_loader.py
------------------
Step 1 of the ML pipeline: load the raw dataset, inspect it thoroughly,
and report any data quality issues.

Run this file standalone to see a full data report:
    python src/data_loader.py

NOTE: This project uses the "Individual Carbon Footprint Calculation" dataset
by Mesut Duman (Kaggle). That dataset is SYNTHETICALLY GENERATED using
weights from published carbon calculators to approximate real-world values.
It is NOT collected from real people. License: CC0 Public Domain.
"""

import os
import sys
import pandas as pd
import numpy as np

# Make sure we can import config from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DATA_PATH


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV file from disk.

    Parameters
    ----------
    filepath : str
        Path to 'Carbon Emission.csv'

    Returns
    -------
    pd.DataFrame
        Raw dataframe exactly as it comes from the CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n\n❌ Dataset not found at: {filepath}\n"
            "Please download 'Carbon Emission.csv' from:\n"
            "https://www.kaggle.com/datasets/dumanmesut/individual-carbon-footprint-calculation\n"
            "and place it in: data/raw/\n"
        )

    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a thorough data quality report: shape, dtypes, missing values,
    duplicates, and target distribution.
    """
    print("=" * 70)
    print("DATA INSPECTION REPORT")
    print("=" * 70)

    # --- Shape ---
    print(f"\n📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # --- Columns and dtypes ---
    print("\n📋 Columns and dtypes:")
    for col in df.columns:
        print(f"   {col:<45} {str(df[col].dtype):<12}")

    # --- Missing values ---
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print("\n🔍 Missing values:")
    if missing.sum() == 0:
        print("   ✅ No missing values found!")
    else:
        for col in missing[missing > 0].index:
            print(f"   {col:<45} {missing[col]:>6}  ({missing_pct[col]}%)")

    # --- Duplicates ---
    n_dups = df.duplicated().sum()
    print(f"\n🔁 Duplicate rows: {n_dups:,}")

    # --- Target distribution ---
    target_col = "CarbonEmission"
    if target_col in df.columns:
        print(f"\n📊 Target column '{target_col}' distribution (yearly kg CO₂e):")
        print(df[target_col].describe().round(2).to_string())
        print(f"\n   Monthly equivalent (÷12):")
        monthly = df[target_col] / 12
        print(monthly.describe().round(2).to_string())

    # --- Categorical columns unique values ---
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"\n🏷️  Categorical columns unique values:")
    for col in cat_cols:
        vals = df[col].unique().tolist()
        print(f"   {col:<45} {vals}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    df = load_raw_data()
    inspect_data(df)
