"""
src/preprocess.py
-----------------
Step 2 of the ML pipeline: clean the raw data and return the final
feature matrix (X) and target vector (y).

Cleaning decisions (each one documented with rationale):
1. Standardise column names → clean snake_case
2. Convert target to MONTHLY kg CO₂e (divide by 12)
   Rationale: CarbonEmission is in kg/year (mean ~2269 ÷ 12 = ~189/month),
   matching the monthly inputs the user provides.
3. No duplicate rows found (confirmed in data inspection)
4. Vehicle Type: 6,721 / 10,000 NaN values — these all correspond to rows
   where Transport = "public" or "walk/bicycle" (person has no vehicle).
   Impute as "none" rather than dropping 67% of the data.
5. Cap numeric outliers with IQR × 3 rule (soft cap: keeps extreme but
   legitimate values, only removes statistical errors / data entry mistakes)
6. Recycling and Cooking_With are stored as Python-list-formatted strings
   e.g. "['Paper', 'Metal']". Parse with ast.literal_eval and compute count.
7. Strip whitespace from all string columns and lowercase for consistency.

NOTE: NO encoding or scaling is done here — that is handled inside the
sklearn Pipeline in train.py so there is ZERO data leakage.
"""

import os
import sys
import ast
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_DATA_PATH, PROCESSED_PATH, TARGET_COL


# ---------------------------------------------------------------------------
# Column name mapping: raw name → clean name
# ---------------------------------------------------------------------------
RENAME_MAP = {
    "Body Type":                        "body_type",
    "Sex":                              "sex",
    "Diet":                             "diet",
    "How Often Shower":                 "how_often_shower",
    "Heating Energy Source":            "heating_energy_source",
    "Transport":                        "transport",
    "Vehicle Type":                     "vehicle_type",
    "Social Activity":                  "social_activity",
    "Monthly Grocery Bill":             "monthly_grocery_bill",
    "Frequency of Traveling by Air":    "air_travel_frequency",
    "Vehicle Monthly Distance Km":      "vehicle_monthly_distance_km",
    "Waste Bag Size":                   "waste_bag_size",
    "Waste Bag Weekly Count":           "waste_bag_weekly_count",
    "How Long TV PC Daily Hour":        "how_long_tv_pc_daily_hour",
    "How Many New Clothes Monthly":     "how_many_new_clothes_monthly",
    "How Long Internet Daily Hour":     "how_long_internet_daily_hour",
    "Energy efficiency":                "energy_efficiency",
    "Recycling":                        "recycling",
    "Cooking_With":                     "cooking_with",
    "CarbonEmission":                   "carbon_emission_yearly",
}


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to clean snake_case names."""
    df = df.rename(columns=RENAME_MAP)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def parse_list_string_to_count(value) -> int:
    """
    Parse a Python-list-formatted string into a count of items.

    Examples:
        "['Paper', 'Metal']"  → 2
        "[]"                  → 0
        NaN                   → 0
    """
    if pd.isna(value):
        return 0
    s = str(value).strip()
    if not s or s in ("[]", "['']", '[""]'):
        return 0
    try:
        parsed = ast.literal_eval(s)
        return len([x for x in parsed if str(x).strip()])
    except (ValueError, SyntaxError):
        # Fallback: count comma-separated items
        return len([x for x in s.strip("[]").split(",") if x.strip()])


def cap_outliers_iqr(df: pd.DataFrame, numeric_cols: list, factor: float = 3.0) -> pd.DataFrame:
    """
    Cap outliers using IQR × factor rule (Tukey fences).

    Why 3× instead of 1.5×?
    1.5× is for detecting mild outliers; 3× is for extreme outliers only.
    We preserve genuine high-emission lifestyles and only remove clear
    data entry errors (e.g., 99999 km driven in a month).
    """
    df = df.copy()
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_capped > 0:
            print(f"      Capped {n_capped} outliers in '{col}' "
                  f"[floor={lower:.1f}, ceil={upper:.1f}]")
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns a clean DataFrame with:
    - Standardised column names
    - Monthly target column (TARGET_COL = 'monthly_carbon_kg')
    - No NaN in Vehicle Type (imputed as 'none')
    - Capped numeric outliers
    - Multi-select columns converted to integer counts
    - All strings lowercased and stripped

    Parameters
    ----------
    df : pd.DataFrame  — raw dataframe from load_raw_data()
    verbose : bool     — print cleaning progress

    Returns
    -------
    pd.DataFrame — cleaned dataframe ready for train/test split
    """
    if verbose:
        print("\n🧹 CLEANING PIPELINE")
        print("-" * 60)

    # Step 1: Standardise column names
    df = standardise_column_names(df)
    if verbose:
        print(f"   Step 1 ✅ Columns renamed. Shape: {df.shape}")

    # Step 2: Convert target to monthly kg CO₂e
    # CarbonEmission is in kg CO₂e per YEAR (mean ~2,269).
    # Dividing by 12 gives per-month values (~189 avg), which matches the
    # monthly inputs the user provides in the app form.
    df[TARGET_COL] = df["carbon_emission_yearly"] / 12.0
    df = df.drop(columns=["carbon_emission_yearly"])
    if verbose:
        monthly_min = df[TARGET_COL].min()
        monthly_max = df[TARGET_COL].max()
        monthly_mean = df[TARGET_COL].mean()
        print(f"   Step 2 ✅ Target = monthly kg CO₂e (yearly ÷ 12). "
              f"Range: [{monthly_min:.1f}, {monthly_max:.1f}], mean={monthly_mean:.1f}")

    # Step 3: No duplicates found (confirmed in inspection), log anyway
    n_dups = df.duplicated().sum()
    if verbose:
        print(f"   Step 3 ✅ Duplicate rows: {n_dups} (none to drop)")
    if n_dups > 0:
        df = df.drop_duplicates()

    # Step 4: Vehicle Type NaN → 'none'
    # 6,721 of 10,000 rows have NaN Vehicle Type because those persons use
    # public transport or walk/cycle. Imputing as 'none' is semantically
    # correct (they have no vehicle) rather than dropping 67% of the data.
    n_veh_null = df["vehicle_type"].isnull().sum()
    df["vehicle_type"] = df["vehicle_type"].fillna("none")
    if verbose:
        print(f"   Step 4 ✅ Vehicle Type: imputed {n_veh_null:,} NaN → 'none' "
              f"(these rows use public/walk transport)")

    # Step 5: Parse multi-select string columns into counts
    # Recycling: "['Paper', 'Metal']" → 2 (number of things recycled)
    # Cooking_With: "['Stove', 'Oven']" → 2 (number of cooking devices)
    df["recycling_count"] = df["recycling"].apply(parse_list_string_to_count)
    df["cooking_devices_count"] = df["cooking_with"].apply(parse_list_string_to_count)
    df = df.drop(columns=["recycling", "cooking_with"])
    if verbose:
        print(f"   Step 5 ✅ recycling → recycling_count, cooking_with → cooking_devices_count")

    # Step 6: Strip whitespace and lowercase all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.lower()
    if verbose:
        print(f"   Step 6 ✅ Lowercased and stripped all categorical columns")

    # Step 7: Cap numeric outliers (IQR × 3)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != TARGET_COL]
    if verbose:
        print(f"   Step 7 — Checking outliers in {len(numeric_cols)} numeric columns:")
    df = cap_outliers_iqr(df, numeric_cols, factor=3.0)
    if verbose:
        print(f"            ✅ Outlier capping complete")

    if verbose:
        print(f"\n   📐 Final cleaned shape: {df.shape}")
        print(f"   📊 Target stats (monthly kg CO₂e):")
        print(df[TARGET_COL].describe().round(2).to_string())
        print("-" * 60)

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple:
    """
    Return (categorical_columns, numeric_columns) lists.
    Excludes the target column.
    """
    all_cols = [c for c in df.columns if c != TARGET_COL]
    cat_cols = df[all_cols].select_dtypes(include="object").columns.tolist()
    num_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    return cat_cols, num_cols


def save_processed(df: pd.DataFrame, filepath: str = PROCESSED_PATH) -> None:
    """Save the cleaned dataframe to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"   💾 Saved cleaned data to: {filepath}")


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw, verbose=True)
    save_processed(df_clean)
    print("\nSample cleaned row:")
    print(df_clean.iloc[0].to_dict())
