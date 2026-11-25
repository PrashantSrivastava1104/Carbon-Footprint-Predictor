"""
src/train.py
------------
Step 3 of the ML pipeline: train 3 models, evaluate them, pick the best one,
and save everything needed by the Flask app.

What this script does (in order):
  1. Load and clean the raw data
  2. Split into train (80%) / test (20%) with random_state=42
  3. Compute Low / Medium / High thresholds from TRAINING data only
  4. Build sklearn Pipelines with ColumnTransformer (no data leakage!)
  5. Train: LinearRegression, RandomForestRegressor, XGBRegressor
  6. Evaluate each on HELD-OUT TEST SET: R2, MAE, RMSE
  7. Run 5-fold cross-validation on training set
  8. Report train vs test R2 and flag overfitting
  9. Select best model, explain why
 10. Save: best_pipeline.joblib, label_thresholds.json
 11. Save: model_comparison.csv and model_comparison.md
 12. Generate and save feature importance chart

Usage:
    python src/train.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on Windows)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_PATH, THRESHOLDS_PATH, METRICS_CSV_PATH, METRICS_MD_PATH,
    CHART_PATH, PROCESSED_PATH, TARGET_COL
)
from src.data_loader import load_raw_data
from src.preprocess import clean_data, get_feature_columns, save_processed


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_metrics(model_name: str, pipeline, X_train, y_train, X_test, y_test) -> dict:
    """
    Evaluate a fitted pipeline on both train and test sets.
    Reports R2, MAE, and RMSE. All numbers come from real predictions -
    never hardcoded.
    """
    y_train_pred = pipeline.predict(X_train)
    y_test_pred  = pipeline.predict(X_test)

    train_r2  = r2_score(y_train, y_train_pred)
    test_r2   = r2_score(y_test,  y_test_pred)
    test_mae  = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    overfit_flag = "YES" if (train_r2 - test_r2) > 0.05 else "No"

    return {
        "model":        model_name,
        "train_r2":     round(train_r2, 4),
        "test_r2":      round(test_r2, 4),
        "test_mae":     round(test_mae, 4),
        "test_rmse":    round(test_rmse, 4),
        "overfit_flag": overfit_flag,
    }


def run_cross_validation(model_name: str, pipeline, X_train, y_train, cv: int = 5) -> dict:
    """
    5-fold cross-validation on the TRAINING set only.
    Returns mean and std of R2 across folds.
    """
    print(f"   Running {cv}-fold CV for {model_name} ...")
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=cv, scoring="r2", n_jobs=-1)
    return {
        "model":       model_name,
        "cv_r2_mean":  round(scores.mean(), 4),
        "cv_r2_std":   round(scores.std(), 4),
    }


def build_pipeline(model, cat_cols: list, num_cols: list) -> Pipeline:
    """
    Build a full sklearn Pipeline:
      - Numeric features: StandardScaler
      - Categorical features: OneHotEncoder (unknown categories -> all zeros)

    Why Pipeline + ColumnTransformer?
    - The scaler and encoder are FIT on training data only.
    - When we call pipeline.predict() later, they TRANSFORM using training stats.
    - This guarantees zero data leakage between train and test sets.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def compute_label_thresholds(y_train: pd.Series) -> dict:
    """
    Compute Low / Medium / High boundaries using training set quantiles.

    Why quantiles?
    - They adapt to the actual data distribution rather than arbitrary absolutes.
    - Computed from TRAINING data only - test labels never influence thresholds.
    - 33rd and 67th percentiles split the data into roughly equal thirds.

    Returns a dict with 'low_max' and 'high_min' (monthly kg CO2e).
    """
    low_max  = float(np.percentile(y_train, 33))
    high_min = float(np.percentile(y_train, 67))
    return {"low_max": round(low_max, 2), "high_min": round(high_min, 2)}


def label_footprint(value: float, thresholds: dict) -> str:
    """Convert a numeric prediction to Low / Medium / High."""
    if value <= thresholds["low_max"]:
        return "Low"
    elif value >= thresholds["high_min"]:
        return "High"
    else:
        return "Medium"


def get_feature_names(pipeline: Pipeline, cat_cols: list, num_cols: list) -> list:
    """
    Extract the full list of feature names after preprocessing.
    Numeric features keep their original names; OHE features are expanded.
    """
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    return num_cols + ohe_names


def plot_feature_importance(pipeline: Pipeline, cat_cols: list, num_cols: list,
                             model_name: str, save_path: str) -> None:
    """
    Generate and save a horizontal bar chart of the top 15 features
    by importance (for tree-based models) or |coefficient| (for LinearRegression).
    """
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline, cat_cols, num_cols)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_type = "Feature Importance (Gini / Gain)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        importance_type = "|Coefficient| (Standardised)"
    else:
        print("   WARNING: Model has no feature importances. Skipping chart.")
        return

    # Build a sorted DataFrame
    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(15)

    # Clean up OHE names for readability: "diet_vegan" -> "Diet: vegan"
    def prettify(name: str) -> str:
        if "_" in name:
            parts = name.split("_", 1)
            return f"{parts[0].replace('_', ' ').title()}: {parts[1].replace('_', ' ')}"
        return name.replace("_", " ").title()

    fi_df["label"] = fi_df["feature"].apply(prettify)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_df)))
    bars = ax.barh(fi_df["label"][::-1], fi_df["importance"][::-1],
                   color=colors[::-1], edgecolor="none", height=0.6)

    ax.set_xlabel(importance_type, color="#94a3b8", fontsize=11)
    ax.set_title(f"Top 15 Features - {model_name}", color="#f1f5f9", fontsize=14, pad=15)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="x", color="#1e293b", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   [SAVED] Feature importance chart: {save_path}")


def save_metrics_table(results: list, cv_results: list):
    """
    Merge train/test results with CV results and save as CSV and Markdown.
    Every number comes from actual model runs - nothing is hardcoded.
    """
    df_res = pd.DataFrame(results)
    df_cv  = pd.DataFrame(cv_results)
    df_all = df_res.merge(df_cv, on="model")

    # Reorder columns
    df_all = df_all[[
        "model", "train_r2", "test_r2", "cv_r2_mean", "cv_r2_std",
        "test_mae", "test_rmse", "overfit_flag"
    ]]

    # Save CSV
    os.makedirs(os.path.dirname(METRICS_CSV_PATH), exist_ok=True)
    df_all.to_csv(METRICS_CSV_PATH, index=False)
    print(f"   [SAVED] Metrics CSV: {METRICS_CSV_PATH}")

    # Save Markdown
    md_lines = [
        "# Model Comparison Results\n",
        "_Generated automatically by src/train.py - all numbers from real model runs._\n\n",
        "| Model | Train R2 | Test R2 | CV R2 (mean+-std) | Test MAE | Test RMSE | Overfit? |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in df_all.iterrows():
        md_lines.append(
            f"| {row['model']} | {row['train_r2']:.4f} | {row['test_r2']:.4f} | "
            f"{row['cv_r2_mean']:.4f}+-{row['cv_r2_std']:.4f} | "
            f"{row['test_mae']:.2f} | {row['test_rmse']:.2f} | {row['overfit_flag']} |"
        )
    with open(METRICS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"   [SAVED] Metrics Markdown: {METRICS_MD_PATH}")

    return df_all


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    print("=" * 70)
    print("CARBON FOOTPRINT PREDICTOR - TRAINING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and clean data
    # ------------------------------------------------------------------
    print("\n[LOAD] Loading raw data ...")
    df_raw   = load_raw_data()
    print(f"   Raw shape: {df_raw.shape}")

    print("\n[CLEAN] Cleaning data ...")
    df_clean = clean_data(df_raw, verbose=True)
    save_processed(df_clean)

    # ------------------------------------------------------------------
    # 2. Prepare features and target
    # ------------------------------------------------------------------
    cat_cols, num_cols = get_feature_columns(df_clean)
    X = df_clean[cat_cols + num_cols]
    y = df_clean[TARGET_COL]

    print(f"\n[FEATURES] {len(cat_cols)} categorical + {len(num_cols)} numeric = {len(cat_cols)+len(num_cols)} total")
    print(f"   Categorical: {cat_cols}")
    print(f"   Numeric:     {num_cols}")

    # ------------------------------------------------------------------
    # 3. Train / test split (80/20, fixed seed) BEFORE any preprocessing
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[SPLIT] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # ------------------------------------------------------------------
    # 4. Compute label thresholds FROM TRAINING DATA ONLY
    # ------------------------------------------------------------------
    thresholds = compute_label_thresholds(y_train)
    print(f"\n[THRESHOLDS] Derived from training set quantiles only:")
    print(f"   Low   : <= {thresholds['low_max']:.2f} kg CO2e/month  (33rd pct)")
    print(f"   Medium: {thresholds['low_max']:.2f} - {thresholds['high_min']:.2f} kg CO2e/month")
    print(f"   High  : >= {thresholds['high_min']:.2f} kg CO2e/month  (67th pct)")

    # ------------------------------------------------------------------
    # 5. Define the 3 models to compare
    # ------------------------------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":           XGBRegressor(
                                 n_estimators=200,
                                 random_state=42,
                                 eval_metric="rmse",
                                 verbosity=0,
                             ),
    }

    # ------------------------------------------------------------------
    # 6. Train, evaluate, and cross-validate each model
    # ------------------------------------------------------------------
    fitted_pipelines = {}
    all_results      = []
    all_cv           = []

    print("\n[TRAINING] Fitting 3 models (this may take 1-2 minutes) ...")
    for name, estimator in models.items():
        print(f"\n   === {name} ===")
        pipeline = build_pipeline(estimator, cat_cols, num_cols)

        # Fit on training data ONLY - scaler/encoder see only training rows
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on HELD-OUT test set
        metrics = compute_metrics(name, pipeline, X_train, y_train, X_test, y_test)
        all_results.append(metrics)
        print(f"   Train R2 : {metrics['train_r2']:.4f}")
        print(f"   Test  R2 : {metrics['test_r2']:.4f}")
        print(f"   Test  MAE: {metrics['test_mae']:.2f} kg CO2e/month")
        print(f"   Test RMSE: {metrics['test_rmse']:.2f} kg CO2e/month")
        print(f"   Overfit? : {metrics['overfit_flag']}")

        # 5-fold CV on training set only
        cv_scores = run_cross_validation(name, pipeline, X_train, y_train, cv=5)
        all_cv.append(cv_scores)
        print(f"   CV R2    : {cv_scores['cv_r2_mean']:.4f} +/- {cv_scores['cv_r2_std']:.4f}")

    # ------------------------------------------------------------------
    # 7. Save metrics table (CSV + Markdown)
    # ------------------------------------------------------------------
    print("\n[METRICS] Saving results table ...")
    df_metrics = save_metrics_table(all_results, all_cv)
    print("\nFull comparison table:")
    print(df_metrics.to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Select the best model by test R2 (CV R2 as tiebreaker)
    # ------------------------------------------------------------------
    df_metrics_sorted = df_metrics.sort_values(
        ["test_r2", "cv_r2_mean"], ascending=False
    ).reset_index(drop=True)
    best_name     = df_metrics_sorted.loc[0, "model"]
    best_pipeline = fitted_pipelines[best_name]

    print(f"\n[WINNER] Best model: {best_name}")
    print(f"   Test R2: {df_metrics_sorted.loc[0, 'test_r2']:.4f}")
    print(f"   CV R2  : {df_metrics_sorted.loc[0, 'cv_r2_mean']:.4f}")

    explanations = {
        "XGBoost": (
            "XGBoost won because it handles non-linear interactions between "
            "features (e.g., how diet interacts with transport) better than "
            "linear models, and its gradient-boosting approach corrects "
            "previous errors iteratively."
        ),
        "Random Forest": (
            "Random Forest won because it handles non-linear relationships well "
            "and is robust to outliers. It beat XGBoost possibly because the "
            "data distribution is smoother, making bagging more effective than boosting."
        ),
        "Linear Regression": (
            "Linear Regression won, suggesting the relationship between features "
            "and carbon emissions is approximately linear in this synthetic dataset. "
            "This is the most interpretable model and easiest to explain."
        ),
    }
    print(f"\n   Why {best_name}?")
    print(f"   {explanations.get(best_name, '')}")

    # ------------------------------------------------------------------
    # 9. Save best pipeline and label thresholds
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\n[SAVED] Pipeline -> {MODEL_PATH}")

    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[SAVED] Thresholds -> {THRESHOLDS_PATH}")

    # ------------------------------------------------------------------
    # 10. Feature importance chart
    # ------------------------------------------------------------------
    print("\n[CHART] Generating feature importance chart ...")
    plot_feature_importance(best_pipeline, cat_cols, num_cols, best_name, CHART_PATH)

    # ------------------------------------------------------------------
    # 11. Final summary (copy these numbers into README / INTERVIEW_NOTES)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    best_row = df_metrics[df_metrics["model"] == best_name].iloc[0]
    print(f"  Best model       : {best_name}")
    print(f"  Test R2          : {best_row['test_r2']:.4f}")
    print(f"  Test MAE         : {best_row['test_mae']:.2f} kg CO2e/month")
    print(f"  Test RMSE        : {best_row['test_rmse']:.2f} kg CO2e/month")
    print(f"  CV R2 (5-fold)   : {best_row['cv_r2_mean']:.4f} +/- {best_row['cv_r2_std']:.4f}")
    print(f"  Label thresholds :")
    print(f"    Low    <= {thresholds['low_max']:.2f} kg CO2e/month")
    print(f"    Medium    {thresholds['low_max']:.2f} - {thresholds['high_min']:.2f}")
    print(f"    High   >= {thresholds['high_min']:.2f} kg CO2e/month")
    if best_row["test_r2"] > 0.97:
        print("\n  NOTE: R2 > 0.97 is expected because this dataset is synthetically")
        print("  generated from a formula. The model essentially re-learns the formula.")
        print("  See README Limitations section for full explanation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
