"""
app.py
------
Flask web application for the Carbon Footprint Predictor.

Routes:
  GET  /          → Home page with input form
  POST /predict   → Validate inputs, run model, show result
  GET  /about     → Model comparison table + feature importance chart

The model is loaded ONCE at startup (not per-request) for efficiency.
All inputs are validated with friendly error messages — no crashes on bad input.
"""

import os
import sys
import json
import joblib
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request, jsonify

from config import (
    MODEL_PATH, THRESHOLDS_PATH, METRICS_MD_PATH,
    ALLOWED_VALUES, NUMERIC_LIMITS, TIPS, DEFAULT_TIP,
    SECRET_KEY, DEBUG, PORT, TARGET_COL, INR_TO_USD_RATE
)

# ============================================================
# App initialisation
# ============================================================
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ============================================================
# Load model and thresholds ONCE at startup
# ============================================================


def load_model_artifacts():
    """
    Load the trained pipeline and label thresholds from disk.
    Called once when the app starts — not on every request.
    Returns (pipeline, thresholds) or raises a clear error.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run 'python src/train.py' first to train and save the model."
        )
    if not os.path.exists(THRESHOLDS_PATH):
        raise FileNotFoundError(
            f"Thresholds not found at {THRESHOLDS_PATH}. "
            "Run 'python src/train.py' first."
        )
    pipeline = joblib.load(MODEL_PATH)
    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)
    return pipeline, thresholds


try:
    PIPELINE, THRESHOLDS = load_model_artifacts()
    print("[OK] Model and thresholds loaded successfully.")
except FileNotFoundError as e:
    PIPELINE, THRESHOLDS = None, None
    print(f"[WARN]  {e}")


# ============================================================
# Input validation
# ============================================================

def validate_inputs(form: dict) -> tuple:
    """
    Validate all form inputs.

    Returns:
        (cleaned_data: dict, errors: list)
        If errors is non-empty, cleaned_data is None.
    """
    errors = []
    cleaned = {}

    # --- Categorical fields ---
    categorical_fields = {
        "body_type":          "Body Type",
        "sex":                "Sex",
        "diet":               "Diet",
        "how_often_shower":   "Shower Frequency",
        "heating_energy_source": "Heating Energy Source",
        "transport":          "Main Transport Mode",
        "vehicle_type":       "Vehicle Type",
        "social_activity":    "Social Activity",
        "air_travel_frequency":  "Air Travel Frequency",
        "waste_bag_size":     "Waste Bag Size",
        "energy_efficiency":  "Energy Efficiency",
    }
    for field, label in categorical_fields.items():
        val = form.get(field, "").strip().lower()
        allowed = ALLOWED_VALUES.get(field, [])
        if not val:
            errors.append(f"{label} is required.")
        elif val not in allowed:
            errors.append(f"{label}: '{val}' is not a valid option.")
        else:
            cleaned[field] = val

    # --- Multi-select: recycling (stored as count) ---
    recycling_vals = form.getlist("recycling")
    cleaned["recycling_count"] = len(recycling_vals)

    # --- Multi-select: cooking_with (stored as count) ---
    cooking_vals = form.getlist("cooking_with")
    if not cooking_vals:
        errors.append("Please select at least one cooking method.")
    else:
        cleaned["cooking_devices_count"] = len(cooking_vals)

    # --- Numeric fields ---
    numeric_fields = {
        "monthly_grocery_bill":        "Monthly Grocery Bill",
        "vehicle_monthly_distance_km": "Monthly Vehicle Distance (km)",
        "waste_bag_weekly_count":      "Waste Bags per Week",
        "how_long_tv_pc_daily_hour":   "Daily TV/PC Hours",
        "how_many_new_clothes_monthly": "New Clothes per Month",
        "how_long_internet_daily_hour": "Daily Internet Hours",
    }
    for field, label in numeric_fields.items():
        raw = form.get(field, "").strip()
        if not raw:
            errors.append(f"{label} is required.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label}: must be a number, got '{raw}'.")
            continue
        min_v, max_v = NUMERIC_LIMITS.get(field, (0, 1e9))
        if val < min_v or val > max_v:
            errors.append(f"{label}: must be between {min_v} and {max_v}.")
            continue
        # Store as int for count fields, float for others
        cleaned[field] = int(val) if field in ["waste_bag_weekly_count",
                                               "how_many_new_clothes_monthly"] else val

    return (None, errors) if errors else (cleaned, [])


# ============================================================
# Prediction logic
# ============================================================

def label_footprint(value: float) -> str:
    """Convert numeric prediction to Low / Medium / High."""
    if value <= THRESHOLDS["low_max"]:
        return "Low"
    elif value >= THRESHOLDS["high_min"]:
        return "High"
    return "Medium"


FACTOR_LABELS = {
    "diet": "Diet",
    "how_often_shower": "Shower frequency",
    "heating_energy_source": "Heating / cooking energy source",
    "transport": "Main transport mode",
    "vehicle_type": "Vehicle fuel type",
    "social_activity": "Social activity",
    "air_travel_frequency": "Air travel",
    "waste_bag_size": "Waste bag size",
    "energy_efficiency": "Appliance efficiency",
    "recycling_count": "Recycling",
    "cooking_devices_count": "Cooking methods",
    "monthly_grocery_bill": "Grocery spending",
    "vehicle_monthly_distance_km": "Monthly vehicle distance",
    "waste_bag_weekly_count": "Waste bags per week",
    "how_long_tv_pc_daily_hour": "Screen time",
    "how_many_new_clothes_monthly": "New clothes bought",
    "how_long_internet_daily_hour": "Internet / streaming time",
}

# Dataset columns whose names differ from the keys in config.TIPS
TIP_KEY_OVERRIDES = {
    "recycling_count": "recycling",
    "cooking_devices_count": "cooking_with",
}

# Things the user can't reasonably change, so we never pick them as the factor
NOT_ACTIONABLE = {"sex", "body_type"}

# Answers that mean the user is already doing well, so no tip is needed there
ALREADY_LOW_VALUES = {
    "diet": {"vegan", "vegetarian"},
    "transport": {"public", "walk/bicycle"},
    "vehicle_type": {"none", "electric"},
    "air_travel_frequency": {"never"},
    "energy_efficiency": {"yes"},
}


def _already_low(col: str, cleaned: dict) -> bool:
    value = cleaned.get(col)
    if col in ALREADY_LOW_VALUES:
        return value in ALREADY_LOW_VALUES[col]
    if col == "vehicle_monthly_distance_km":
        return value == 0
    if col == "recycling_count":
        return value is not None and value >= 3
    return False


def _base_column(feature_name: str, num_cols: list, cat_cols: list) -> str:
    """Map a transformed feature name (e.g. 'diet_vegan') back to its original column."""
    if feature_name in num_cols:
        return feature_name
    for col in sorted(cat_cols, key=len, reverse=True):
        if feature_name.startswith(col + "_"):
            return col
    return feature_name


def get_top_factor_and_tip(cleaned: dict, input_df: pd.DataFrame, pipeline) -> tuple:
    """
    Find the factor that adds the most CO2e to THIS user's prediction.

    Uses XGBoost's per-prediction contributions (pred_contribs), which are in
    kg CO2e/month and sum to the prediction. One-hot columns are added back
    together per original column. Returns (readable_name, tip, kg_added).
    """
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]

        num_cols = list(preprocessor.transformers_[0][2])
        cat_cols = list(preprocessor.transformers_[1][2])
        ohe = preprocessor.named_transformers_["cat"]
        all_names = num_cols + ohe.get_feature_names_out(cat_cols).tolist()

        booster = model.get_booster()
        X = preprocessor.transform(input_df)
        dmat = xgb.DMatrix(X, feature_names=booster.feature_names)
        contribs = booster.predict(dmat, pred_contribs=True)[
            0][:-1]  # drop bias term

        if len(all_names) != len(contribs):
            raise ValueError("Feature names and contributions do not line up.")

        totals = {}
        for name, value in zip(all_names, contribs):
            col = _base_column(name, num_cols, cat_cols)
            totals[col] = totals.get(col, 0.0) + float(value)

        candidates = {
            col: v for col, v in totals.items()
            if v > 0 and col not in NOT_ACTIONABLE and not _already_low(col, cleaned)
        }
        if not candidates:
            return "Your overall lifestyle", DEFAULT_TIP, None

        top_col = max(candidates, key=candidates.get)
        label = FACTOR_LABELS.get(
            top_col, top_col.replace("_", " ").capitalize())
        tip = TIPS.get(TIP_KEY_OVERRIDES.get(top_col, top_col), DEFAULT_TIP)
        return label, tip, round(candidates[top_col], 1)

    except Exception as e:
        print(f"[WARN] Per-user factor failed, using default tip: {e}")
        return "Your overall lifestyle", DEFAULT_TIP, None


def build_input_df(cleaned: dict) -> pd.DataFrame:
    """
    Convert the validated form data dict into a one-row DataFrame
    with the same column names as the training data.
    """
    return pd.DataFrame([cleaned])


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    """Home page — renders the input form."""
    return render_template(
        "index.html",
        allowed=ALLOWED_VALUES,
        model_ready=(PIPELINE is not None)
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST route: validate inputs → run model → render result page.
    Returns 400 with error messages if validation fails.
    Returns 503 if the model hasn't been trained yet.
    """
    if PIPELINE is None:
        return render_template(
            "index.html",
            error="Model not ready. Please run 'python src/train.py' first.",
            allowed=ALLOWED_VALUES,
            model_ready=False
        ), 503

    cleaned, errors = validate_inputs(request.form)

    if errors:
        return render_template(
            "index.html",
            errors=errors,
            form_data=request.form,
            allowed=ALLOWED_VALUES,
            model_ready=True
        ), 400

    # Build input DataFrame and predict
    # India localisation: grocery bill was entered in INR — convert to USD
    # before passing to the model (which was trained on USD values).
    input_df = build_input_df(cleaned)
    input_df["monthly_grocery_bill"] = input_df["monthly_grocery_bill"] / \
        INR_TO_USD_RATE
    prediction = float(PIPELINE.predict(input_df)[0])
    # Clamp to non-negative (CO2 can't be negative)
    prediction = max(0.0, prediction)

    label = label_footprint(prediction)
    top_factor, tip, factor_kg = get_top_factor_and_tip(
        cleaned, input_df, PIPELINE)

    return render_template(
        "result.html",
        prediction=round(prediction, 1),
        label=label,
        top_factor=top_factor,
        tip=tip,
        factor_kg=factor_kg,
        thresholds=THRESHOLDS,
        input_data=cleaned,
    )


@app.route("/about")
def about():
    """About page — model comparison table and feature importance chart."""
    metrics_md = ""
    metrics_table = []

    if os.path.exists(METRICS_MD_PATH):
        with open(METRICS_MD_PATH) as f:
            metrics_md = f.read()

    # Also parse CSV for structured table display
    from config import METRICS_CSV_PATH
    import csv
    if os.path.exists(METRICS_CSV_PATH):
        with open(METRICS_CSV_PATH) as f:
            reader = csv.DictReader(f)
            metrics_table = list(reader)

    chart_exists = os.path.exists(
        os.path.join(os.path.dirname(__file__), "static",
                     "charts", "feature_importance.png")
    )

    return render_template(
        "about.html",
        metrics_table=metrics_table,
        chart_exists=chart_exists,
        thresholds=THRESHOLDS,
    )


@app.route("/health")
def health():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        "status": "ok",
        "model_loaded": PIPELINE is not None,
    })


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT, host="0.0.0.0")
