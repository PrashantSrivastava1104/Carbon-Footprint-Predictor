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
    pipeline   = joblib.load(MODEL_PATH)
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
        "how_many_new_clothes_monthly":"New Clothes per Month",
        "how_long_internet_daily_hour":"Daily Internet Hours",
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


def get_top_factor_and_tip(input_data: dict, pipeline) -> tuple:
    """
    Identify the feature with the highest importance for this model,
    then return a human-readable factor name and an actionable tip.

    Strategy:
    - For tree-based models (RF, XGBoost): use feature_importances_ from the model.
    - Map the most important feature name back to a readable label.
    - Look up the tip from TIPS dict.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get feature names after preprocessing
    cat_cols = preprocessor.transformers_[1][2]   # categorical columns list
    num_cols = preprocessor.transformers_[0][2]   # numeric columns list

    ohe = preprocessor.named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + ohe_names

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_)
    else:
        return "your lifestyle choices", DEFAULT_TIP

    # Find most important feature
    best_idx = importances.argmax()
    best_feature = all_feature_names[best_idx]

    # Strip OHE suffix to get base column name (e.g., "diet_vegan" → "diet")
    base_col = best_feature.split("_")[0] if "_" in best_feature else best_feature
    # Try exact match first, then prefix match
    tip_key = best_feature if best_feature in TIPS else base_col
    # Try matching any key that starts with base_col
    if tip_key not in TIPS:
        for k in TIPS:
            if best_feature.startswith(k):
                tip_key = k
                break

    # Human-readable factor name
    readable = best_feature.replace("_", " ").title()
    tip = TIPS.get(tip_key, DEFAULT_TIP)

    return readable, tip


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
    input_df["monthly_grocery_bill"] = input_df["monthly_grocery_bill"] / INR_TO_USD_RATE
    prediction = float(PIPELINE.predict(input_df)[0])
    prediction = max(0.0, prediction)   # Clamp to non-negative (CO2 can't be negative)

    label = label_footprint(prediction)
    top_factor, tip = get_top_factor_and_tip(cleaned, PIPELINE)

    return render_template(
        "result.html",
        prediction=round(prediction, 1),
        label=label,
        top_factor=top_factor,
        tip=tip,
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
        os.path.join(os.path.dirname(__file__), "static", "charts", "feature_importance.png")
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
