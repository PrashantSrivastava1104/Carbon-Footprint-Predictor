"""
tests/test_predictor.py
-----------------------
Basic tests for the Carbon Footprint Predictor.
Tests run with: python -m pytest tests/ -v

What we test:
1. Prediction function returns a positive float
2. Invalid input is rejected with 400 status
3. Label threshold logic works correctly
4. Feature importance chart was generated
5. Metrics CSV was saved with all 3 models
6. Cleaned data has no NaN in important columns
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_client():
    """Create a test client for the Flask app."""
    from app import app
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client


@pytest.fixture(scope="module")
def pipeline_and_thresholds():
    """Load the trained pipeline and thresholds."""
    import joblib
    from config import MODEL_PATH, THRESHOLDS_PATH
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
    pipeline = joblib.load(MODEL_PATH)
    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)
    return pipeline, thresholds


# A valid input that covers all required fields
VALID_INPUT = {
    "body_type":                    "normal",
    "sex":                          "male",
    "diet":                         "omnivore",
    "how_often_shower":             "daily",
    "heating_energy_source":        "natural gas",
    "transport":                    "private",
    "vehicle_type":                 "petrol",
    "social_activity":              "sometimes",
    "monthly_grocery_bill":         "300",
    "air_travel_frequency":         "rarely",
    "vehicle_monthly_distance_km":  "400",
    "waste_bag_size":               "medium",
    "waste_bag_weekly_count":       "3",
    "how_long_tv_pc_daily_hour":    "4",
    "how_many_new_clothes_monthly": "2",
    "how_long_internet_daily_hour": "5",
    "energy_efficiency":            "sometimes",
    # Multi-selects (sent as multiple values)
    "recycling":                    "paper",
    "cooking_with":                 "stove",
}


# ---------------------------------------------------------------------------
# Test 1: Prediction returns a positive number
# ---------------------------------------------------------------------------
def test_prediction_returns_positive_number(pipeline_and_thresholds):
    """
    The model should return a positive float when given a valid input.
    Carbon emissions cannot be negative.
    """
    pipeline, thresholds = pipeline_and_thresholds
    input_data = {
        "body_type": "normal",
        "sex": "male",
        "diet": "omnivore",
        "how_often_shower": "daily",
        "heating_energy_source": "natural gas",
        "transport": "private",
        "vehicle_type": "petrol",
        "social_activity": "sometimes",
        "monthly_grocery_bill": 300.0,
        "air_travel_frequency": "rarely",
        "vehicle_monthly_distance_km": 400.0,
        "waste_bag_size": "medium",
        "waste_bag_weekly_count": 3,
        "how_long_tv_pc_daily_hour": 4.0,
        "how_many_new_clothes_monthly": 2,
        "how_long_internet_daily_hour": 5.0,
        "energy_efficiency": "sometimes",
        "recycling_count": 1,
        "cooking_devices_count": 1,
    }
    df = pd.DataFrame([input_data])
    prediction = float(pipeline.predict(df)[0])
    assert isinstance(prediction, float), "Prediction should be a float"
    assert prediction >= 0, f"Prediction should be non-negative, got {prediction}"
    print(f"\n✅ Prediction = {prediction:.2f} kg CO₂e/month")


# ---------------------------------------------------------------------------
# Test 2: Valid form input returns 200 on /predict
# ---------------------------------------------------------------------------
def test_valid_input_returns_200(app_client):
    """A complete, valid POST to /predict should return 200."""
    response = app_client.post("/predict", data=VALID_INPUT, follow_redirects=True)
    assert response.status_code == 200, (
        f"Expected 200 for valid input, got {response.status_code}. "
        f"Response: {response.data[:500]}"
    )


# ---------------------------------------------------------------------------
# Test 3: Invalid input (non-numeric) returns 400
# ---------------------------------------------------------------------------
def test_invalid_numeric_input_rejected(app_client):
    """
    Sending a string where a number is expected should return 400.
    The app should NEVER crash on bad input.
    """
    bad_input = dict(VALID_INPUT)
    bad_input["monthly_grocery_bill"] = "not_a_number"
    response = app_client.post("/predict", data=bad_input)
    assert response.status_code == 400, (
        f"Expected 400 for invalid numeric input, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Test 4: Out-of-range input returns 400
# ---------------------------------------------------------------------------
def test_out_of_range_input_rejected(app_client):
    """
    Values outside the allowed range should be rejected with 400.
    For example, monthly_grocery_bill > 2000 is too high.
    """
    bad_input = dict(VALID_INPUT)
    bad_input["monthly_grocery_bill"] = "99999"
    response = app_client.post("/predict", data=bad_input)
    assert response.status_code == 400, (
        f"Expected 400 for out-of-range input, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Test 5: Missing required field returns 400
# ---------------------------------------------------------------------------
def test_missing_field_returns_400(app_client):
    """Missing a required categorical field should return 400."""
    bad_input = dict(VALID_INPUT)
    del bad_input["diet"]
    response = app_client.post("/predict", data=bad_input)
    assert response.status_code == 400, (
        f"Expected 400 for missing field, got {response.status_code}"
    )


# ---------------------------------------------------------------------------
# Test 6: Label thresholds logic
# ---------------------------------------------------------------------------
def test_label_thresholds_correct(pipeline_and_thresholds):
    """
    Given known thresholds, the label_footprint function should return
    the correct label for values in each bin.
    """
    from app import label_footprint
    _, thresholds = pipeline_and_thresholds
    low_max = thresholds["low_max"]
    high_min = thresholds["high_min"]

    # A value clearly below low_max should be "Low"
    assert label_footprint(low_max * 0.5) == "Low", "Should be Low"
    # A value clearly between thresholds should be "Medium"
    mid = (low_max + high_min) / 2
    assert label_footprint(mid) == "Medium", "Should be Medium"
    # A value clearly above high_min should be "High"
    assert label_footprint(high_min * 1.5) == "High", "Should be High"
    # Edge: exactly at low_max boundary
    assert label_footprint(low_max) in ("Low", "Medium"), "Boundary Low/Med"
    print(f"\n✅ Thresholds: Low≤{low_max:.1f}, High≥{high_min:.1f}")


# ---------------------------------------------------------------------------
# Test 7: Feature importance chart exists
# ---------------------------------------------------------------------------
def test_feature_importance_chart_exists():
    """The training script should have generated the feature importance chart."""
    from config import CHART_PATH
    assert os.path.exists(CHART_PATH), (
        f"Feature importance chart not found at {CHART_PATH}. "
        "Run src/train.py first."
    )
    # Also check it's not empty
    size = os.path.getsize(CHART_PATH)
    assert size > 1000, f"Chart file seems too small ({size} bytes). May be corrupt."
    print(f"\n✅ Feature chart exists ({size:,} bytes)")


# ---------------------------------------------------------------------------
# Test 8: Metrics CSV was generated with all 3 models
# ---------------------------------------------------------------------------
def test_metrics_csv_has_all_three_models():
    """The metrics CSV should contain exactly 3 rows (one per model)."""
    from config import METRICS_CSV_PATH
    if not os.path.exists(METRICS_CSV_PATH):
        pytest.skip("Metrics CSV not found. Run src/train.py first.")
    df = pd.read_csv(METRICS_CSV_PATH)
    assert len(df) == 3, f"Expected 3 models, got {len(df)}"
    expected_models = {"Linear Regression", "Random Forest", "XGBoost"}
    actual_models = set(df["model"].tolist())
    assert actual_models == expected_models, (
        f"Expected models {expected_models}, got {actual_models}"
    )
    # Check all metric columns are present
    for col in ["train_r2", "test_r2", "cv_r2_mean", "test_mae", "test_rmse"]:
        assert col in df.columns, f"Missing column {col}"
    # Check no NaN in metric columns (all numbers must be real)
    assert df[["train_r2", "test_r2", "cv_r2_mean", "test_mae", "test_rmse"]].isnull().sum().sum() == 0
    print(f"\n✅ Metrics CSV has 3 models: {actual_models}")


# ---------------------------------------------------------------------------
# Test 9: Cleaned data has no NaN in feature columns
# ---------------------------------------------------------------------------
def test_cleaned_data_no_nan():
    """The processed data should have no NaN in feature or target columns."""
    from config import PROCESSED_PATH
    if not os.path.exists(PROCESSED_PATH):
        pytest.skip("Processed data not found. Run src/train.py first.")
    df = pd.read_csv(PROCESSED_PATH)
    nulls = df.isnull().sum()
    cols_with_nulls = nulls[nulls > 0]
    assert len(cols_with_nulls) == 0, (
        f"Found NaN in cleaned data: {cols_with_nulls.to_dict()}"
    )
    print(f"\n✅ Cleaned data has zero NaN across {df.shape[1]} columns")


# ---------------------------------------------------------------------------
# Test 10: Home page loads
# ---------------------------------------------------------------------------
def test_home_page_loads(app_client):
    """GET / should return 200 and contain the form."""
    response = app_client.get("/")
    assert response.status_code == 200
    assert b"Carbon" in response.data or b"carbon" in response.data
    print("\n✅ Home page loads successfully")


# ---------------------------------------------------------------------------
# Test 11: About page loads
# ---------------------------------------------------------------------------
def test_about_page_loads(app_client):
    """GET /about should return 200."""
    response = app_client.get("/about")
    assert response.status_code == 200
    print("\n✅ About page loads successfully")


# ---------------------------------------------------------------------------
# Test 12: Health check endpoint
# ---------------------------------------------------------------------------
def test_health_check(app_client):
    """GET /health should return JSON with status ok."""
    response = app_client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "ok"
    print(f"\n✅ Health check: {data}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
