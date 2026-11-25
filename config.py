"""
config.py
---------
Central configuration for the Carbon Footprint Predictor app.
Keeps all magic numbers, file paths, and allowed field values in one place
so they are easy to change without touching other files.

India localisation notes:
- Monthly grocery bill is collected in INR (Indian Rupees).
  It is converted to USD before model prediction using INR_TO_USD_RATE.
  (The underlying model was trained on USD values from the original dataset.)
- Vehicle distance is in km (already metric, no conversion needed).
- Tips are written for an Indian audience.
"""

import os

# ---------------------------------------------------------------------------
# Base paths — everything relative to this file's directory
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_pipeline.joblib")
THRESHOLDS_PATH = os.path.join(BASE_DIR, "models", "label_thresholds.json")
METRICS_CSV_PATH = os.path.join(
    BASE_DIR, "models", "metrics", "model_comparison.csv")
METRICS_MD_PATH = os.path.join(
    BASE_DIR, "models", "metrics", "model_comparison.md")
CHART_PATH = os.path.join(BASE_DIR, "static", "charts",
                          "feature_importance.png")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Carbon Emission.csv")
PROCESSED_PATH = os.path.join(
    BASE_DIR, "data", "processed", "cleaned_data.csv")

# ---------------------------------------------------------------------------
# Flask settings
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
PORT = int(os.environ.get("PORT", 5000))

# ---------------------------------------------------------------------------
# Target column name after cleaning
# ---------------------------------------------------------------------------
TARGET_COL = "monthly_carbon_kg"   # = CarbonEmission / 12

# ---------------------------------------------------------------------------
# India localisation
# ---------------------------------------------------------------------------
# The model was trained with grocery bill in USD.
# The UI collects it in INR. We divide by this rate before prediction.
INR_TO_USD_RATE = 83.0   # approximate INR per 1 USD (update as needed)

# ---------------------------------------------------------------------------
# Allowed categorical values (used for input validation in app.py)
# These match the categories in the dataset exactly.
# ---------------------------------------------------------------------------
ALLOWED_VALUES = {
    "body_type": ["underweight", "normal", "overweight", "obese"],
    "sex": ["male", "female"],
    "diet": ["vegan", "vegetarian", "pescatarian", "omnivore"],
    "how_often_shower": ["less frequently", "daily", "twice a day", "more frequently"],
    "heating_energy_source": ["coal", "natural gas", "wood", "electricity"],
    "transport": ["public", "walk/bicycle", "private"],
    "vehicle_type": ["none", "petrol", "diesel", "hybrid", "electric", "lpg"],
    "social_activity": ["never", "sometimes", "often"],
    "air_travel_frequency": ["never", "rarely", "frequently", "very frequently"],
    "waste_bag_size": ["small", "medium", "large", "extra large"],
    "energy_efficiency": ["no", "sometimes", "yes"],
    "recycling": ["none", "paper", "glass", "metal", "plastic"],
    "cooking_with": ["stove", "oven", "microwave", "grill", "airfryer"],
}

# Numeric field validation: (min_value, max_value)
# grocery bill is in INR (converted to USD internally before prediction)
NUMERIC_LIMITS = {
    # INR (approx Rs 0 – Rs 50,000/month)
    "monthly_grocery_bill":        (0,  50000),
    "vehicle_monthly_distance_km": (0,   5000),  # km
    "waste_bag_weekly_count":      (0,     20),  # bags
    "how_long_tv_pc_daily_hour":   (0,     24),  # hours
    "how_many_new_clothes_monthly": (0,     50),  # items
    "how_long_internet_daily_hour": (0,     24),  # hours
}

# ---------------------------------------------------------------------------
# Tips dictionary — India-specific, one tip per top contributing factor
# Key must match the cleaned column name from the dataset.
# ---------------------------------------------------------------------------
TIPS = {
    "transport": (
        "Public transport, walking, or cycling usually produces far less CO₂ per trip "
        "than driving alone. Replacing even a few trips a week with the metro or a bus "
        "lowers your transport emissions."
    ),
    "vehicle_type": (
        "Fuel type has a large effect on emissions per km. If you drive a petrol or diesel "
        "vehicle, a hybrid, CNG, or electric one would reduce it. If you already use a "
        "low-emission vehicle, the next step is driving less by combining trips."
    ),
    "vehicle_monthly_distance_km": (
        "Combining errands into one trip, carpooling, or working from home on some days "
        "can reduce the distance you drive each month."
    ),
    "diet": (
        "Food choices affect your footprint. Meals built around lentils, vegetables, and "
        "grains usually have lower emissions than meals with red meat, so shifting a few "
        "meals a week in that direction helps."
    ),
    "monthly_grocery_bill": (
        "Buying local, seasonal produce and avoiding food waste can lower the emissions "
        "linked to your groceries."
    ),
    "air_travel_frequency": (
        "For shorter journeys, a train is usually a lower-emission option than a flight. "
        "Taking fewer, longer trips also helps."
    ),
    "how_many_new_clothes_monthly": (
        "Buying fewer new clothes, choosing durable pieces, and using second-hand or "
        "local handloom options can reduce emissions from clothing."
    ),
    "waste_bag_weekly_count": (
        "Separating dry and wet waste and composting kitchen scraps can reduce the "
        "amount of waste sent to landfill."
    ),
    "waste_bag_size": (
        "Using reusable bags, bottles, and containers instead of single-use plastic "
        "helps reduce the waste you produce."
    ),
    "heating_energy_source": (
        "The energy source you use for heating and cooking matters. Coal and wood "
        "generally have higher emissions than gas or electricity, and an efficient "
        "appliance helps whichever source you use."
    ),
    "how_long_tv_pc_daily_hour": (
        "Switching screens off when not in use and choosing energy-efficient models "
        "lowers the electricity you use, and the emissions that come with it."
    ),
    "how_long_internet_daily_hour": (
        "Streaming and always-on devices use energy. Downloading videos for offline "
        "viewing and switching off your router at night can help a little."
    ),
    "energy_efficiency": (
        "Look for the BEE star rating when you buy appliances. A higher rating "
        "generally means lower electricity use."
    ),
    "how_often_shower": (
        "Shorter showers or bucket baths use less water, which means less energy "
        "spent heating it."
    ),
    "social_activity": (
        "Choosing nearby venues you can walk or cycle to reduces the transport "
        "emissions of going out."
    ),
    "recycling": (
        "Separating dry waste such as paper, plastic, glass, and metal and handing it "
        "to a local recycler or kabadiwala keeps it out of landfill."
    ),
    "cooking_with": (
        "Cooking methods differ in energy use. Using a lid, a pressure cooker, or an "
        "induction cooktop can reduce the energy needed for cooking."
    ),
    "body_type": (
        "Walking or cycling for short errands can replace vehicle trips and cut your "
        "transport emissions."
    ),
    "sex": (
        "Lifestyle choices matter more than demographics. Transport, diet, and energy "
        "use are usually the best places to start."
    ),
}

# Default tip if the top factor isn't in TIPS
DEFAULT_TIP = (
    "Small daily changes add up. Try public transport, more plant-based meals, "
    "local produce, and energy-efficient appliances."
)
