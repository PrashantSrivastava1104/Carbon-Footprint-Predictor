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

MODEL_PATH       = os.path.join(BASE_DIR, "models", "best_pipeline.joblib")
THRESHOLDS_PATH  = os.path.join(BASE_DIR, "models", "label_thresholds.json")
METRICS_CSV_PATH = os.path.join(BASE_DIR, "models", "metrics", "model_comparison.csv")
METRICS_MD_PATH  = os.path.join(BASE_DIR, "models", "metrics", "model_comparison.md")
CHART_PATH       = os.path.join(BASE_DIR, "static", "charts", "feature_importance.png")
RAW_DATA_PATH    = os.path.join(BASE_DIR, "data", "raw", "Carbon Emission.csv")
PROCESSED_PATH   = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# ---------------------------------------------------------------------------
# Flask settings
# ---------------------------------------------------------------------------
SECRET_KEY   = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
DEBUG        = os.environ.get("FLASK_DEBUG", "1") == "1"
PORT         = int(os.environ.get("PORT", 5000))

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
    "monthly_grocery_bill":        (0,  50000),  # INR (approx Rs 0 – Rs 50,000/month)
    "vehicle_monthly_distance_km": (0,   5000),  # km
    "waste_bag_weekly_count":      (0,     20),  # bags
    "how_long_tv_pc_daily_hour":   (0,     24),  # hours
    "how_many_new_clothes_monthly":(0,     50),  # items
    "how_long_internet_daily_hour":(0,     24),  # hours
}

# ---------------------------------------------------------------------------
# Tips dictionary — India-specific, one tip per top contributing factor
# Key must match the cleaned column name from the dataset.
# ---------------------------------------------------------------------------
TIPS = {
    "transport": (
        "🚌 Use the metro, BEST bus, or city bus instead of driving alone. "
        "Indian Railways and metro systems emit up to 80% less CO₂ per km "
        "than a solo petrol car. Switching even 3 days a week makes a big difference."
    ),
    "vehicle_type": (
        "🔋 Consider switching to a CNG vehicle or an electric two-wheeler. "
        "India's EV adoption is growing fast — Ola Electric, Ather, and TVS "
        "offer affordable options. EVs on India's grid emit ~40% less than petrol."
    ),
    "vehicle_monthly_distance_km": (
        "🗺️ Combine errands into one trip and use carpooling apps like Quick Ride. "
        "Every 100 km you skip saves roughly 17 kg of CO₂. "
        "Working from home even 2 days a week can cut your monthly distance by 40%."
    ),
    "diet": (
        "🥦 India's vegetarian cuisine is naturally low-carbon — embrace it more! "
        "Dal, sabzi, and rice emit a fraction of the CO₂ of red meat. "
        "Even replacing one non-veg meal per day can cut your food footprint by 20%."
    ),
    "monthly_grocery_bill": (
        "🛒 Shop at your local sabzi mandi or kirana store instead of large supermarkets. "
        "Locally grown seasonal produce travels far fewer kilometres than imported goods, "
        "and reduces cold-chain refrigeration emissions significantly."
    ),
    "air_travel_frequency": (
        "✈️ Take the train instead of flying for journeys under 500 km. "
        "Indian Railways emits ~90% less CO₂ per passenger-km than domestic flights. "
        "A Mumbai–Pune or Delhi–Agra trip is faster by train anyway."
    ),
    "how_many_new_clothes_monthly": (
        "👕 Buy from local weavers, thrift stores, or platforms like ThriftKart "
        "instead of fast fashion brands. Indian handloom fabric has one of the "
        "lowest carbon footprints of any textile — and supports local artisans."
    ),
    "waste_bag_weekly_count": (
        "♻️ Separate dry and wet waste at home — most Indian cities have door-to-door "
        "waste collection for both. Compost your kitchen scraps (peels, leftovers) "
        "in a small bin; this can divert up to 50% of household waste from landfill."
    ),
    "waste_bag_size": (
        "🗑️ Carry a cloth bag (jhola) for shopping and refuse single-use plastic. "
        "India banned many single-use plastics in 2022 — support this by using "
        "steel dabbas and reusable bottles instead of disposable packaging."
    ),
    "heating_energy_source": (
        "🏠 If you use a room heater, switch to an energy-efficient BEE 5-star rated model. "
        "In warmer states, good cross-ventilation and ceiling fans are enough — "
        "they use 10x less energy than air conditioning."
    ),
    "how_long_tv_pc_daily_hour": (
        "💡 Switch to LED TVs and BEE 5-star rated monitors — they use up to 60% "
        "less electricity than older screens. India's grid still relies on coal, "
        "so every unit of electricity you save prevents real CO₂ emissions."
    ),
    "how_long_internet_daily_hour": (
        "🌐 Streaming video consumes significant data-centre energy. "
        "Download content for offline viewing on trains or commutes instead of streaming. "
        "Also, switch off your router at night — it uses electricity even when idle."
    ),
    "energy_efficiency": (
        "⚡ Look for the BEE Star Rating label when buying appliances in India. "
        "A 5-star AC or refrigerator uses up to 40% less electricity than a 1-star model. "
        "This saves both money on your electricity bill and CO₂ from the grid."
    ),
    "how_often_shower": (
        "🚿 Take bucket baths instead of showers — the average Indian bucket bath "
        "uses just 15–20 litres vs 60+ litres for a 10-minute shower. "
        "Heating less water saves both LPG/electricity and reduces your footprint."
    ),
    "social_activity": (
        "🎉 Choose local parks, community centres, or neighbourhood venues for outings. "
        "Walking or cycling to social events is very doable in most Indian neighbourhoods "
        "and completely eliminates transport emissions for that trip."
    ),
    "recycling": (
        "♻️ Hand segregated dry waste (paper, plastic, glass, metal) to your local "
        "kabadiwala — India's informal recycling network is one of the most efficient "
        "in the world and diverts millions of tonnes from landfill every year."
    ),
    "cooking_with": (
        "🍳 If you use LPG, consider switching to an induction cooktop for some meals. "
        "Induction is 85% efficient vs ~40% for gas burners. Solar cookers are also "
        "ideal for Indian sunny climates and have zero running cost."
    ),
    "body_type": (
        "🏃 Walk or cycle for short errands within your neighbourhood — most Indian "
        "streets have everything within 1–2 km. It improves health and cuts "
        "vehicle emissions at the same time."
    ),
    "sex": (
        "🌍 Lifestyle choices matter more than demographics. "
        "Focus on transport, diet, and energy use for the biggest impact on your footprint."
    ),
}

# Default tip if the top factor isn't in TIPS
DEFAULT_TIP = (
    "🌱 Small daily changes add up. Try eating more vegetarian meals, "
    "using public transport, buying from local markets, and switching to "
    "BEE 5-star rated appliances."
)
