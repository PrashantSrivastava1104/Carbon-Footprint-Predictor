"""
Smoke test for Flask app - run via: python smoke_test.py
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
app.config['TESTING'] = True
client = app.test_client()

VALID = {
    'body_type': 'normal', 'sex': 'male', 'diet': 'omnivore',
    'how_often_shower': 'daily', 'heating_energy_source': 'natural gas',
    'transport': 'private', 'vehicle_type': 'petrol',
    'social_activity': 'sometimes', 'monthly_grocery_bill': '300',
    'air_travel_frequency': 'rarely', 'vehicle_monthly_distance_km': '400',
    'waste_bag_size': 'medium', 'waste_bag_weekly_count': '3',
    'how_long_tv_pc_daily_hour': '4', 'how_many_new_clothes_monthly': '2',
    'how_long_internet_daily_hour': '5', 'energy_efficiency': 'sometimes',
    'recycling': ['paper', 'plastic'],
    'cooking_with': ['stove'],
}

results = []

def chk(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

print("=" * 55)
print("FLASK SMOKE TESTS")
print("=" * 55)

# Test 1: Home page
r = client.get('/')
chk("GET / returns 200", r.status_code == 200, f"got {r.status_code}")
chk("GET / has form tag", b'form' in r.data)

# Test 2: Valid predict
r = client.post('/predict', data=VALID, follow_redirects=True)
chk("POST /predict valid -> 200", r.status_code == 200, f"got {r.status_code}")
body = r.data.decode('utf-8', errors='replace')
m = re.search(r'score-number[^>]*>(\d+\.?\d*)', body)
chk("Prediction value in result", m is not None, m.group(0) if m else "NOT FOUND")
has_label = any(x in body for x in ['Low', 'Medium', 'High'])
chk("Label in result page", has_label)

# Test 3: Invalid numeric
bad = dict(VALID); bad['monthly_grocery_bill'] = 'not_a_number'
r = client.post('/predict', data=bad)
chk("Invalid number -> 400", r.status_code == 400, f"got {r.status_code}")

# Test 4: Out-of-range
bad2 = dict(VALID); bad2['monthly_grocery_bill'] = '99999'
r = client.post('/predict', data=bad2)
chk("Out-of-range -> 400", r.status_code == 400, f"got {r.status_code}")

# Test 5: Missing field
bad3 = dict(VALID); del bad3['diet']
r = client.post('/predict', data=bad3)
chk("Missing field -> 400", r.status_code == 400, f"got {r.status_code}")

# Test 6: About page
r = client.get('/about')
chk("GET /about -> 200", r.status_code == 200, f"got {r.status_code}")
chk("/about has XGBoost", b'XGBoost' in r.data)

# Test 7: Health check
r = client.get('/health')
chk("GET /health -> 200", r.status_code == 200, f"got {r.status_code}")
data = json.loads(r.data)
chk("/health status=ok", data.get('status') == 'ok', str(data))
chk("/health model_loaded=True", data.get('model_loaded') is True, str(data))

print("=" * 55)
passes = sum(1 for s, _, _ in results if s == "PASS")
total  = len(results)
print(f"RESULT: {passes}/{total} tests passed")
if passes < total:
    print("\nFailed tests:")
    for s, name, detail in results:
        if s == "FAIL":
            print(f"  - {name}: {detail}")
    sys.exit(1)
else:
    print("All tests passed!")
