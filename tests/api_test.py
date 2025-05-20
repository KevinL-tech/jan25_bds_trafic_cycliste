import requests

API_URL = "http://localhost:8000"

MODEL_INPUT_PARAMS = {
    "jour": 24,
    "mois": 12,
    "annee": 2025,
    "heure": 8,
    "jour_semaine": 2,
    "jour_ferie": False,
    "vacances_scolaires": True,
    "site_comptage": "Pont des Invalides"
}

def test_api_root():
    r = requests.get(f"{API_URL}/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Traffic Cycliste Prediction API"}

def test_predict():
    r = requests.post(f"{API_URL}/predict", json=MODEL_INPUT_PARAMS)
    assert r.status_code == 200
    assert r.json()["input_parameters"] == MODEL_INPUT_PARAMS
    assert r.json()["prediction"] # assert that the prediction value exists

def test_predict_missing_data():
    r = requests.post(f"{API_URL}/predict", json={})
    assert r.status_code == 422

def test_metrics():
    r = requests.get(f"{API_URL}/metrics")
    assert r.status_code == 200
    assert "inference_time_seconds_count" in r.text

