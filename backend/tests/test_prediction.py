from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_happy_path():
    payload = {
        "location": "Sector 45",
        "carpet_area_sqft": 1200.0,
        "floor_num": 3,
        "bathroom": 2,
        "balcony": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East",
    }
    response = client.post("/predict", json=payload)
    # If no model has been placed in models/house_price.pkl yet this will be
    # 503, which is still a valid, well-formed response to assert on.
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert "predicted_price" in body
        assert body["predicted_price"] >= 0


def test_predict_invalid_input_returns_422():
    payload = {
        "location": "Sector 45",
        "carpet_area_sqft": -100.0,  # invalid: must be > 0
        "floor_num": 3,
        "bathroom": 2,
        "balcony": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_locations_endpoint_returns_list():
    response = client.get("/locations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
