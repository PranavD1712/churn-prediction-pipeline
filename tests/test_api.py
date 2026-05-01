import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.serving.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_high_risk():
    response = client.post("/predict", json={
        "tenure": 2,
        "monthly_charges": 95.0,
        "total_charges": 190.0,
        "num_support_tickets": 7,
        "avg_monthly_usage_gb": 5.0,
        "days_since_last_interaction": 200,
        "senior_citizen": 0,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No",
        "payment_method": "Electronic check"
    })
    assert response.status_code == 200
    assert response.json()["risk_level"] == "HIGH"
    assert response.json()["will_churn"] == True

def test_predict_low_risk():
    response = client.post("/predict", json={
        "tenure": 48,
        "monthly_charges": 35.0,
        "total_charges": 1680.0,
        "num_support_tickets": 1,
        "avg_monthly_usage_gb": 50.0,
        "days_since_last_interaction": 10,
        "senior_citizen": 0,
        "contract": "Two year",
        "internet_service": "DSL",
        "tech_support": "Yes",
        "payment_method": "Bank transfer"
    })
    assert response.status_code == 200
    assert response.json()["risk_level"] == "LOW"
    assert response.json()["will_churn"] == False