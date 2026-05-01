import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time
import uuid

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Load model and artifacts
model = tf.keras.models.load_model("models/churn_model.keras")
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/feature_cols.pkl", "rb") as f:
    FEATURE_COLS = pickle.load(f)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("churn_predictions_total", "Total predictions", ["result"])
PREDICTION_LATENCY = Histogram("churn_prediction_latency_seconds", "Prediction latency")
HIGH_RISK_CUSTOMERS = Counter("high_risk_customers_total", "High risk customers detected")

class Customer(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    num_support_tickets: int
    avg_monthly_usage_gb: float
    days_since_last_interaction: int
    senior_citizen: int
    contract: str
    internet_service: str
    tech_support: str
    payment_method: str

class ChurnResponse(BaseModel):
    customer_id: str
    will_churn: bool
    churn_probability: float
    risk_level: str
    retention_action: str

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=ChurnResponse)
def predict(customer: Customer):
    start_time = time.time()

    # Engineer features
    is_fiber = 1 if customer.internet_service == "Fiber optic" else 0
    has_support = 1 if customer.tech_support == "Yes" else 0
    is_month_to_month = 1 if customer.contract == "Month-to-month" else 0
    uses_electronic_check = 1 if customer.payment_method == "Electronic check" else 0
    contract_risk = 3 if customer.contract == "Month-to-month" else 2 if customer.contract == "One year" else 1

    avg_revenue_per_tenure = round(customer.total_charges / (customer.tenure + 1), 2)
    charge_to_tenure_ratio = round(customer.monthly_charges / (customer.tenure + 1), 2)
    is_long_tenure = 1 if customer.tenure >= 24 else 0
    is_high_spender = 1 if customer.monthly_charges >= 80 else 0
    is_inactive = 1 if customer.days_since_last_interaction >= 180 else 0
    high_support_tickets = 1 if customer.num_support_tickets >= 5 else 0
    low_usage = 1 if customer.avg_monthly_usage_gb <= 10 else 0

    import math
    log_total_charges = round(math.log1p(customer.total_charges), 4)
    log_monthly_charges = round(math.log1p(customer.monthly_charges), 4)

    churn_risk_score = (contract_risk * 2 + is_fiber + high_support_tickets +
                       is_inactive + is_high_spender + (1 - is_long_tenure))

    features = np.array([[
        customer.tenure, customer.monthly_charges, customer.total_charges,
        customer.num_support_tickets, customer.avg_monthly_usage_gb,
        customer.days_since_last_interaction, customer.senior_citizen,
        avg_revenue_per_tenure, charge_to_tenure_ratio,
        is_long_tenure, is_high_spender, is_inactive,
        high_support_tickets, low_usage, contract_risk,
        log_total_charges, log_monthly_charges,
        is_fiber, has_support, is_month_to_month,
        uses_electronic_check, churn_risk_score
    ]])

    features_scaled = scaler.transform(features)
    prob = float(model.predict(features_scaled, verbose=0)[0][0])
    will_churn = prob > 0.5

    if prob < 0.3:
        risk_level = "LOW"
        retention_action = "No action needed"
    elif prob < 0.6:
        risk_level = "MEDIUM"
        retention_action = "Send loyalty discount offer"
    else:
        risk_level = "HIGH"
        retention_action = "Immediate outreach - assign retention specialist"

    PREDICTIONS_TOTAL.labels(result="churn" if will_churn else "stay").inc()
    PREDICTION_LATENCY.observe(time.time() - start_time)
    if risk_level == "HIGH":
        HIGH_RISK_CUSTOMERS.inc()

    return ChurnResponse(
        customer_id=str(uuid.uuid4()),
        will_churn=will_churn,
        churn_probability=round(prob, 4),
        risk_level=risk_level,
        retention_action=retention_action
    )

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()