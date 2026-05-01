# 🔄 Customer Churn Prediction — MLOps End-to-End Pipeline

![CI/CD](https://github.com/PranavD1712/churn-prediction-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.5.1-orange)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-3.2-green)

A production-grade, end-to-end MLOps pipeline that predicts telecom customer churn 30 days ahead and automatically triggers retention campaigns — reducing churn by targeting high-risk customers before they leave.

---

## 🏗️ Business Problem

A telecom company loses revenue daily to customer churn. This system:
- Predicts which customers will churn in the next 30 days
- Scores each customer with a risk level (LOW / MEDIUM / HIGH)
- Automatically triggers personalized retention campaigns
- Runs daily via Apache Airflow DAG

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Data Generation | Python + Faker |
| Feature Engineering | Apache Spark (PySpark) |
| ML Model | TensorFlow Deep Learning |
| Experiment Tracking | MLflow |
| Pipeline Orchestration | Apache Airflow |
| Model Serving | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Language | Python 3.11 |

---

## 📁 Project Structure

```
churn-prediction-pipeline/
├── src/
│   ├── ingestion/
│   │   └── data_simulator.py
│   ├── processing/
│   │   └── feature_engineering.py
│   ├── training/
│   │   └── train_model.py
│   └── serving/
│       └── app.py
├── pipelines/
│   └── airflow_dags/
│       └── churn_pipeline_dag.py
├── tests/
│   └── test_api.py
├── models/
│   ├── churn_model.keras
│   ├── scaler.pkl
│   └── feature_cols.pkl
├── .github/
│   └── workflows/
│       └── ci.yml
├── requirements.txt
└── README.md
```

## 🔄 Pipeline Architecture
Daily Trigger (Airflow)
↓
Data Generation → Spark Feature Engineering → TF Model Prediction
↓
Risk Scoring (LOW/MEDIUM/HIGH)
↓
Retention Campaign Alerts
↓
FastAPI REST Endpoint
↓
Prometheus → Grafana Dashboard

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | 0.6508 |
| Accuracy | 0.6660 |
| Precision (Churn) | 0.3458 |
| Recall (Churn) | 0.4723 |
| F1 Score (Churn) | 0.3993 |

> Realistic performance for telecom churn — class imbalance handled with class weights (1:3 ratio)

---

## 🚀 Quick Start

### 1. Clone and setup
```bash
git clone https://github.com/PranavD1712/churn-prediction-pipeline.git
cd churn-prediction-pipeline
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2. Generate data and train model
```bash
python src/ingestion/data_simulator.py
python src/processing/feature_engineering.py
python src/training/train_model.py
```

### 3. Start the API
```bash
uvicorn src.serving.app:app --reload --port 8000
```

### 4. Test prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "monthly_charges": 95.0, "total_charges": 190.0,
       "num_support_tickets": 7, "avg_monthly_usage_gb": 5.0,
       "days_since_last_interaction": 200, "senior_citizen": 0,
       "contract": "Month-to-month", "internet_service": "Fiber optic",
       "tech_support": "No", "payment_method": "Electronic check"}'
```

### 5. Run Airflow pipeline
```bash
python -c "
from pipelines.airflow_dags.churn_pipeline_dag import *
from datetime import datetime
ctx = {'execution_date': datetime.now()}
generate_data(**ctx)
run_feature_engineering(**ctx)
run_batch_predictions(**ctx)
send_retention_alerts(**ctx)
"
```

---

## 🧪 Run Tests
```bash
pytest tests/ -v
```

---

## 🎯 Key Skills Demonstrated

- Deep learning with **TensorFlow** for tabular classification
- Feature engineering with **Apache Spark** (22+ engineered features)
- Pipeline orchestration with **Apache Airflow** DAGs
- MLOps with **MLflow** experiment tracking
- Production API with **FastAPI** + retention action logic
- Monitoring with **Prometheus + Grafana**
- CI/CD with **GitHub Actions** + pytest
- Class imbalance handling with weighted loss

---

## 👤 Author

**Pranav Deshmukh** — Data Science & ML
📧 Connect on [LinkedIn](https://www.linkedin.com/in/pranav-deshmukh2004)
⭐ Star this repo if you found it helpful!