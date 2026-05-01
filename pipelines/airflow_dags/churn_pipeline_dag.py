from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess
import sys
import os

default_args = {
    'owner': 'pranav',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction pipeline',
    schedule='@daily',
    catchup=False,
    tags=['ml', 'churn', 'production']
)

def generate_data(**context):
    print("Step 1: Generating customer data...")
    exec_date = context['execution_date']
    print(f"Execution date: {exec_date}")
    import pandas as pd
    import numpy as np
    import random
    
    np.random.seed(int(exec_date.timestamp()) % 1000)
    n = 1000
    merchants = ['Month-to-month', 'One year', 'Two year']
    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(n)],
        'tenure': np.random.randint(1, 72, n),
        'monthly_charges': np.random.uniform(20, 120, n).round(2),
        'total_charges': np.random.uniform(100, 8000, n).round(2),
        'num_support_tickets': np.random.randint(0, 10, n),
        'avg_monthly_usage_gb': np.random.uniform(1, 100, n).round(2),
        'days_since_last_interaction': np.random.randint(1, 365, n),
        'senior_citizen': np.random.choice([0, 1], n),
        'contract': np.random.choice(merchants, n),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
        'tech_support': np.random.choice(['Yes', 'No'], n),
        'payment_method': np.random.choice([
            'Electronic check', 'Mailed check',
            'Bank transfer', 'Credit card'], n),
        'churn': np.random.choice([0, 1], n, p=[0.77, 0.23])
    }
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/daily_customers.csv', index=False)
    print(f"Generated {len(df)} customer records for {exec_date.date()}")
    return len(df)

def run_feature_engineering(**context):
    print("Step 2: Running Spark feature engineering...")
    import pandas as pd
    import numpy as np
    import math

    df = pd.read_csv('data/daily_customers.csv')
    df['avg_revenue_per_tenure'] = (df['total_charges'] / (df['tenure'] + 1)).round(2)
    df['charge_to_tenure_ratio'] = (df['monthly_charges'] / (df['tenure'] + 1)).round(2)
    df['is_long_tenure'] = (df['tenure'] >= 24).astype(int)
    df['is_high_spender'] = (df['monthly_charges'] >= 80).astype(int)
    df['is_inactive'] = (df['days_since_last_interaction'] >= 180).astype(int)
    df['high_support_tickets'] = (df['num_support_tickets'] >= 5).astype(int)
    df['low_usage'] = (df['avg_monthly_usage_gb'] <= 10).astype(int)
    df['contract_risk'] = df['contract'].map({
        'Month-to-month': 3, 'One year': 2, 'Two year': 1})
    df['log_total_charges'] = df['total_charges'].apply(lambda x: round(math.log1p(x), 4))
    df['log_monthly_charges'] = df['monthly_charges'].apply(lambda x: round(math.log1p(x), 4))
    df['is_fiber'] = (df['internet_service'] == 'Fiber optic').astype(int)
    df['has_support'] = (df['tech_support'] == 'Yes').astype(int)
    df['is_month_to_month'] = (df['contract'] == 'Month-to-month').astype(int)
    df['uses_electronic_check'] = (df['payment_method'] == 'Electronic check').astype(int)
    df['churn_risk_score'] = (df['contract_risk'] * 2 + df['is_fiber'] +
                              df['high_support_tickets'] + df['is_inactive'] +
                              df['is_high_spender'] + (1 - df['is_long_tenure']))

    df.to_parquet('data/daily_features.parquet', index=False)
    print(f"Feature engineering complete. {len(df.columns)} features created.")
    return len(df.columns)

def run_batch_predictions(**context):
    print("Step 3: Running batch churn predictions...")
    import pandas as pd
    import numpy as np
    import pickle
    import tensorflow as tf

    df = pd.read_parquet('data/daily_features.parquet')
    model = tf.keras.models.load_model('models/churn_model.keras')
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    X = df[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled, verbose=0).flatten()

    df['churn_probability'] = probs.round(4)
    df['predicted_churn'] = (probs > 0.5).astype(int)
    df['risk_level'] = pd.cut(probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH'])

    high_risk = df[df['risk_level'] == 'HIGH']
    print(f"Total customers: {len(df)}")
    print(f"Predicted churners: {df['predicted_churn'].sum()}")
    print(f"High risk customers: {len(high_risk)}")

    df.to_csv('data/predictions.csv', index=False)
    high_risk[['customer_id', 'churn_probability',
               'risk_level']].to_csv('data/high_risk_customers.csv', index=False)
    print("Predictions saved!")
    return int(df['predicted_churn'].sum())

def send_retention_alerts(**context):
    print("Step 4: Sending retention campaign alerts...")
    import pandas as pd

    high_risk = pd.read_csv('data/high_risk_customers.csv')
    print(f"Sending alerts for {len(high_risk)} high-risk customers...")
    for _, row in high_risk.head(5).iterrows():
        print(f"  ALERT: {row['customer_id']} | "
              f"Churn prob: {row['churn_probability']:.2%} | "
              f"Action: Immediate outreach")
    print(f"Retention campaign triggered for {len(high_risk)} customers!")
    return len(high_risk)

# Define tasks
t1 = PythonOperator(
    task_id='generate_daily_data',
    python_callable=generate_data,
    dag=dag
)

t2 = PythonOperator(
    task_id='feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag
)

t3 = PythonOperator(
    task_id='batch_predictions',
    python_callable=run_batch_predictions,
    dag=dag
)

t4 = PythonOperator(
    task_id='send_retention_alerts',
    python_callable=send_retention_alerts,
    dag=dag
)

# Pipeline order
t1 >> t2 >> t3 >> t4