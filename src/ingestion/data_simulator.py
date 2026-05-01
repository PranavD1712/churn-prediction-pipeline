import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
 

def generate_churn_dataset(n_customers=5000):
    print(f"Generating {n_customers} customer records..!")
    np.random.seed(42)

    customers = []
    for i in range(n_customers):
        tenure = random.randint(1, 72)
        monthly_charges = round(random.uniform(20, 120), 2)

        #churn pattern: short tenure + high charges + no contract
        contract = random.choice(['Month-to-Month', 'One year', 'Two year'])
        internet = random.choice(['DSL', 'Fiber optic', 'No'])
        support = random.choice(['Yes', 'No'])

        churn_prob = 0.05
        if contract == 'Month-to-Month':
            churn_prob += 0.20
        if internet == 'Fiber optic':
            churn_prob += 0.10
        if tenure < 12:
            churn_prob += 0.15
        if monthly_charges > 80:
            churn_prob += 0.10
        if support == 'No':
            churn_prob += 0.05

        is_churn = 1 if random.random() < churn_prob else 0

        customers.append({
            'customer_id': f'CUST_{i+1:05d}',
            'tenure': tenure,
            'monthly_charges': monthly_charges,
            'total_charges': round(monthly_charges * tenure, 2),
            'contract': contract,
            'internet_service': internet,
            'tech_support': support,
            'senior_citizen': random.choice([0, 1]),
            'partner': random.choice(['Yes', 'No']),
            'dependents': random.choice(['Yes', 'No']),
            'phone_service': random.choice(['Yes', 'No']),
            'paperless_billing': random.choice(['Yes', 'No']),
            'payment_method': random.choice([
                'Electronic check', 'Mailed check',
                'Bank transfer', 'Credit card'
                ]),
                'num_support_tickets': random.randint(0, 10),
            'avg_monthly_usage_gb': round(random.uniform(1, 100), 2),
            'days_since_last_interaction': random.randint(1, 365),
            'churn': is_churn
        })

    df = pd.DataFrame(customers)

    churn_rate = df['churn'].mean()
    print(f"Dataset Created: {len(df)} customers | Churn rate: {churn_rate:.2%}")

    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/customers.csv', index=False)
    print("Saved to data/customers.csv")
    return df

if __name__ == "__main__":
    df = generate_churn_dataset()
    print(df.head())
    print(df['churn'].value_counts())
 