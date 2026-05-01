import pickle
import os
import mlflow
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow.tensorflow
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print(f"Tensorflow version: {tf.__version__}")

# Load engineered features
print("Loading features...")
df = pd.read_parquet("data/features.parquet")
print(f"Loaded {len(df)} records with {len(df.columns)} columns")

#select features for training
FEATURE_COLS = [
    'tenure', 'monthly_charges', 'total_charges',
    'num_support_tickets', 'avg_monthly_usage_gb',
    'days_since_last_interaction', 'senior_citizen',
    'avg_revenue_per_tenure', 'charge_to_tenure_ratio',
    'is_long_tenure', 'is_high_spender', 'is_inactive',
    'high_support_tickets', 'low_usage', 'contract_risk',
    'log_total_charges', 'log_monthly_charges',
    'is_fiber', 'has_support', 'is_month_to_month',
    'uses_electronic_check', 'churn_risk_score'
]

TARGET_COL = 'churn'

X = df[FEATURE_COLS].fillna(0)
Y = df[TARGET_COL]

# Train/Test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"Training: {len(x_train)} | Test: {len(x_test)}")
print(f"Churn rate: {Y.mean():.2%}")

# MLflow experiment
mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="tensorflow-deep-learning"):

    # Build Tensorflow model
    model = keras.Sequential([
        keras.layers.Input(shape=(len(FEATURE_COLS),)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    model.summary()

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=10,
        restore_best_weights=True, mode='max'
    )
    
    class_weight = {0: 1.0, 1: 3.0}

    # Train
    history = model.fit(
        x_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=1
    )

    # Evaluate
    y_prob = model.predict(x_test_scaled).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred,
                                   target_names=['Stay', 'Churn'],
                                   output_dict=True)

    precision = report['Churn']['precision']
    recall = report['Churn']['recall']
    f1 = report['Churn']['f1-score']
    accuracy = report['accuracy']

    # Log to MLflow
    mlflow.log_params({
        "model_type": "TensorFlow Deep Learning",
        "layers": "64-32-16-1",
        "dropout": "0.3-0.2",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "epochs": len(history.history['loss']),
        "batch_size": 32,
        "features": len(FEATURE_COLS)
    })

    mlflow.log_metrics({
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4)
    })

    print("\n=== Model Performance ===")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Stay', 'Churn']))

# Save model and scaler
os.makedirs("models", exist_ok=True)
model.save("models/churn_model.keras")
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print("\nModel saved to models/churn_model.keras")
print("Run 'mlflow ui' to see experiment tracking!")