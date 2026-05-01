import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['HADOOP_HOME'] = 'C:/Users/PRANAV/hadoop'

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, log1p, round as spark_round, 
    percentile_approx, mean, stddev
)

from pyspark.sql.types import DoubleType, IntegerType
import pandas as pd

def create_spark_session():
    return SparkSession.builder \
        .appName("ChurnFeatureEngineering") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def engineer_features(spark, input_path = "data/customers.csv"):
    print("Loading customer data with Spark...")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    print(f"Loaded {df.count()} records with {len(df.columns)} columns")

    print("Engineering features...")

    # 1. Revenue features
    df = df.withColumn("avg_revenue_per_tenure",
        spark_round(col("total_charges") / (col("tenure") + 1), 2))

    df = df.withColumn("charge_to_tenure_ratio",
        spark_round(col("monthly_charges") / (col("tenure") + 1), 2))

    # 2. Engagement features
    df = df.withColumn("is_long_tenure",
        when(col("tenure") >= 24, 1).otherwise(0))

    df = df.withColumn("is_high_spender",
        when(col("monthly_charges") >= 80, 1).otherwise(0))

    df = df.withColumn("is_inactive",
        when(col("days_since_last_interaction") >= 180, 1).otherwise(0))

    # 3. Risk features
    df = df.withColumn("high_support_tickets",
        when(col("num_support_tickets") >= 5, 1).otherwise(0))

    df = df.withColumn("low_usage",
        when(col("avg_monthly_usage_gb") <= 10, 1).otherwise(0))

    # 4. Contract risk score
    df = df.withColumn("contract_risk",
        when(col("contract") == "Month-to-month", 3)
        .when(col("contract") == "One year", 2)
        .otherwise(1))

    # 5. Log transform skewed features
    df = df.withColumn("log_total_charges", 
        spark_round(log1p(col("total_charges")), 4))
    df = df.withColumn("log_monthly_charges",
        spark_round(log1p(col("monthly_charges")), 4))

    # 6. Encode categorical features
    df = df.withColumn("is_fiber",
        when(col("internet_service") == "Fiber optic", 1).otherwise(0))
    df = df.withColumn("has_support",
        when(col("tech_support") == "Yes", 1).otherwise(0))
    df = df.withColumn("is_month_to_month",
        when(col("contract") == "Month-to-month", 1).otherwise(0))
    df = df.withColumn("uses_electronic_check",
        when(col("payment_method") == "Electronic check", 1).otherwise(0))

    # 7. Composite churn risk score (0-10)
    df = df.withColumn("churn_risk_score",
        (col("contract_risk") * 2 +
         col("is_fiber") +
         col("high_support_tickets") +
         col("is_inactive") +
         col("is_high_spender") +
         (1 - col("is_long_tenure"))))

    print(f"Feature engineering complete. Total features: {len(df.columns)}")

    # Show sample
    print("\nSample engineered features:")
    df.select("customer_id", "tenure", "monthly_charges",
              "churn_risk_score", "is_month_to_month",
              "is_fiber", "churn").show(5)

    # Save features
    os.makedirs("data", exist_ok=True)
    output_path = "data/features.parquet"
    df.toPandas().to_parquet(output_path, index=False)
    print(f"\nFeatures saved to {output_path}")

    # Show churn distribution by risk score
    print("\nChurn rate by risk score:")
    df.groupBy("churn_risk_score") \
      .agg(
          mean("churn").alias("churn_rate"),
          spark_round(mean("churn") * 100, 2).alias("churn_pct")
      ) \
      .orderBy("churn_risk_score") \
      .show()

    return df

if __name__ == "__main__":
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    df = engineer_features(spark)
    print("Feature engineering completed successfully!")
    spark.stop()
    