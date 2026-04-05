import asyncio
import pandas as pd
import numpy as np
import os
from app.ml.trainer import train_model, UPLOADED_CSV_PATH
from app.ml.fairness import compute_fairness_report

def main():
    # Create an artificial REAL dataset
    np.random.seed(111)
    df = pd.DataFrame({
        "sgpa": np.random.uniform(5.0, 10.0, 500),
        "jee_score": np.random.randint(50, 360, 500),
        "marks_12": np.random.uniform(50.0, 100.0, 500),
        "attendance": np.random.uniform(50.0, 100.0, 500),
        "gender": np.random.choice([0, 1], 500)
    })
    
    os.makedirs(os.path.dirname(UPLOADED_CSV_PATH), exist_ok=True)
    df.to_csv(UPLOADED_CSV_PATH, index=False)
    print("Dataset without labels created at", UPLOADED_CSV_PATH)
    
    metadata = train_model(n_samples=500)
    print("Test Accuracy:", metadata.get("accuracy"))
    
    report = compute_fairness_report()
    metrics = {m["name"]: m["value"] for m in report["metrics"]}
    print("Metrics for dataset without labels:", metrics)
    
    # Dataset 2
    np.random.seed(999)
    df2 = pd.DataFrame({
        "sgpa": np.random.uniform(4.0, 9.0, 500),
        "jee_score": np.random.randint(40, 200, 500),
        "marks_12": np.random.uniform(40.0, 90.0, 500),
        "attendance": np.random.uniform(40.0, 90.0, 500),
        "gender": np.random.choice([0, 1], 500)
    })
    df2.to_csv(UPLOADED_CSV_PATH, index=False)
    print("\nDataset 2 without labels created.")
    
    metadata = train_model(n_samples=500)
    print("Test Accuracy 2:", metadata.get("accuracy"))
    
    report = compute_fairness_report()
    metrics2 = {m["name"]: m["value"] for m in report["metrics"]}
    print("Metrics for dataset 2 without labels:", metrics2)

if __name__ == "__main__":
    main()
