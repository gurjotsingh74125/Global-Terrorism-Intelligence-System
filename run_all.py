import os
import json
import time
from src.data_preprocessing import load_and_preprocess_gtd
from src.model_perpetrator_attribution import train_perpetrator_attribution
from src.model_success_prediction import train_success_prediction
from src.model_casualty_estimation import train_casualty_estimation
from src.model_spatiotemporal import train_spatiotemporal

def main():
    print("=================================================================")
    print("   GLOBAL TERRORISM DATABASE (GTD) MACHINE LEARNING PIPELINE     ")
    print("=================================================================")
    start_time = time.time()

    # Step 1: Preprocessing & Data Cleaning
    print("\n--- Step 1: Preprocessing & Feature Engineering ---")
    if not (os.path.exists('data/train_gtd.pkl') and os.path.exists('data/test_gtd.pkl')):
        load_and_preprocess_gtd()
    else:
        print("Processed dataset found in data/ directory. Skipping preprocessing step.")

    # Step 2: Perpetrator Attribution Model
    print("\n--- Step 2: Perpetrator Attribution Model ---")
    _, attribution_metrics = train_perpetrator_attribution()

    # Step 3: Attack Success Prediction Model
    print("\n--- Step 3: Attack Success Prediction Model ---")
    _, success_metrics = train_success_prediction()

    # Step 4: Casualty Estimation Model
    print("\n--- Step 4: Casualty Severity & Quantile Model ---")
    _, casualty_metrics = train_casualty_estimation()

    # Step 5: Spatiotemporal Model
    print("\n--- Step 5: Spatiotemporal Clustering & Forecasting ---")
    spatiotemporal_metrics = train_spatiotemporal()

    # Save Metrics JSON
    os.makedirs('outputs', exist_ok=True)
    all_metrics = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'execution_time_seconds': round(time.time() - start_time, 2),
        'perpetrator_attribution': attribution_metrics,
        'attack_success_prediction': success_metrics,
        'casualty_risk_estimation': casualty_metrics,
        'spatiotemporal_forecasting': spatiotemporal_metrics
    }

    metrics_file = 'outputs/metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print("\n=================================================================")
    print(f" PIPELINE EXECUTION COMPLETED IN {all_metrics['execution_time_seconds']}s")
    print(f" Saved consolidated metrics to: {metrics_file}")
    print(" Saved model checkpoints to: models/")
    print("=================================================================")

if __name__ == '__main__':
    main()
