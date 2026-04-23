# pipelines/retrain.py
"""
Run this weekly via cron or GitHub Actions scheduled trigger.
Checks if retraining is needed and runs it if so.
"""
import pandas as pd
import json
from pathlib import Path
from monitoring.monitor import DataDriftMonitor
from model import train_crop_recommendation_model

DRIFT_REPORT_DIR = Path("monitoring/reports")
LOG_FILE = Path("monitoring/prediction_logs.csv")
MIN_NEW_SAMPLES = 200   # retrain only if enough new data

def should_retrain() -> bool:
    """Decide if retraining is needed"""
    
    # Check 1: Drift detected in recent reports
    reports = sorted(DRIFT_REPORT_DIR.glob("drift_*.json"))
    if reports:
        with open(reports[-1]) as f:
            latest_report = json.load(f)
        if latest_report.get("drift_detected"):
            print("🔁 Retraining triggered: data drift detected")
            return True
    
    # Check 2: Enough new labeled data accumulated
    if LOG_FILE.exists():
        logs = pd.read_csv(LOG_FILE)
        # In a real system you'd check if labels were provided for these
        if len(logs) >= MIN_NEW_SAMPLES:
            print(f"🔁 Retraining triggered: {len(logs)} new samples accumulated")
            return True
    
    print("✅ No retraining needed")
    return False

def run_retraining():
    """Full retraining with evaluation"""
    print("Starting retraining pipeline...")
    
    # Step 1: Validate new data
    from tests.validate_data import validate_crop_data
    validate_crop_data()
    
    # Step 2: Retrain
    model, minmax, std_scaler, labels = train_crop_recommendation_model()
    
    # Step 3: Evaluate and check gate
    from tests.check_performance_gate import check_gate
    check_gate()
    
    print("✅ Retraining complete. New model ready for promotion.")

if __name__ == "__main__":
    if should_retrain():
        run_retraining()