# monitoring/monitor.py
import pandas as pd
import numpy as np
from scipy import stats
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDriftMonitor:
    """Detects when incoming prediction data drifts from training distribution"""
    
    def __init__(self, reference_data_path: str = "data/raw/Crop_recommendation.csv"):
        ref_df = pd.read_csv(reference_data_path)
        self.feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        
        # Store reference statistics
        self.reference_stats = {
            col: {
                "mean": ref_df[col].mean(),
                "std": ref_df[col].std(),
                "min": ref_df[col].min(),
                "max": ref_df[col].max(),
                "q25": ref_df[col].quantile(0.25),
                "q75": ref_df[col].quantile(0.75),
            }
            for col in self.feature_cols
        }
        self.reference_data = ref_df[self.feature_cols]
    
    def check_drift(self, recent_predictions: pd.DataFrame, p_threshold: float = 0.05) -> dict:
        """
        KS test to detect distribution shift.
        Returns drift report with alerts.
        """
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(recent_predictions),
            "features": {},
            "drift_detected": False,
            "drifted_features": []
        }
        
        for col in self.feature_cols:
            if col not in recent_predictions.columns:
                continue
            
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[col].values,
                recent_predictions[col].values
            )
            
            is_drift = p_value < p_threshold
            
            drift_report["features"][col] = {
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 4),
                "drift_detected": is_drift,
                "current_mean": round(recent_predictions[col].mean(), 3),
                "reference_mean": round(self.reference_stats[col]["mean"], 3),
            }
            
            if is_drift:
                drift_report["drift_detected"] = True
                drift_report["drifted_features"].append(col)
                logger.warning(f"⚠️ DRIFT DETECTED in '{col}': p={p_value:.4f}")
        
        # Save report
        report_path = Path("monitoring/reports")
        report_path.mkdir(parents=True, exist_ok=True)
        with open(report_path / f"drift_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "w") as f:
            json.dump(drift_report, f, indent=2)
        
        return drift_report
    
    def check_prediction_distribution(self, predictions: list) -> dict:
        """Check if prediction class distribution has shifted"""
        from collections import Counter
        counts = Counter(predictions)
        total = len(predictions)
        
        distribution = {crop: count / total for crop, count in counts.items()}
        
        # Alert if any single class dominates > 60% (possible model degradation)
        max_share = max(distribution.values()) if distribution else 0
        if max_share > 0.6:
            logger.warning(f"⚠️ Prediction imbalance: {max(counts, key=counts.get)} = {max_share:.1%}")
        
        return distribution


class PredictionLogger:
    """Logs every prediction for monitoring and future retraining"""
    
    def __init__(self, log_path: str = "monitoring/prediction_logs.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with headers if it doesn't exist
        if not self.log_path.exists():
            pd.DataFrame(columns=[
                "timestamp", "N", "P", "K", "temperature",
                "humidity", "ph", "rainfall", "prediction", "confidence"
            ]).to_csv(self.log_path, index=False)
    
    def log(self, inputs: dict, prediction: str, confidence: float):
        row = {
            "timestamp": datetime.now().isoformat(),
            **inputs,
            "prediction": prediction,
            "confidence": round(confidence, 2)
        }
        pd.DataFrame([row]).to_csv(self.log_path, mode="a", header=False, index=False)