# tests/test_model.py
import pytest
import numpy as np
from model import predict_crop, train_crop_recommendation_model

class TestModelAccuracy:
    """Test that model meets minimum performance gates"""
    
    def test_model_accuracy_above_threshold(self):
        """Model must achieve > 90% accuracy"""
        import pandas as pd
        from sklearn.metrics import accuracy_score
        import joblib
        
        df = pd.read_csv("Crop_recommendation.csv")
        X = df.drop("label", axis=1)
        y = df["label"]
        
        model = joblib.load("crop_recommendation_model.pkl")
        minmax = joblib.load("minmax_scaler.pkl")
        std_scaler = joblib.load("standard_scaler.pkl")
        
        X_scaled = std_scaler.transform(minmax.transform(X))
        predictions = model.predict(X_scaled)
        accuracy = accuracy_score(y, predictions)
        
        assert accuracy >= 0.90, f"Accuracy {accuracy:.3f} below threshold 0.90"
    
    def test_prediction_returns_valid_crop(self):
        """Prediction must return a valid crop name"""
        valid_crops = [
            "rice", "maize", "chickpea", "banana", "mango", "grapes",
            "watermelon", "apple", "orange", "cotton", "jute", "coffee"
        ]
        crop, confidence = predict_crop(90, 42, 43, 20.87, 82.00, 6.5, 202.94)
        assert crop.lower() in valid_crops
        assert 0 <= confidence <= 100
    
    def test_confidence_is_reasonable(self):
        """Confidence should be above 50% for valid inputs"""
        crop, confidence = predict_crop(90, 42, 43, 20.87, 82.00, 6.5, 202.94)
        assert confidence >= 50, f"Low confidence: {confidence}"
    
    def test_rice_conditions_predict_rice(self):
        """Classic rice conditions should predict rice"""
        crop, _ = predict_crop(
            N=90, P=42, K=43,
            temperature=21.0, humidity=82.0,
            ph=6.5, rainfall=220.0
        )
        assert crop.lower() == "rice"
    
    def test_prediction_handles_edge_values(self):
        """Model should not crash on boundary values"""
        crop, confidence = predict_crop(0, 5, 5, 8.8, 14.3, 3.5, 20.2)
        assert crop is not None
        assert confidence >= 0
    
    @pytest.mark.parametrize("N,P,K,temp,hum,ph,rain", [
        (90, 42, 43, 20.87, 82.0, 6.5, 202.94),   # rice
        (20, 30, 10, 26.5, 52.5, 7.0, 150.25),    # mango-ish
        (40, 60, 35, 8.9, 16.0, 7.5, 75.0),       # chickpea-ish
    ])
    def test_batch_predictions(self, N, P, K, temp, hum, ph, rain):
        crop, confidence = predict_crop(N, P, K, temp, hum, ph, rain)
        assert crop is not None
        assert isinstance(confidence, float)