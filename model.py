import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

# Optional MLflow import – used only when explicitly called
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def train_crop_recommendation_model():
    """Train and save the crop recommendation model."""
    print("Loading dataset...")
    crop = pd.read_csv("Crop_recommendation.csv")

    # Check for missing values
    if crop.isnull().sum().sum() > 0:
        print(f"Found {int(crop.isnull().sum().sum())} missing values. Cleaning data...")
        crop = crop.dropna()

    # Create label encoding for 12 available crops
    crop_dict = {
        'rice': 1, 'maize': 2, 'chickpea': 3, 'banana': 4, 'mango': 5, 'grapes': 6,
        'watermelon': 7, 'apple': 8, 'orange': 9, 'cotton': 10, 'jute': 11, 'coffee': 12
    }

    inverse_crop_dict = {v: k for k, v in crop_dict.items()}
    crop['label'] = crop['label'].map(crop_dict)

    # Split features and target
    X = crop.drop('label', axis=1)
    y = crop['label']

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Applying feature scaling...")
    minmax = MinMaxScaler()
    X_train_minmax = minmax.fit_transform(X_train)
    X_test_minmax = minmax.transform(X_test)

    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train_minmax)
    X_test_scaled = std_scaler.transform(X_test_minmax)

    print("Training Random Forest model with cross-validation...")
    base_model = RandomForestClassifier(random_state=42)
    cv_scores = cross_val_score(base_model, X_train_scaled, y_train, cv=5)
    print(f"Cross Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f}")

    print("Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 25, 35, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nFeature Importances:")
    feature_names = X.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(len(feature_names)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # Optional: log to MLflow if available
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("crop-recommendation")
            with mlflow.start_run(run_name="random-forest-v1"):
                mlflow.log_param("n_estimators", best_params["n_estimators"])
                mlflow.log_param("max_depth", best_params["max_depth"])
                mlflow.log_param("test_size", 0.2)
                mlflow.log_param("random_state", 42)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("cv_mean_score", cv_scores.mean())
                mlflow.log_metric("cv_std_score", cv_scores.std())
                mlflow.sklearn.log_model(best_model, "model")

                importances_df = pd.DataFrame({
                    "feature": X.columns,
                    "importance": best_model.feature_importances_
                }).sort_values("importance", ascending=False)
                importances_df.to_csv("feature_importances.csv", index=False)
                mlflow.log_artifact("feature_importances.csv")

                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("num_classes", len(crop_dict))
        except Exception as e:
            print(f"MLflow logging skipped: {e}")

    print("\nSaving model and preprocessing objects...")
    joblib.dump(best_model, 'crop_recommendation_model.pkl')
    joblib.dump(minmax, 'minmax_scaler.pkl')
    joblib.dump(std_scaler, 'standard_scaler.pkl')
    joblib.dump(inverse_crop_dict, 'crop_labels.pkl')

    print("Model training completed successfully!")
    return best_model, minmax, std_scaler, inverse_crop_dict


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Predict the best crop given soil and weather parameters."""
    model = None
    minmax = None
    std_scaler = None
    crop_labels = None

    # Try loading from MLflow registry first
    if MLFLOW_AVAILABLE:
        try:
            model = mlflow.sklearn.load_model("models:/CropRecommender/Production")
        except Exception:
            pass  # fall through to pkl loading

    # Fallback to pkl files
    if model is None:
        try:
            model = joblib.load('crop_recommendation_model.pkl')
            minmax = joblib.load('minmax_scaler.pkl')
            std_scaler = joblib.load('standard_scaler.pkl')
            crop_labels = joblib.load('crop_labels.pkl')
        except FileNotFoundError:
            print("Model artifacts not found. Training a new model...")
            model, minmax, std_scaler, crop_labels = train_crop_recommendation_model()

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_minmax = minmax.transform(features)
    features_scaled = std_scaler.transform(features_minmax)

    # Get scalar prediction and probability vector for the single sample
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    max_prob = float(np.max(probabilities)) * 100.0

    # Look up crop name with an int key from the inverse mapping
    crop_name = crop_labels[int(prediction[0])]

    return crop_name, max_prob


if __name__ == "__main__":
    print("Starting training via CLI...")
    train_crop_recommendation_model()

    test_cases = [
        {
            'N': 90, 'P': 42, 'K': 43, 'temperature': 20.87,
            'humidity': 82.00, 'ph': 6.5, 'rainfall': 202.94
        },
        {
            'N': 20, 'P': 30, 'K': 10, 'temperature': 26.5,
            'humidity': 52.5, 'ph': 7.0, 'rainfall': 150.25
        }
    ]

    print("\nTesting prediction with sample data:")
    for i, test in enumerate(test_cases):
        crop_name, probability = predict_crop(
            test['N'], test['P'], test['K'], test['temperature'],
            test['humidity'], test['ph'], test['rainfall']
        )
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {test}")
        print(f"Predicted Crop: {crop_name}")
        print(f"Confidence: {probability:.2f}%")
