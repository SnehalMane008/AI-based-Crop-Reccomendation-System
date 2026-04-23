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

# i18n: use Flask-Babel's gettext for user-facing messages
from flask_babel import gettext as _

warnings.filterwarnings('ignore')

def train_crop_recommendation_model():
    """Train and save the crop recommendation model."""
    print(_("Loading dataset..."))
    crop = pd.read_csv("Crop_recommendation.csv")

    # Check for missing values
    if crop.isnull().sum().sum() > 0:
        print(_("Found %(n)d missing values. Cleaning data...", n=int(crop.isnull().sum().sum())))
        crop = crop.dropna()

    # Create label encoding for 12 available crops (keys are programmatic; do not translate)
    crop_dict = {
        'rice': 1, 'maize': 2, 'chickpea': 3, 'banana': 4, 'mango': 5, 'grapes': 6,
        'watermelon': 7, 'apple': 8, 'orange': 9, 'cotton': 10, 'jute': 11, 'coffee': 12
    }

    inverse_crop_dict = {v: k for k, v in crop_dict.items()}
    crop['label'] = crop['label'].map(crop_dict)

    # Split features and target
    X = crop.drop('label', axis=1)
    y = crop['label']

    print(_("Splitting data into training and testing sets..."))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(_("Applying feature scaling..."))
    minmax = MinMaxScaler()
    X_train_minmax = minmax.fit_transform(X_train)
    X_test_minmax = minmax.transform(X_test)

    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train_minmax)
    X_test_scaled = std_scaler.transform(X_test_minmax)

    print(_("Training Random Forest model with cross-validation..."))
    base_model = RandomForestClassifier(random_state=42)
    cv_scores = cross_val_score(base_model, X_train_scaled, y_train, cv=5)
    print(_("Cross Validation Scores: %(scores)s", scores=cv_scores))
    print(_("Mean CV Score: %(score).4f", score=cv_scores.mean()))

    print(_("Performing hyperparameter tuning..."))
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

    print(_("Best Parameters: %(params)s", params=grid_search.best_params_))
    print(_("Best CV Score: %(score).4f", score=grid_search.best_score_))

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(_("Test Accuracy: %(acc).4f", acc=test_accuracy))
    print(_("\nClassification Report:"))
    print(classification_report(y_test, y_pred))

    print(_("\nFeature Importances:"))
    feature_names = X.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(len(feature_names)):
        print(_("%(name)s: %(imp).4f", name=feature_names[indices[i]], imp=importances[indices[i]]))

    print(_("\nSaving model and preprocessing objects..."))
    joblib.dump(best_model, 'crop_recommendation_model.pkl')
    joblib.dump(minmax, 'minmax_scaler.pkl')
    joblib.dump(std_scaler, 'standard_scaler.pkl')
    joblib.dump(inverse_crop_dict, 'crop_labels.pkl')

    print(_("Model training completed successfully!"))
    return best_model, minmax, std_scaler, inverse_crop_dict

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict the recommended crop based on soil and climate parameters.
    Returns:
        Tuple(str, float): Recommended crop name, Confidence %
    """
    try:
        model = joblib.load('crop_recommendation_model.pkl')
        minmax = joblib.load('minmax_scaler.pkl')
        std_scaler = joblib.load('standard_scaler.pkl')
        crop_labels = joblib.load('crop_labels.pkl')
    except FileNotFoundError:
        print(_("Model artifacts not found. Training a new model..."))
        model, minmax, std_scaler, crop_labels = train_crop_recommendation_model()

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_minmax = minmax.transform(features)
    features_scaled = std_scaler.transform(features_minmax)

    # Get scalar prediction and probability vector for the single sample
    prediction = model.predict(features_scaled)           # (n_samples,) array
    probabilities = model.predict_proba(features_scaled)  # (n_samples, n_classes) array
    max_prob = float(np.max(probabilities)) * 100.0          # ensure Python float for formatting

    # Look up crop name with an int key from the inverse mapping
    crop_name = crop_labels[int(prediction[0])]  # Extract scalar from prediction array  

    return crop_name, max_prob

if __name__ == "__main__":
    print(_("Starting training via CLI..."))
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

    print(_("\nTesting prediction with sample data:"))
    for i, test in enumerate(test_cases):
        crop_name, probability = predict_crop(
            test['N'], test['P'], test['K'], test['temperature'],
            test['humidity'], test['ph'], test['rainfall']
        )
        print(_("\nTest Case %(n)d:", n=i + 1))
        print(_("Input: %(inp)s", inp=test))
        print(_("Predicted Crop: %(crop)s", crop=crop_name))
        print(_("Confidence: %(p).2f%%", p=probability))
