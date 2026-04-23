# KRUSHIMITRA - AI-Based Crop Recommendation System (MLOps)

**जय जवान जय किसान**

## 🌾 Project Overview
KRUSHIMITRA is an advanced AI-powered system designed to provide accurate crop recommendations based on soil health and climate factors. It has been upgraded to a full **MLOps (Machine Learning Operations)** pipeline, ensuring high reliability, experiment tracking, and automated model maintenance.

## 🚀 MLOps Features
This project implements a complete lifecycle for machine learning:
- **Experiment Tracking**: Integrated with **MLflow** to track metrics, parameters, and model versions.
- **Automated Validation**: Robust data validation engine (replaces Great Expectations for Python 3.14+ compatibility).
- **Model Registry**: Management of production-ready models.
- **Monitoring**: Real-time prediction logging and **data drift detection** to ensure model accuracy over time.
- **Retraining Pipeline**: Automated scripts to retrain the model when new data accumulates or drift is detected.
- **Containerization**: Fully dockerized environment for consistent deployment.

## 🛠️ Technology Stack
- **Backend**: Python 3.10+, Flask
- **Machine Learning**: Scikit-learn (Random Forest), Pandas, NumPy
- **MLOps**: MLflow, DVC, Pytest
- **Frontend**: HTML5, CSS3 (Modern Glassmorphism UI), JavaScript
- **DevOps**: Docker, Docker Compose

## 📦 Installation & Local Setup

### 1. Requirements
*   Python 3.10 or 3.14+
*   Docker & Docker Compose (Recommended)

### 2. Manual Installation
```bash
# Clone the repository
git clone <repo-url>
cd AI-based-Crop-Reccomendation-System

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```
*The app will be available at `http://localhost:5001`.*

## 🐳 Docker Deployment (Recommended)
Deployment is simplified using Docker Compose to handle the App, MLflow, and Monitoring services.

```bash
# Start all services
docker-compose up --build -d
```

### 🔗 Service Links (Docker)
| Service | Link | Description |
| :--- | :--- | :--- |
| **Main App** | [http://localhost:5002](http://localhost:5002) | User interface for crop prediction. |
| **MLflow UI** | [http://localhost:5003](http://localhost:5003) | Track experiments and view model performance. |

## 🧪 Testing & Validation
To ensure system integrity, run the automated test suite:

```bash
# Run Model & Prediction tests
python -m pytest tests/test_model.py -v

# Run Data Validation
python tests/validate_data.py
```

## 📈 Model Performance
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~98.75%
- **Features**: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall.

## 🏗️ Project Structure
```
├── app.py                # Flask application & API routes
├── model.py              # ML training & MLflow logging logic
├── monitoring/           # Drift detection & prediction logs
├── pipelines/            # Automated retraining workflows
├── registry.py           # MLflow model promotion logic
├── tests/                # Unit tests & data validation
├── static/ & templates/  # Modern UI assets
├── Dockerfile            # Container configuration
└── docker-compose.yml    # Multi-container orchestration
```

---
*Built with ❤️ for Indian Farmers.*
