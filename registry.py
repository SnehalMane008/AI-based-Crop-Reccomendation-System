# registry.py
import mlflow
from mlflow.tracking import MlflowClient

def register_model(run_id: str, model_name: str = "CropRecommender"):
    client = MlflowClient()
    
    # Register model from a run
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    
    print(f"Registered: {model_name} version {mv.version}")
    return mv.version

def promote_to_production(model_name: str, version: int):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f"Model version {version} promoted to Production")

def load_production_model(model_name: str = "CropRecommender"):
    """Load the current production model instead of pkl files"""
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    return model