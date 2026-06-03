import joblib
from pathlib import Path


def load_model(project_name: str, model_name: str):
    """Load a model from a project's models directory."""
    path = Path(__file__).parent.parent / "projects" / project_name / "models" / f"{model_name}.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    model = joblib.load(path)
    print(f"[OK] Model '{model_name}' from project '{project_name}' loaded successfully.")
    return model
