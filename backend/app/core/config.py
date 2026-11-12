"""Environment-based backend configuration; intentionally UI-independent."""
import os
from pathlib import Path
from laptop_price.config import PROJECT_ROOT

def resolve_path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

MODEL_PATH = resolve_path("MODEL_PATH", PROJECT_ROOT / "prediction/models/current_model.joblib")
PREPROCESSOR_PATH = resolve_path("PREPROCESSOR_PATH", PROJECT_ROOT / "artifacts/transformed/preprocessor.joblib")
FEATURE_LIST_PATH = resolve_path("FEATURE_LIST_PATH", PROJECT_ROOT / "artifacts/transformed/feature_list.json")
TRAIN_CSV_PATH = resolve_path("TRAIN_CSV_PATH", PROJECT_ROOT / "artifacts/transformed/train.csv")
CORS_ORIGINS = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if x.strip()]
