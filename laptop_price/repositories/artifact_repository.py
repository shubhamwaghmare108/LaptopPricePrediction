"""Centralized filesystem access and cached loading for serving artifacts."""
import os
from functools import lru_cache
from pathlib import Path
import joblib
from laptop_price.config import PROJECT_ROOT
from laptop_price.exception import PricePredictorException

def _path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

MODEL_PATH = _path("MODEL_PATH", PROJECT_ROOT / "prediction" / "models" / "current_model.joblib")
PREPROCESSOR_PATH = _path("PREPROCESSOR_PATH", PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib")
FEATURE_LIST_PATH = _path("FEATURE_LIST_PATH", PROJECT_ROOT / "artifacts" / "transformed" / "feature_list.json")
TRAIN_CSV_PATH = _path("TRAIN_CSV_PATH", PROJECT_ROOT / "artifacts" / "transformed" / "train.csv")

def load_object(path: Path):
    path = Path(path)
    if not path.is_file():
        raise PricePredictorException(f"Artifact not found: {path}")
    try:
        return joblib.load(path)
    except Exception as error:
        raise PricePredictorException(f"Unable to load artifact {path}: {error}") from error

@lru_cache(maxsize=1)
def load_prediction_artifacts():
    """Load one compatible model/preprocessor pair per backend process."""
    return load_object(MODEL_PATH), load_object(PREPROCESSOR_PATH)
