"""Centralized filesystem access for trained model artifacts."""
from pathlib import Path
import joblib
from laptop_price.config import PROJECT_ROOT
from laptop_price.exception import PricePredictorException
MODEL_PATH = PROJECT_ROOT / "prediction" / "models" / "current_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
FEATURE_LIST_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "feature_list.json"
TRAIN_CSV_PATH = PROJECT_ROOT / "artifacts" / "transformed" / "train.csv"
def load_object(path: Path):
    path = Path(path)
    if not path.is_file():
        raise PricePredictorException(f"Artifact not found: {path}")
    try:
        return joblib.load(path)
    except Exception as error:
        raise PricePredictorException(f"Unable to load artifact {path}: {error}") from error
def load_prediction_artifacts():
    return load_object(MODEL_PATH), load_object(PREPROCESSOR_PATH)
