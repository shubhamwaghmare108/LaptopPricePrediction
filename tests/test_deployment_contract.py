import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSOR = ROOT / "artifacts" / "transformed" / "preprocessor.joblib"
MODEL = ROOT / "artifacts" / "model" / "best_model.joblib"
FEATURES = ROOT / "artifacts" / "transformed" / "feature_list.json"
TRAIN = ROOT / "artifacts" / "transformed" / "train.csv"


def test_required_serving_artifacts_exist():
    required = (PREPROCESSOR, MODEL, FEATURES, TRAIN)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    assert not missing, f"Missing serving artifacts: {missing}"


def test_serving_artifacts_can_be_loaded():
    preprocessor = joblib.load(PREPROCESSOR)
    model = joblib.load(MODEL)
    assert hasattr(preprocessor, "transform")
    assert hasattr(model, "predict")


def test_training_schema_matches_feature_metadata():
    with FEATURES.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_names = metadata.get("num_cols", []) + metadata.get("cat_cols", [])
    train_columns = pd.read_csv(TRAIN, nrows=0).columns.tolist()
    assert feature_names
    assert set(feature_names).issubset(set(train_columns))
    assert "Price_INR" in train_columns
